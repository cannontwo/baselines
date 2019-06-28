import numpy as np
import tensorflow as tf
from baselines.a2c import utils
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch
from baselines.common.mpi_running_mean_std import RunningMeanStd

### SkNet imports
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


mapping = {}

def register(name):
    def _thunk(func):
        mapping[name] = func
        return func
    return _thunk

def nature_cnn(unscaled_images, **conv_kwargs):
    """
    CNN from Nature paper.
    """
    scaled_images = tf.cast(unscaled_images, tf.float32) / 255.
    activ = tf.nn.relu
    h = activ(conv(scaled_images, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2),
                   **conv_kwargs))
    h2 = activ(conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2), **conv_kwargs))
    h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2), **conv_kwargs))
    h3 = conv_to_fc(h3)
    return activ(fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2)))

def build_impala_cnn(unscaled_images, depths=[16,32,32], **conv_kwargs):
    """
    Model used in the paper "IMPALA: Scalable Distributed Deep-RL with
    Importance Weighted Actor-Learner Architectures" https://arxiv.org/abs/1802.01561
    """

    layer_num = 0

    def get_layer_num_str():
        nonlocal layer_num
        num_str = str(layer_num)
        layer_num += 1
        return num_str

    def conv_layer(out, depth):
        return tf.layers.conv2d(out, depth, 3, padding='same', name='layer_' + get_layer_num_str())

    def residual_block(inputs):
        depth = inputs.get_shape()[-1].value

        out = tf.nn.relu(inputs)

        out = conv_layer(out, depth)
        out = tf.nn.relu(out)
        out = conv_layer(out, depth)
        return out + inputs

    def conv_sequence(inputs, depth):
        out = conv_layer(inputs, depth)
        out = tf.layers.max_pooling2d(out, pool_size=3, strides=2, padding='same')
        out = residual_block(out)
        out = residual_block(out)
        return out

    out = tf.cast(unscaled_images, tf.float32) / 255.

    for depth in depths:
        out = conv_sequence(out, depth)

    out = tf.layers.flatten(out)
    out = tf.nn.relu(out)
    out = tf.layers.dense(out, 256, activation=tf.nn.relu, name='layer_' + get_layer_num_str())

    return out


@register("mlp")
def mlp(num_layers=2, num_hidden=64, activation=tf.tanh, layer_norm=False):
    """
    Stack of fully-connected layers to be used in a policy / q-function approximator

    Parameters:
    ----------

    num_layers: int                 number of fully-connected layers (default: 2)

    num_hidden: int                 size of fully-connected layers (default: 64)

    activation:                     activation function (default: tf.tanh)

    Returns:
    -------

    function that builds fully connected network with a given input tensor / placeholder
    """
    ######################
    # SkNet functions
    ######################

    # Number of grid points
    N = 350
    # Grid
    TIME = np.linspace(-7,7,N)
    X = np.meshgrid(TIME, TIME)
    X = np.stack([X[0].flatten(), X[1].flatten()], 1).astype('float32')


    def states2values(states, state2value_dict=None, return_dict=False):
        """Given binary masks obtained for example from the ReLU activation, 
        convert the binary vectors to a single float scalar, the same scalar for
        the same masks. This allows to drastically reduce the memory overhead to
        keep track of a large number of masks. The mapping mask -> real is kept
        inside a dict and thus allow to go from one to another. Thus given a large
        collection of masks, one ends up with a large collection of scalars and
        a mapping mask <-> real only for unique masks and thus allow reduced memory
        requirements.
        Parameters
        ----------
        states : bool matrix
            the matrix of shape (#samples,#binary values). Thus if using a deep net
            the masks of ReLU for all the layers must first be flattened prior
            using this function.
        state2value_dict : dict
            optional dict containing an already built mask <-> real mapping which
            should be used and updated given the states value.
        return_dict : bool
            if the update/created dict should be return as part of the output
        Returns
        -------
        values : scalar vector
            a vector of length #samples in which each entry is the scalar
            representation of the corresponding mask from states
        state2value_dict : dict (optional)
            the newly created or updated dict mapping mask to value and vice-versa.
        """
        if state2value_dict is None:
            state2value_dict = dict()
        if 'count' not in state2value_dict:
            state2value_dict['count']=0
        values = np.zeros(states.shape[0])
        for i,state in enumerate(states):
            str_s = ''.join(state.astype('uint8').astype('str'))
            if(str_s not in state2value_dict):
                state2value_dict[str_s]   = state2value_dict['count']
                state2value_dict['count']+= 0.001
            values[i] = state2value_dict[str_s]
        return values

    def grad(x,duplicate=0):
        #compute each directional (one step) derivative as a boolean mask
        #representing jump from one region to another and add them (boolean still)
        g_vertical   = np.greater(np.pad(np.abs(x[1:]-x[:-1]),((1,0),(0,0)),'constant'),0)
        g_horizontal = np.greater(np.pad(np.abs(x[:,1:]-x[:,:-1]),[[0,0],[1,0]],'constant'),0)
        g_diagonaldo = np.greater(np.pad(np.abs(x[1:,1:]-x[:-1,:-1]),[[1,0],[1,0]],'constant'),0)
        g_diagonalup = np.greater(np.pad(np.abs(x[:-1:,1:]-x[1:,:-1]),[[1,0],[1,0]],'constant'),0)
        overall      = g_vertical+g_horizontal+g_diagonaldo+g_diagonalup
        if duplicate>0:
            overall                 = np.stack([np.roll(overall,k,1) for k in range(duplicate+1)]\
                                        +[np.roll(overall,k,0) for k in range(duplicate+1)]).sum(0)
            overall[:duplicate]    *= 0
            overall[-duplicate:]   *= 0
            overall[:,:duplicate]  *= 0
            overall[:,-duplicate:] *= 0
        return np.greater(overall,0).astype('float32')

    def get_input_space_partition(states,N,M,duplicate=1):
        #the following should take as input a collection of points
        #in the input space and return a list of binary states, each 
        #element of the list is for 1 specific layer and it is a 2D array
        if states.ndim>1:
            states = states2values(states)
        partitioning  = grad(states.reshape((N,M)).astype('float32'),duplicate)
        return partitioning
    ######################
    # End SkNet functions
    ######################

    def network_fn(X):
        outputs = []
        h = tf.layers.flatten(X)
        for i in range(num_layers):
            h = fc(h, 'mlp_fc{}'.format(i), nh=num_hidden, init_scale=np.sqrt(2))
            if layer_norm:
                h = tf.contrib.layers.layer_norm(h, center=True, scale=True)
            h = activation(h)
            outputs.append(h)

        return outputs

    return network_fn


@register("cnn")
def cnn(**conv_kwargs):
    def network_fn(X):
        return nature_cnn(X, **conv_kwargs)
    return network_fn

@register("impala_cnn")
def impala_cnn(**conv_kwargs):
    def network_fn(X):
        return build_impala_cnn(X)
    return network_fn

@register("cnn_small")
def cnn_small(**conv_kwargs):
    def network_fn(X):
        h = tf.cast(X, tf.float32) / 255.

        activ = tf.nn.relu
        h = activ(conv(h, 'c1', nf=8, rf=8, stride=4, init_scale=np.sqrt(2), **conv_kwargs))
        h = activ(conv(h, 'c2', nf=16, rf=4, stride=2, init_scale=np.sqrt(2), **conv_kwargs))
        h = conv_to_fc(h)
        h = activ(fc(h, 'fc1', nh=128, init_scale=np.sqrt(2)))
        return h
    return network_fn

@register("lstm")
def lstm(nlstm=128, layer_norm=False):
    """
    Builds LSTM (Long-Short Term Memory) network to be used in a policy.
    Note that the resulting function returns not only the output of the LSTM
    (i.e. hidden state of lstm for each step in the sequence), but also a dictionary
    with auxiliary tensors to be set as policy attributes.

    Specifically,
        S is a placeholder to feed current state (LSTM state has to be managed outside policy)
        M is a placeholder for the mask (used to mask out observations after the end of the episode, but can be used for other purposes too)
        initial_state is a numpy array containing initial lstm state (usually zeros)
        state is the output LSTM state (to be fed into S at the next call)


    An example of usage of lstm-based policy can be found here: common/tests/test_doc_examples.py/test_lstm_example

    Parameters:
    ----------

    nlstm: int          LSTM hidden state size

    layer_norm: bool    if True, layer-normalized version of LSTM is used

    Returns:
    -------

    function that builds LSTM with a given input tensor / placeholder
    """

    def network_fn(X, nenv=1):
        nbatch = X.shape[0]
        nsteps = nbatch // nenv

        h = tf.layers.flatten(X)

        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, 2*nlstm]) #states

        xs = batch_to_seq(h, nenv, nsteps)
        ms = batch_to_seq(M, nenv, nsteps)

        if layer_norm:
            h5, snew = utils.lnlstm(xs, ms, S, scope='lnlstm', nh=nlstm)
        else:
            h5, snew = utils.lstm(xs, ms, S, scope='lstm', nh=nlstm)

        h = seq_to_batch(h5)
        initial_state = np.zeros(S.shape.as_list(), dtype=float)

        return h, {'S':S, 'M':M, 'state':snew, 'initial_state':initial_state}

    return network_fn


@register("cnn_lstm")
def cnn_lstm(nlstm=128, layer_norm=False, conv_fn=nature_cnn, **conv_kwargs):
    def network_fn(X, nenv=1):
        nbatch = X.shape[0]
        nsteps = nbatch // nenv

        h = conv_fn(X, **conv_kwargs)

        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, 2*nlstm]) #states

        xs = batch_to_seq(h, nenv, nsteps)
        ms = batch_to_seq(M, nenv, nsteps)

        if layer_norm:
            h5, snew = utils.lnlstm(xs, ms, S, scope='lnlstm', nh=nlstm)
        else:
            h5, snew = utils.lstm(xs, ms, S, scope='lstm', nh=nlstm)

        h = seq_to_batch(h5)
        initial_state = np.zeros(S.shape.as_list(), dtype=float)

        return h, {'S':S, 'M':M, 'state':snew, 'initial_state':initial_state}

    return network_fn

@register("impala_cnn_lstm")
def impala_cnn_lstm():
    return cnn_lstm(nlstm=256, conv_fn=build_impala_cnn)

@register("cnn_lnlstm")
def cnn_lnlstm(nlstm=128, **conv_kwargs):
    return cnn_lstm(nlstm, layer_norm=True, **conv_kwargs)


@register("conv_only")
def conv_only(convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)], **conv_kwargs):
    '''
    convolutions-only net

    Parameters:
    ----------

    conv:       list of triples (filter_number, filter_size, stride) specifying parameters for each layer.

    Returns:

    function that takes tensorflow tensor as input and returns the output of the last convolutional layer

    '''

    def network_fn(X):
        out = tf.cast(X, tf.float32) / 255.
        with tf.variable_scope("convnet"):
            for num_outputs, kernel_size, stride in convs:
                out = tf.contrib.layers.convolution2d(out,
                                           num_outputs=num_outputs,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           activation_fn=tf.nn.relu,
                                           **conv_kwargs)

        return out
    return network_fn

def _normalize_clip_observation(x, clip_range=[-5.0, 5.0]):
    rms = RunningMeanStd(shape=x.shape[1:])
    norm_x = tf.clip_by_value((x - rms.mean) / rms.std, min(clip_range), max(clip_range))
    return norm_x, rms


def get_network_builder(name):
    """
    If you want to register your own network outside models.py, you just need:

    Usage Example:
    -------------
    from baselines.common.models import register
    @register("your_network_name")
    def your_network_define(**net_kwargs):
        ...
        return network_fn

    """
    if callable(name):
        return name
    elif name in mapping:
        return mapping[name]
    else:
        raise ValueError('Unknown network type: {}'.format(name))
