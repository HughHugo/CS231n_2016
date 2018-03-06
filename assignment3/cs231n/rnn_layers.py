import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of vanilla RNN that uses a tanh activation function.
    
    The input data has dimension D, the hidden state has dimension H, and we use a minibatch size of N.
    
    Inputs:
    - x: Input data for this timestep, of shape (N, D);
    - prev_h: Hidden state from previous timestep, of shape (N, H);
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H);
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H);
    - b: Biases of shape (H, ).
    
    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H);
    - cache: Tuple of values needed for the backward pass.
    """
    affine_output = x.dot(Wx) + prev_h.dot(Wh) + b
    next_h = np.tanh(affine_output)
    cache = (x, prev_h, Wx, Wh, affine_output, next_h)
    
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.
    
    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state;
    - cache: Cache object from the forward apss.
    
    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D);
    - dprev_h: Gradients of previous hidden state, of shape (N, H);
    - dWx: Gradients of input-to-hidden weights, of shape (N, H);
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H);
    - db: Gradients of bias vector, of shape (H, ).
    """
    x, prev_h, Wx, Wh, affine_output, next_h = cache
    
    daffine_output = dnext_h * (1 - next_h * next_h)
    dWx = x.T.dot(daffine_output)
    dx = daffine_output.dot(Wx.T)
    dWh = prev_h.T.dot(daffine_output)
    dprev_h = daffine_output.dot(Wh.T)
    db = np.sum(daffine_output, axis = 0)
    
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input sequence composed of T vectors, each of 
    dimension D. The RNN uses a hidden size of H, and we work over a minibatch containing N sequences. After running the 
    RNN forward, we return the hidden states for all timesteps.
    
    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D);
    - h0: Initial hidden state, of shape (N, H);
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H);
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H);
    - b: Biases of shape (H, ).
    
    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H);
    - cache: Values needed in the backward pass.
    """
    N, T, D = x.shape
    _, H = h0.shape
    h = np.zeros((N, T, H))
    prev_h = h0
    cache = {}
    
    for t in range(T):
        h[:, t, :], cache[t] = rnn_step_forward(x[:, t, :], prev_h, Wx, Wh, b)
        prev_h = h[:, t, :]
        
    return h, cache


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of datea.
    
    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H).
    
    Returns a tuple of:
    - dx: Gradient of inputs, of shape (N, T, D);
    - dh0: Gradient of initial hidden state, of shape (N, H);
    - dWx: Gradient of input-to-hidden weights, of shape (D, H);
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H);
    - db: Gradient of biases, of shape (H, ).
    """
    N, T, H = dh.shape
    _, D = cache[0][0].shape
    dx = np.zeros((N, T, D))
    dh0 = np.zeros((N, H))
    dWx = np.zeros((D, H))
    dWh = np.zeros((H, H))
    db = np.zeros((H, ))
    dprev_h = np.zeros((N, H))
    for t in range(T - 1, -1, -1):
        dx_step, dprev_h, dWx_step, dWh_step, db_step = rnn_step_backward(dh[:, t, :] + dprev_h, cache[t])
        dx[:, t, :] += dx_step
        dWx += dWx_step
        dWh += dWh_step
        db += db_step
    dh0 = dprev_h
    
    return dx, dh0, dWx, dWh, db



def word_embedding_forward(x, W):
    """
    Forward pass for word embeddings. We operate on minibatches of size N where each sequence has length T. We assume a 
    vocabulary of V words, assigning each to a vector of dimension D.
    
    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx of x must be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.
    
    Returns:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass.
    """
    out = W[x, :]
    cache = (x, W)
    
    return out, cache



def word_embedding_backward(dout, cache):
    """
    Backward pass for word embeddings. We cannot back-propagate into the words since they are integers, so we only return 
    gradient for the word embedding matrix.
    
    HINT: Look up the function np.add.at.
    
    Inputs:
    - dout: Uptream gradients of shape (N, T, D);
    - cache: Values from the forward pass.
    
    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D).
    """
    x, W = cache
    dW = np.zeros_like(W)
    np.add.at(dW, x, dout)
    
    return dW


def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.
    The input data has dimension D, the hidden state has dimension H, and we use a minibatch of N.
    
    Inputs:
    - x: Input data, of shape (N, D);
    - prev_h: Previous hidden state, of shape (N, H);
    - prev_c: Previous cell state, of shape (N, H);
    - Wx: Input-to-hidden weights, of shape (D, 4H);
    - Wh: Hidden-to-hidden weights, of shape (H, 4H);
    - b: Biases, of shape (4H, ).
    
    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H);
    - next_c: Next cell state of shape (N, H);
    - cache: Tupleof values needed for backward pass.
    """
    N, D = x.shape
    _, H = prev_h.shape
    a = x.dot(Wx) + prev_h.dot(Wh) + b
    i = sigmoid(a[:, :H])
    f = sigmoid(a[:, H: 2*H])
    o = sigmoid(a[:, 2*H: 3*H])
    g = np.tanh(a[:, 3*H:])
    
    next_c = f * prev_c + i * g
    next_h = o * np.tanh(next_c)
    cache = (prev_c, prev_h, x, Wx, Wh, i, f, o, g, next_c)
    
    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Backward pass for a single timestep of an LSTM.
    
    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H);
    - dnext_c: Gradients of next cell state, of shape (N, H);
    - cache: Values from the forward pass.
    
    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D);
    - dprev_h: Gradient of previous hidden state, of shape (N, H);
    - dprev_c: Gradient of previous cell state, of shape (N, H);
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H);
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H);
    - db: Gradient of biases, of shape (4H, ).
    """
    prev_c, prev_h, x, Wx, Wh, i, f, o, g, next_c = cache
    N, H = dnext_h.shape
    _, D = x.shape
    
    do = dnext_h * np.tanh(next_c)
    dnext_c += dnext_h * o * (1 - np.tanh(next_c) * np.tanh(next_c))
    dprev_c = dnext_c * f
    df, di, dg = dnext_c * prev_c, dnext_c * g, dnext_c * i
    da = np.hstack((di * i * (1 - i), df * f * (1 - f), do * o * (1 - o), dg * (1 - g * g)))
    dx = da.dot(Wx.T)
    dWx = x.T.dot(da)
    dprev_h = da.dot(Wh.T)
    dWh = prev_h.T.dot(da)
    db = np.sum(da, axis = 0)
    
    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
    """
    Forward pass for an LSTM over an entire sequence of data. We assume an input sequence composed of T vectors, each of 
    dimension D. The LSTM uses a hidden size of H, and we work over a minibatch containing N sequences. After running the LSTM 
    forward, we return the hidden states for all timesteps.
    
    Note that the initial cell state is passed as input, but the initial cell state is set to zero. Also note that the cell 
    state is not returned; it is an internal variable to the LSTM and is not accessed from outside.
    
    Inputs:
    - x: Input data of shape (N, T, D);
    - h0: Initial hidden state of shape (N, H);
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H);
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H);
    - b: Biases of shape (4H, ).
    
    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H);
    - cache: Values needed for the backward pass.
    """
    N, T, D = x.shape
    _, H = h0.shape
    prev_c = np.zeros((N, H))
    prev_h = h0
    h = np.zeros((N, T, H))
    cache = {}
    for t in range(T):
        next_h, next_c, cache[t] = lstm_step_forward(x[:, t, :], prev_h, prev_c, Wx, Wh, b)
        h[:, t, :] = next_h
        prev_h, prev_c = next_h, next_c
    
    return h, cache


def lstm_backward(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.
    
    Inputs:
    - dh: Upstream gradients of hidden states, of shape (N, T, H);
    - cache: Values from the forward pass.
    
    Returns a tuple of:
    - dx: Gradient of input data of shape (N, T, D);
    - dh0: Gradient of initial hidden state of shape (N, H);
    - dWx: Gradient of input-to-hidden weights matrix of shape (D, 4H);
    - dWh: Gradient of hidden-to-hidden weights matrix of shape (H, 4H);
    - db: Gradient of biases, of shape (4H, ).
    """
    N, T, H = dh.shape
    _, D = cache[0][2].shape
    dprev_c = np.zeros((N, H))
    dprev_h = np.zeros((N, H))
    dx = np.zeros((N, T, D))
    dWx = np.zeros((D, 4 * H))
    dWh = np.zeros((H, 4 * H))
    db = np.zeros((4 * H, ))
    for t in range(T-1, -1, -1):
        dx[:, t, :], dprev_h, dprev_c, dWx_step, dWh_step, db_step = lstm_step_backward(dh[:, t, :] + dprev_h, dprev_c, cache[t])
        dWx += dWx_step
        dWh += dWh_step
        db += db_step
    dh0 = dprev_h
    
    return dx, dh0, dWx, dWh, db











def temporal_affine_forward(x, w, b):
    """
    Forward pass for a temporal affine layer. The input is a set of D-dimensional vectors arranged into a minibatch of 
    N timeseries, each of length T. We use an affine function to transform each of those vectors into a new vector of 
    dimension M.
    
    Inputs:
    - x: Input data of shape (N, T, D);
    - w: Weights of shape (D, M);
    - b: Biases of shape (M, ).
    
    Returns a tuple of:
    - out: Output data of shape (N, T, M);
    - cache: Values needed for the backward pass.
    """
    N, T, D = x.shape
    _, M = w.shape
    out = x.reshape(-1, D).dot(w).reshape(N, T, M) + b
    cache = (x, w, b, out)
    return out, cache


def temporal_affine_backward(dout, cache):
    """
    Backward pass for temporal affine layer.
    
    Inputs:
    - dout: Upstream gradients of shape (N, T, M);
    - cache: Values from forward pass.
    
    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D);
    - dw: Gradient of weights, of shape (D, M);
    - db: Gradient of biases, of shape (M, ).
    """
    x, w, b, out = cache
    N, T, D = x.shape
    _, M = w.shape
    dx = dout.reshape(-1, M).dot(w.T).reshape(N, T, D)
    dw = x.reshape(-1, D).T.dot(dout.reshape(-1, M))
    db = np.sum(dout, axis= (0, 1))
    
    return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose = False):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are making predictions over a vocabulary of size V
    for each timestep of a timeseries of length T, over a minibatch of size N. The input x gives scores for all vocabulary 
    elements all all timesteps, and y gives the indices of the ground-truth element at each timestep. We use a cross-entropy 
    loss at each timestep, summing the loss over all timesteps and averaging across the minibatch.
    
    As an additional complication, we may want to ignore the model output at some timesteps, since sequences of different 
    length may have been combined into a minibatch and padded with NULL tokens. The optional mask argument tells us which 
    elements should contribute to the loss.
    
    Inputs:
    - x: Input scores, of shape (N, T, V);
    - y: Ground-truth indices, of shape (N, T) where each element is in the range <= y[i, t] < V;
    - mask: Boolean array of shape (N, T) where mask [i, t] tells whether or not the scores at x[i, t] should contribute to the
    loss.
    
    Returns a tuple of:
    - loss: Scalar giving loss;
    - dx: Gradient of loss with respect to scores x.
    """
    N, T, V = x.shape
    x_flat = x.reshape(-1, V)
    y_flat = y.reshape(-1)
    mask_flat = mask.reshape(-1)
    
    probs = np.exp(x_flat - np.max(x_flat, axis = 1, keepdims = True))
    probs /= np.sum(probs, axis = 1, keepdims = True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]
    
    if verbose :
        print('dx_flat:', dx_flat.shape)
    
    dx = dx_flat.reshape(N, T, V)
    
    return loss, dx

