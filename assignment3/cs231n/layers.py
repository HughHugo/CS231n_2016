import numpy as np

def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.
    
    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N examples, where each example x[i] has
    shape (d_1, ..., d_k). We  will reshape each input into a vector fo dimension D = d_1 * ... * d_k, and then
    transform it to an output vector of dimension M.
    
    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M, )
    
    Return a tuple of:
    - out: output of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    
    N = x.shape[0]
    out = x.reshape(N, np.prod(x.shape[1:])).dot(w) + b
    
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.
    
    Inputs:
    - dout: UPstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ..., d_k)
      - w: Weights, of shape (D, M)
    
    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d_1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M, )
    """
    
    x, w, b = cache
    dx, dw, db = None, None, None
    
    N = x.shape[0]
    dx = dout.dot(w.T).reshape(x.shape)
    dw = x.reshape(N, np.prod(x.shape[1:])).T.dot(dout)
    db = np.sum(dout, axis = 0)
    
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).
    
    Inputs:
    - x: Inputs, of any shape.
    
    Returns a tuple of:
    - out: Output, of the same shape as x.
    - cache: x.
    """
    out = np.maximum(0, x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Conputes the backward pass for a layer of rectified linear units (ReLUs).
    
    Inputs:
    - dout: Upstream derivatives, of any shape.
    - cache: Input x, of same shape as dout.
    
    Returns:
    - dx: Gradient with respect to x.
    """
    x = cache
    dx = np.where(x > 0, dout, 0)
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.
    
    During training the sample mean and (uncorrected) sample variance are computed from minibatch statistics and used to 
    normalize the incoming data. During training we alse keep an exponentially decaying running mean of the mean and variance 
    of each feature, and these averages are used to normalize data at test-time.
    
    At each timestep we update the running averages for mean and variance using an exponential decay based on the momentum 
    parameter:
    
    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var
    
    Note that the batch normalization paper suggests a different test-time behavior: they compute sample mean and variance for 
    each feature using a large number of training images rather than using a running average. For this implementaion we have 
    chosen to use running averages instead since they do not require an additional estimation step; the torch7 implementaion 
    of batch normalization also uses running averages.
    
    Input:
    - x: Data of shape (N, D) 
    - gamma: Scale parameter of shape (D, )
    - beta: Shift parameter of shape (D, )
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running / variance.
      - running_mean: Array of shape (D, ) giving running mean of features
      - running_var: Array of shape (D, ) giving running variance of features
      
    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)
    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype = x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype = x.dtype))
    
    if mode == 'train':
        sample_mean = np.mean(x, axis = 0)
        sample_var = np.var(x, axis = 0)
        x_norm = (x - sample_mean) / np.sqrt(sample_var + eps)
        out = gamma * x_norm + beta
        
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        
        # Store the updated running means bach into bn_param
        bn_param['running_mean'] = running_mean
        bn_param['running_var'] = running_var
        
        cache = (bn_param, x, x_norm, sample_mean, sample_var, gamma, beta, eps)
        
    elif mode == 'test':
        x_norm = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_norm + beta
        cache = (bn_param, x, x_norm, gamma, beta)
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)
    
    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.
    
    For this implementation, you should write out a computation graph for batch normalization on paper and propagate gradients 
    backward through internediate nodes.
    
    Inputs:
    - dout: Upstream derivatives, of shape (N, D);
    - cache: Variable of intermediates from batchnorm_forward.
    
    Returns a tupe of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gama, of shape (D, )
    - dbeta: Gradient with respect to shift parameter beta, of shape (D, )
    """
    mode = cache[0]['mode']
    if mode == 'train':
        bn_param, x, x_norm, sample_mean, sample_var, gamma, beta, eps = cache
        N = x.shape[0]
        
        dgamma = np.sum(dout * x_norm, axis = 0)
        dbeta = np.sum(dout, axis = 0)
        dx_norm = dout * gamma
        dsample_var = -0.5 * np.sum(x_norm * dx_norm, axis = 0) / (sample_var + eps)
        dsample_mean = -np.sum(dx_norm, axis = 0) / np.sqrt(sample_var + eps)
        dx = dx_norm / np.sqrt(sample_var + eps) + dsample_mean / N + 2 * (x - sample_mean) * dsample_var / N
    elif mode == 'test':
        bn_param, x, x_norm, gamma, beta = cache
        eps = bn_param.get('eps', 1e-5)
        running_var = bn_param.get('running_var', np.zeros(x.shape[1], dtype = x.dtype))
        
        dbeta = np.sum(dout, axis = 0)
        dgamma = np.sum(x_norm * dout, axis = 0)
        dx_norm = dout * gamma
        dx = dx_norm / np.sqrt(running_var + eps)
    
    return dx, dgamma, dbeta


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.
    
    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C, )
    - beta: Shift parameter, of shape (C, )
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum = 0 means that old information is discarded completely at every 
        time step, while momentum = 1 means that new information is never incorporated. The default of momentum = 0.9 should 
        work well in most situations.
      - running_mean: Array of shape (C, ) giving running mean of features
      - running_var: Array of shape (C, ) giving running variance of features
    
    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backwars pass
    """
    N, C, H, W = x.shape
    out_reshaped, cache = batchnorm_forward(x.transpose(0, 2, 3, 1).reshape(-1, C), gamma, beta, bn_param)
    out = out_reshaped.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.
    
    Inputs:
    - dout: Upstream derivatives, of shape (H, C, H, W)
    - cache: Valuese from the forward pass
    
    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C, )
    - dbeta: Gradient with respect to shift parameter, of shape (C, )
    """
    N, C, H, W = dout.shape
    dx_reshaped, dgamma, dbeta = batchnorm_backward(dout.transpose(0, 2, 3, 1).reshape(-1, C), cache)
    dx = dx_reshaped.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Compute the loss and gradeint using for multiclass SVM classification. 
    
    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class for the ith input.
    - y: Vector of labels, of shape (N, ) where y[i] is the label for x[i] and 0 <= y[i] < C
    
    Returns a tuple of :
    - loss: Scalar giving the loss.
    - dx: Gradient of the loss with respect to x.
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis = 1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.
    
    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth classes for the ith input.
    - y: Vector of labels, of shape (N, ) where y[i] is the label for x[i] and 0 <= y[i] < C.
    
    Returns a tuple of:
    - loss: Scalar giving the loss.
    - dx: Gradient of the loss with respect to x.
    """
    probs = np.exp(x - np.max(x, axis = 1, keepdims = True))
    probs /= np.sum(probs, axis = 1, keepdims = True)
    N = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx