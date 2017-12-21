import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *

class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nolinearity and softmax loss that uses a modular layer design. 
    We assume an input dimension of D, a hidden dimension of H, and perform classification over C classes.
    
    The architecture should be affine - relu - affine - softmax.
    
    Note that this does not implement gradient descent; instead, it will interact with a separate Solver object that is 
    responsible for running optimization.
    
    The learnable parameters of the model are stored in the dictionary self.params that maps parameter names to numpy 
    arrays.
    """
    def __init__(self, input_dim = 3 * 32 * 32, hidden_dim = 100, num_classes = 10, weight_scale = 1e-3, reg = 0.0):
        """
        Initialize a new network.
        
        Inputs:
        - input_dim: An integer giving the size of the input.
        - hidden_dim: An integer giving the size of the hidden layer.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random initializaiton of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg
        self.params['W1'] = np.random.randn(input_dim, hidden_dim) * weight_scale
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = np.random.randn(hidden_dim, num_classes) * weight_scale
        self.params['b2'] = np.zeros(num_classes)
    
    def loss(self, x, y = None):
        """
        Computes loss and gradient for a minibatch of data.
        
        Inputs:
        - x: Array of input data of shape (N, d_1, ..., d_k).
        - y: Array of labels, of shape (N, ). y[i] gives the label for x[i].
        
        Returns:
        If y is None, then run a test-time forward pas of the model and return:
        - scores: Array of shape (N, C) giving cllassification scores, where scores[i, c] is the classification score 
          for x[i] and class c.
        
        If y in not None, then run a traing-time forward and backward pass and return a tuple of:
        - loss: Scalar value giving the loss.
        - grads: Dictionary with the same keys as self.params, mapping parameter names to gradients of the loss with 
          respect to those parameters.
        """
        scores = None
        W1, W2, b1, b2 = self.params['W1'], self.params['W2'], self.params['b1'], self.params['b2']
        out_1, cache_1 = affine_relu_forward(x, W1, b1)
        scores, cache_2 = affine_forward(out_1, W2, b2)
        
        # If y is None, then we are in test mode so just return scores.
        if y is None:
            return scores
        
        loss, grads = 0, {}
        reg = self.reg
        
        loss, d_scores = softmax_loss(scores, y)
        loss += 0.5 * reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
        d_out1, d_W2, d_b2 = affine_backward(d_scores, cache_2)
        d_x, d_W1, d_b1 = affine_relu_backward(d_out1, cache_1)
        grads['W1'] = d_W1 + reg * W1
        grads['b1'] = d_b1
        grads['W2'] = d_W2 + reg * W2
        grads['b2'] = d_b2
        del d_x, d_W1, d_b1, d_W2, d_b2, d_out1, d_scores, cache_1, cache_2
        
        return loss, grads



class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers, ReLU nonlinearities, and a softmax loss 
    function. This will alse implement dropout and batch normalization as options. For a network with L layers, the
    architecture will be :
    
    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
    
    where batch normalization and dropout are optional, and the {...} block is repeated L - 1 times.
    
    Similar to the TwoLayerNet above, learnable parameters are stored in the self.params dictionary and will be learned using 
    the Solver class.
    """
    
    def __init__(self, hidden_dims, input_dim = 3 * 32 * 32, num_classes = 10, dropout = 0, use_batchnorm = False, 
                 reg = 0.0, weight_scale = 1e-2, dtype = np.float32, seed = None):
        """
        Initialize a new FullyConnectedNet.
        
        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the nubmer of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout = 0 then the network should not use dropout 
          at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random initializaiton of the weights.
        - dtype: A numpy datatype object; all computations will be performed using this datatype. float32 is faster but 
          less accurate, so you should user floate64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This will make the dropout layers 
          determinstic so we can gradient check the model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
        
        if type(hidden_dims) != list:
            raise ValueError('hidden_dim must be a list')
        
        dims = [input_dim] + hidden_dims + [num_classes]
        W = {'W' + str(i + 1): 
             np.random.randn(dims[i], dims[i + 1]) * weight_scale for i in range(self.num_layers)}
        b = {'b' + str(i + 1):
             np.zeros(dims[i + 1]) for i in range(self.num_layers)}
        self.params.update(W)
        self.params.update(b)
        
        
        # When using dropout we need to pass a dropout_param dictionary to each dropout layer so that the layer knows the 
        # dropout probability and the mode (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed
        
        # With batch normalization we need to keep track of runing means and variances, so we need to pass a special bn_param 
        # object to each batch normalization layer. You should pass self.bn_params[0] to the forward pass of the first batch 
        # normalization layer, self.bn_params[1] to the forward pass of the second batch normalizaiton layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
            gamma = {'gamma' + str(i + 1): np.ones(dims[i + 1]) for i in range(self.num_layers - 1)}
            beta = {'beta' + str(i + 1): np.zeros(dims[i + 1]) for i in range(self.num_layers -1)}
            self.params.update(gamma)
            self.params.update(beta)
                                                   
        
        # Cast all parameters to the correct data type.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)
    
    
    
    def loss(self, x, y = None):
        """
        Compute loss and gradient for the fully-connected net.
        
        Input / output: Same as TwoLayerNet above.
        """
        x = x.astype(self.dtype)
        mode = 'test' if y is None else 'train'
        
        # Set train/test mode for batchnorm params and dropout param since they behave differently during training and testing.
        if self.dropout_param is not None:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        
        scores = None
        
        # Forward pass
        cache, out, cache_dropout = {}, {}, {}
        out[0] = x
        
        for i in range(1, self.num_layers, 1):
            W, b = self.params['W' + str(i)], self.params['b' + str(i)]
            if self.use_batchnorm:
                gamma, beta = self.params['gamma' + str(i)], self.params['beta' + str(i)]
                out[i], cache[i] = affine_bn_relu_forward(out[i - 1], W, b, gamma, beta, self.bn_params[i - 1])
            else: 
                out[i], cache[i] = affine_relu_forward(out[i - 1], W, b)
            
            if self.use_dropout:
                out[i], cache_dropout[i] = dropout_forward(out[i], self.dropout_param)
        
        W, b = self.params['W' + str(self.num_layers)], self.params['b' + str(self.num_layers)]
        scores, cache[self.num_layers] = affine_forward(out[self.num_layers - 1], W, b)
        
        
        # If test mode return early
        if mode == 'test':
            return scores
        
        # Backward pass
        loss, grads = 0.0, {}
        
        loss, dscores = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (np.sum(W * W))
        
        dx, grads['W' + str(self.num_layers)], grads['b' + str(self.num_layers)] = affine_backward(dscores, 
                                                                                                   cache[self.num_layers])
        grads['W' + str(self.num_layers)] += self.reg * W
        
        for i in range(self.num_layers - 1, 0, -1):
            W_key = 'W' + str(i)
            b_key = 'b' + str(i)
            gamma_key = 'gamma' + str(i)
            beta_key = 'beta' + str(i)
            
            if self.use_dropout:
                dx = dropout_backward(dx, cache_dropout[i])
            
            if self.use_batchnorm:
                dx, grads[W_key], grads[b_key], grads[gamma_key], grads[beta_key] = \
                    affine_bn_relu_backward(dx, cache[i])
            else: 
                dx, grads[W_key], grads[b_key] = affine_relu_backward(dx, cache[i])
            
            loss += 0.5 * self.reg * (np.sum(self.params[W_key] * self.params[W_key]))
            grads[W_key] += self.reg * self.params[W_key]
        
        
        return loss, grads
        