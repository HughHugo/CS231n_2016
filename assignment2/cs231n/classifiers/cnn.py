import numpy as np
from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:
    
    conv - relu - 2x2 max pool - affine - relu - affine - softmax
    
    The network aperates on minibatches of data that have shape (N, C, H, W) consisting of N images, each with height H and 
    width W and with C input channels.
    """
    
    def __init__(self, input_dim = (3, 32, 32), num_filters = 32, filter_size = 7, hidden_dim = 100, num_classes = 10, 
                 weight_scale = 1e-3, reg = 0.0, dtype = np.float32):
        """
        Initialize a new network.
        
        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hiden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer
        - weight_scale: Scalar giving standard deviation for random initialization of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        
        C, H, W = input_dim
        
        stride_conv = 1
        pad_conv = (filter_size - 1) // 2
        H_conv = (H + 2 * pad_conv - filter_size) // stride_conv + 1
        W_conv = (W + 2 * pad_conv - filter_size) // stride_conv + 1
        
        stride_pool = 2
        width_pool = 2
        height_pool = 2
        H_pool = (H_conv - height_pool) // stride_pool + 1
        W_pool = (W_conv - width_pool) // stride_pool + 1
        
        self.params['W1'] = np.random.randn(num_filters, C, filter_size, filter_size) * weight_scale
        self.params['b1'] = np.zeros(num_filters)
        self.params['W2'] = np.random.randn(num_filters * H_pool * W_pool, hidden_dim) * weight_scale
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['W3'] = np.random.randn(hidden_dim, num_classes) * weight_scale
        self.params['b3'] = np.zeros(num_classes)
        
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)
    
    
    def loss(self, x, y = None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.
        
        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        
        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}
        
        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        
        # the forward pass
        out, cache = {}, {}
        out[0] = x
        out[1], cache[1] = conv_relu_pool_forward(out[0], W1, b1, conv_param, pool_param)
        out[2], cache[2] = affine_relu_forward(out[1].reshape(out[1].shape[0], -1), W2, b2)
        out[3], cache[3] = affine_forward(out[2], W3, b3)
        
        scores = out[3]
        
        if y is None:
            return scores
        
        # the backward pass
        dout = {}
        loss, dscores = softmax_loss(scores, y)
        dout[3] = dscores
        dout[2], dW3, db3 = affine_backward(dout[3], cache[3])
        dout[1], dW2, db2 = affine_relu_backward(dout[2], cache[2])
        dout[1] = dout[1].reshape(out[1].shape)
        dout[0], dW1, db1 = conv_relu_pool_backward(dout[1], cache[1])
        
        grads = {'W1': dW1 + self.reg * W1, 
                 'W2': dW2 + self.reg * W2,
                 'W3': dW3 + self.reg * W3, 
                 'b1': db1, 
                 'b2': db2, 
                 'b3': db3}
        loss += 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2) + np.sum(W3 * W3))
        
        del out, cache, dout
        return loss, grads


class FiveLayerConvNet(object):
    """
    A four-layer convolutional network with the following architecture:
    
    {conv - relu - 2x2 max pool} x 3 - affine - relu - affine - softmax
    
    The network aperates on minibatches of data that have shape (N, C, H, W) consisting of N images, each with height H and 
    width W and with C input channels.
    """
    def __init__(self, input_dim = (3, 32, 32), num_filters = [8, 16, 32], filter_size = [3, 3, 3], hidden_dim = 100, 
                 num_classes = 10, weight_scale = 1e-3, reg = 0.0, dtype = np.float32):
        """
        Initialize a new network.
        
        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: List contain numbers of filters to use in each convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer
        - weight_scale: Scalar giving standard deviation for random initialization of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        
        C, H, W = input_dim
        
        stride_conv = 1
        pad_conv = [(x - 1) // 2 for x in filter_size]
        stride_pool = 2
        width_pool = 2
        height_pool = 2
        
        H1_conv = (H + 2 * pad_conv[0] - filter_size[0]) // stride_conv + 1
        W1_conv = (W + 2 * pad_conv[0] - filter_size[0]) // stride_conv + 1
        H1_pool = (H1_conv - height_pool) // stride_pool + 1
        W1_pool = (W1_conv - width_pool) // stride_pool + 1
        
        H2_conv = (H1_pool + 2 * pad_conv[1] - filter_size[1]) // stride_conv + 1
        W2_conv = (W1_pool + 2 * pad_conv[1] - filter_size[1]) // stride_conv + 1
        H2_pool = (H2_conv - height_pool) // stride_pool + 1
        W2_pool = (W2_conv - width_pool) // stride_pool + 1
        
        H3_conv = (H2_pool + 2 * pad_conv[2] - filter_size[2]) // stride_conv + 1
        W3_conv = (W2_pool + 2 * pad_conv[2] - filter_size[2]) // stride_conv + 1
        H3_pool = (H3_conv - height_pool) // stride_pool + 1
        W3_pool = (W3_conv - width_pool) // stride_pool + 1
        
        self.params['W1'] = np.random.randn(num_filters[0], C, filter_size[0], filter_size[0]) * weight_scale
        self.params['b1'] = np.zeros(num_filters[0])
        self.params['W2'] = np.random.randn(num_filters[1], num_filters[0], filter_size[1], filter_size[1]) * weight_scale
        self.params['b2'] = np.zeros(num_filters[1])
        self.params['W3'] = np.random.randn(num_filters[2], num_filters[1], filter_size[2], filter_size[2]) * weight_scale
        self.params['b3'] = np.zeros(num_filters[2])
        
        self.params['W4'] = np.random.randn(num_filters[2] * H3_pool * W3_pool, hidden_dim) * weight_scale
        self.params['b4'] = np.zeros(hidden_dim)
        self.params['W5'] = np.random.randn(hidden_dim, num_classes) * weight_scale
        self.params['b5'] = np.zeros(num_classes)
        
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)
    
    
    def loss(self, x, y = None):
        """
        Evaluate loss and gradient for the five-layer convolutional network.
        
        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']
        W5, b5 = self.params['W5'], self.params['b5']
        
        
        # pass conv_param to the forward pass for the convolutional layer
        filter_size = [W1.shape[2], W2.shape[2], W3.shape[2]]
        conv1_param = {'stride': 1, 'pad': (filter_size[0] - 1) // 2}
        conv2_param = {'stride': 1, 'pad': (filter_size[1] - 1) // 2}
        conv3_param = {'stride': 1, 'pad': (filter_size[2] - 1) // 2}
        
        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        
        # the forward pass
        out, cache = {}, {}
        out[0] = x
        out[1], cache[1] = conv_relu_pool_forward(out[0], W1, b1, conv1_param, pool_param)
        out[2], cache[2] = conv_relu_pool_forward(out[1], W2, b2, conv2_param, pool_param)
        out[3], cache[3] = conv_relu_pool_forward(out[2], W3, b3, conv3_param, pool_param)
        
        #out[4], cache[4] = affine_relu_forward(out[3].reshape(out[3].shape[0], -1), W4, b4)
        out[4], cache[4] = affine_relu_forward(out[3], W4, b4)
        out[5], cache[5] = affine_forward(out[4], W5, b5)
        
        scores = out[5]
        
        if y is None:
            return scores
        
        # the backward pass
        dout = {}
        loss, dscores = softmax_loss(scores, y)
        dout[5] = dscores
        dout[4], dW5, db5 = affine_backward(dout[5], cache[5])
        dout[3], dW4, db4 = affine_relu_backward(dout[4], cache[4])
        
        #dout[3] = dout[3].reshape(out[3].shape)
        dout[2], dW3, db3 = conv_relu_pool_backward(dout[3], cache[3])
        dout[1], dW2, db2 = conv_relu_pool_backward(dout[2], cache[2])
        dout[0], dW1, db1 = conv_relu_pool_backward(dout[1], cache[1])
        
        grads = {'W1': dW1 + self.reg * W1, 
                 'W2': dW2 + self.reg * W2,
                 'W3': dW3 + self.reg * W3, 
                 'W4': dW4 + self.reg * W4, 
                 'W5': dW5 + self.reg * W5, 
                 'b1': db1, 
                 'b2': db2, 
                 'b3': db3, 
                 'b4': db4, 
                 'b5': db5}
        loss += 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2) + np.sum(W3 * W3) + np.sum(W4 * W4) + np.sum(W5 * W5))
        
        self.out = out
        self.cache = cache
        self.dout = dout
        del out, cache, dout
        return loss, grads