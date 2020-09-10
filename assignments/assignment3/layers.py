# coding: utf-8
import numpy as np
from gradient_check import check_layer_gradient

def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    """https://github.com/cthorey/CS231/blob/master/assignment2/cs231n/im2col.py"""
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = (H + 2 * padding - field_height) / stride + 1
    out_width = (W + 2 * padding - field_width) / stride + 1

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    i = i.astype(np.int)
    j = j.astype(np.int)
    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k, i, j)


def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
                                stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1,
                   stride=1):
    """ An implementation of col2im based on fancy indexing and np.add.at """
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding,
                                stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    # TODO implement softmax
    # Your final implementation shouldn't have any loops
    pb = predictions.copy()
    shape = predictions.shape
    if predictions.ndim == 1:
      pb = pb[np.newaxis,:]
    pb -= np.max(pb, axis=1)[:,np.newaxis]
    res = np.exp(pb)/np.sum(np.exp(pb), axis=1)[:,np.newaxis]
    return np.resize(res, shape)


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops
    pb = probs.copy()
    if pb.ndim == 1 and target_index.ndim > 1:
      pb = pb[np.newaxis, :]
      loss = -np.log(pb[np.arange(pb.shape[0])[:, None], target_index])
    else:
      loss = -np.log(pb[np.arange(pb.shape[0]), target_index])
    return np.average(loss)



def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    # TODO: Copy from the previous assignment
    loss = reg_strength*np.sum(W*W)
    grad = 2*reg_strength*W
    return loss, grad


def softmax_with_cross_entropy(predictions, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    # TODO: Copy from the previous assignment
    shape = predictions.shape
    probs = softmax(predictions)
    if probs.ndim == 1:
      probs = probs[np.newaxis, :]
    loss = cross_entropy_loss(probs, target_index)
    dprediction = probs.copy()
    dprediction[np.arange(probs.shape[0]), target_index] -= 1
    # Градиент делим на batch_size, так как при численном вычислении усредняем дельту по одной координате
    # Тогда как при аналитическом надо учесть это здесь
    return loss, np.resize(dprediction, shape)/probs.shape[0]


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        self.grad = np.where(X>=0, 1, 0)
        return np.where(X>=0, X, 0)

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        return d_out*self.grad

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        self.X = X
        return np.dot(self.X, self.W.value) + self.B.value

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment
        self.W.grad += np.dot(self.X.T, d_out)
        self.B.grad += np.sum(d_out, axis=0)[np.newaxis, :]
        return np.dot(d_out, self.W.value.T)


    def params(self):
        return {'W': self.W, 'B': self.B}

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding, stride=1):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.flat_size = self.in_channels*self.filter_size**2
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding
        self.stride = stride


    def forward(self, X):
        self.X = X.copy()
        # Transpose to get a unit cell (two last dimesions) in the form of filter
        self.X_tr = np.transpose(self.X, (0, 3, 1, 2)) 
        self.W_tr = np.transpose(self.W.value, (3, 2, 0, 1))

        self.batch_size, self.height, self.width, _ = X.shape
        
        out_height = self.height - self.filter_size + 2*self.padding + 1
        out_width = self.width - self.filter_size + 2*self.padding + 1
        # Later transpose this shape
        self.out = np.zeros((self.batch_size, self.out_channels, out_height, out_width))
        
        # To get outpput pixels, will be iterate through this array
        self.X1 = np.pad(self.X_tr, ((0,0), (0,0),
                        (self.padding, self.padding), (self.padding, self.padding)))
        _, _, self.height_X_in, self.width_X_in = self.X1.shape

        for y0 in range(out_height):
            for x0 in range(out_width):
                x1, y1 = x0 + self.filter_size, y0 + self.filter_size
                X_col = np.resize(self.X1[:, :, y0:y1, x0:x1], (self.batch_size, self.flat_size))
                W_col = np.resize(self.W_tr, (self.out_channels, self.flat_size))
                self.out[:, :, y0, x0] = X_col @ W_col.T + self.B.value
        self.out = np.transpose(self.out, (0, 2, 3, 1))
        return self.out


    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output
        batch_size, out_height, out_width, out_channels = d_out.shape
        dout_tr = np.transpose(d_out, (0, 3, 1, 2))
        X_in = np.zeros((batch_size, self.in_channels, self.height_X_in, self.width_X_in))
        for y0 in range(out_height):
            for x0 in range(out_width):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)
                dout_col = dout_tr[:, :, y0, x0]
                W_col = np.resize(self.W_tr, (self.out_channels, self.flat_size))
                tmp = np.resize(dout_col @ W_col, (batch_size, self.in_channels, self.filter_size, self.filter_size))                
                
                x1, y1 = x0 + self.filter_size, y0 + self.filter_size
                X_in[:, :, y0:y1, x0:x1] += tmp

                X_col = np.resize(self.X1[:, :, y0:y1, x0:x1], (self.batch_size, self.flat_size))
                dW = np.resize(dout_col.T @ X_col, (self.out_channels, self.in_channels, self.filter_size, self.filter_size))
                self.W.grad += np.transpose(dW, (2,3,1,0))
                self.B.grad += np.sum(dout_tr[:, :, y0, x0], axis=0)
        if self.padding:
            return X_in[:, self.padding:-self.padding, self.padding:-self.padding, :]
        else:
            return X_in

    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        raise Exception("Not implemented!")

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape
        raise Exception("Not implemented!")

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        raise Exception("Not implemented!")

    def backward(self, d_out):
        # TODO: Implement backward pass
        raise Exception("Not implemented!")

    def params(self):
        # No params!
        return {}


if __name__=='__main__':
    X = np.array([
              [
               [[1.0, 0.0], [2.0, 1.0]],
               [[0.0, -1.0], [-1.0, -2.0]]
              ]
              ,
              [
               [[0.0, 1.0], [1.0, -1.0]],
               [[-2.0, 2.0], [-1.0, 0.0]]
              ]
             ])
    layer = ConvolutionalLayer(in_channels=2, out_channels=2, filter_size=2, padding=0)
    result = layer.forward(X)
    d_input = layer.backward(np.ones_like(result))
    check_layer_gradient(layer, X)
