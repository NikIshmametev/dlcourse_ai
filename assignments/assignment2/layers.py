import numpy as np

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
