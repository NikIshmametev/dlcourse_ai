import numpy as np
import os
from dataset import load_svhn, random_split_train_val


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
    if pb.ndim == 1:
      pb = pb[np.newaxis, :]
    loss = -np.log(pb[np.arange(pb.shape[0])[:, None], target_index])
    return np.average(loss)


def softmax_with_cross_entropy(predictions, target_index):
    '''
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
    '''
    # TODO implement softmax with cross-entropy
    # Your final implementation shouldn't have any loops
    shape = predictions.shape
    probs = softmax(predictions)
    if probs.ndim == 1:
      probs = probs[np.newaxis, :]
    loss = cross_entropy_loss(probs, target_index)
    dprediction = probs.copy()
    dprediction[np.arange(probs.shape[0])[:,np.newaxis], target_index] -= 1
    # Градиент делим на batch_size, так как при численном вычислении усредняем дельту по одной координате
    # Тогда как при аналитическом надо учесть это здесь
    return loss, np.resize(dprediction, shape)/probs.shape[0]


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''

    # TODO: implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops
    loss = reg_strength*np.sum(W*W)
    grad = 2*reg_strength*W
    return loss, grad
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    predictions = np.dot(X, W)

    # TODO implement prediction and gradient over W
    # Your final implementation shouldn't have any loops
    probs = softmax(predictions)
    if probs.ndim == 1:
      probs = probs[np.newaxis, :]
    loss = cross_entropy_loss(probs, target_index)
    dprediction = np.dot(X.T, probs)
    flag = np.zeros_like(probs)
    flag[np.arange(probs.shape[0])[:, np.newaxis], target_index] += 1
    dprediction -= np.dot(X.T, flag)
    return loss, dprediction/probs.shape[0]   


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1, mute=False):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        target_index = y[:, np.newaxis]
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            # TODO implement generating batches from indices
            # Compute loss and gradients
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!
            for batch in batches_indices:
              loss, grad = linear_softmax(X[batch], self.W, target_index[batch])
              loss_reg, grad_reg = l2_regularization(self.W, reg)
              loss += loss_reg
              grad += grad_reg
              self.W -= learning_rate*grad
            # end
            loss_history.append(loss)
            if not mute:
              print("Epoch %i, loss: %f" % (epoch, loss))
        print("Final loss for %i epochs: %f" % (epochs, loss))
        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.zeros(X.shape[0], dtype=np.int)

        # TODO Implement class prediction
        # Your final implementation shouldn't have any loops
        y_pred = np.argmax(np.dot(X, self.W), axis=1)
        return y_pred




def prepare_for_linear_classifier(train_X, test_X):
  train_flat = train_X.reshape(train_X.shape[0], -1).astype(np.float) / 255.0
  test_flat = test_X.reshape(test_X.shape[0], -1).astype(np.float) / 255.0
  
  # Subtract mean
  mean_image = np.mean(train_flat, axis = 0)
  train_flat -= mean_image
  test_flat -= mean_image
  
  # Add another channel with ones as a bias term
  train_flat_with_ones = np.hstack([train_flat, np.ones((train_X.shape[0], 1))])
  test_flat_with_ones = np.hstack([test_flat, np.ones((test_X.shape[0], 1))])    
  return train_flat_with_ones, test_flat_with_ones



if __name__=='__main__':
  batch_size = 3
  num_classes = 4
  num_features = 2
  np.random.seed(42)
  W = np.random.randint(-1, 3, size=(num_features, num_classes)).astype(np.float)
  X = np.random.randint(-1, 3, size=(batch_size, num_features)).astype(np.float)
  target_index = np.ones(batch_size, dtype=np.int)

  loss, dW = linear_softmax(X, W, target_index)


  # train_X, train_y, test_X, test_y = load_svhn("./assignments/assignment1/data", max_train=10000, max_test=1000)    
  # train_X, test_X = prepare_for_linear_classifier(train_X, test_X)
  # # Split train into train and val
  # train_X, train_y, val_X, val_y = random_split_train_val(train_X, train_y, num_val = 1000)
  
  # classifier = LinearSoftmaxClassifier()
  # loss_history = classifier.fit(train_X, train_y, epochs=10, learning_rate=1e-3, batch_size=300, reg=1e1)
                
                                                          

            

                
