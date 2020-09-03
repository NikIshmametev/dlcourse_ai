import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        self.lr = 1
        # TODO Create necessary layers
        self.fc1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.fc2 = FullyConnectedLayer(hidden_layer_size, n_output)
        self.layers = (self.fc1, self.fc2)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        params1 = self.fc1.params()
        params2 = self.fc2.params()
        for key in ['W', 'B']:
          params1[key].grad.fill(0)
          params2[key].grad.fill(0)
        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        self.relu1 = ReLULayer()
        x = self.relu1.forward(self.fc1.forward(X))
        y_pred = self.fc2.forward(x)
        loss, dpred = softmax_with_cross_entropy(y_pred, y)

        dout = self.fc2.backward(dpred)
        dout = self.relu1.backward(dout)
        dout = self.fc1.backward(dout)
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        loss_fc1_W_reg, grad_fc1_W_reg = l2_regularization(params1['W'].value, self.reg)
        loss_fc1_B_reg, grad_fc1_B_reg = l2_regularization(params1['B'].value, self.reg)
        loss_fc2_W_reg, grad_fc2_W_reg = l2_regularization(params2['W'].value, self.reg)
        loss_fc2_B_reg, grad_fc2_B_reg = l2_regularization(params2['B'].value, self.reg)

        self.fc2.W.grad += grad_fc2_W_reg
        self.fc2.B.grad += grad_fc2_B_reg
        self.fc1.W.grad += grad_fc1_W_reg
        self.fc1.B.grad += grad_fc1_B_reg
        return loss + (loss_fc1_W_reg+loss_fc1_B_reg+loss_fc2_W_reg+loss_fc2_B_reg)

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        x = self.relu1.forward(self.fc1.forward(X))
        probs = self.fc2.forward(x)
        return np.argmax(probs, axis=1)

    def params(self):
        # TODO Implement aggregating all of the params
        result = {}
        for layer in self.layers:
          for k, param in layer.params().items():
            result[' '.join([str(id(layer)), k])] = param
        return result


if __name__=='__main__':
  train_X, train_y = np.random.rand(10, 20), np.random.randint(0, 10, (10,1))
  model = TwoLayerNet(n_input = train_X.shape[1], n_output = 10, hidden_layer_size = 3, reg = 0)
  loss = model.compute_loss_and_gradients(train_X[:2], train_y[:2])
  model.params()
