import numpy as np

# Implementation of the GRU model
class GRU:

    # Randomly initialize weight and bias matrices with the given input and
    # hidden layer sizes
    def __init__(self, x_train, y_train, learning_rate, num_epochs, input_size, hidden_size, output_size, sequence_length):
      self.x_train = x_train
      self.y_train = y_train
      self.learning_rate = learning_rate
      self.num_epochs = num_epochs
      self.input_size = input_size
      self.hidden_size = hidden_size
      self.output_size = output_size
      self.num_sequences, self.sequence_length = self.x_train.shape[1], self.x_train.shape[2]
      self.weights = {}

      # Weights and bias for Update gate
      self.weights['W_xz'] = np.random.randn(self.hidden_size, self.input_size)
      self.weights['W_hz'] = np.random.randn(self.hidden_size, self.hidden_size)
      self.weights['b_z'] = np.zeros((self.hidden_size, 1))

      # Weights and bias for Reset gate
      self.weights['W_xr'] = np.random.randn(self.hidden_size, self.input_size)
      self.weights['W_hr'] = np.random.randn(self.hidden_size, self.hidden_size)
      self.weights['b_r'] = np.zeros((self.hidden_size, 1))

      # Weights and bias for Candidate hidden state
      self.weights['W_xh'] = np.random.randn(self.hidden_size, self.input_size)
      self.weights['W_hh'] = np.random.randn(self.hidden_size, self.hidden_size)
      self.weights['b_h'] = np.zeros((self.hidden_size, 1))

      # Weights and bias for output
      self.weights['W_o'] = np.random.randn(self.output_size, self.hidden_size)
      self.weights['b_o'] = np.zeros((self.output_size, 1))

      # Hidden state information
      self.ho = np.zeros((self.hidden_size, self.num_sequences))
      self.dh = np.zeros((self.hidden_size, self.num_sequences, self.sequence_length))

    # Train the model for the given number of epochs
    def fit(self):
      self.train_errs = []
      for epoch in range(self.num_epochs):

        # Forward pass
        preds, h, layer_params = self.forward()

        # Compute Mean Sum of Squared Error
        train_err = np.sum(np.square((preds - self.y_train))) / (self.sequence_length * self.num_sequences)
        self.train_errs.append(train_err)

        # Backprogapation
        self.backpropagate(layer_params)

        # Update Weights, Biases, and State parameters
        for w in self.weights:
          if w not in ['W_o', 'b_o']:
            self.weights[w] -= np.multiply(self.learning_rate, self.gradients['d' + w])
        self.dh -= np.multiply(self.learning_rate, self.gradients['dh_previous'])

    # Forward propagation logic for the (entire) GRU model
    def forward(self):
      # Store the calculated paramaters for every layer to use during backpropagation
      layer_parameters = []

      # Initialize state information
      h = np.zeros((self.hidden_size, self.num_sequences, self.sequence_length))
      preds = np.zeros((self.output_size, self.num_sequences, self.sequence_length))
      h_out = self.ho

      # Go through every layer/cell in the GRU model
      for layer in range(self.sequence_length):
        layer_preds, h_out, layer_params = self.forward_(self.x_train[:,:,layer], h_out)
        h[:,:,layer] = h_out
        preds[:,:,layer] = layer_preds
        layer_parameters.append(layer_params)

      return preds, h, layer_parameters

    # Forward propagation logic for the GRU model (for each individual layer/cell)
    def forward_(self, x, h_previous):

      # Update gate computation with sigmoid activation function
      z = self.sigmoid(np.dot(self.weights['W_xz'], x) + np.dot(self.weights['W_hz'], h_previous) + self.weights['b_z'])

      # Reset gate computation with sigmoid activation function
      r = self.sigmoid(np.dot(self.weights['W_xr'], x) + np.dot(self.weights['W_hr'], h_previous) + self.weights['b_r'])

      # Candidate hidden state computation with tanh activation function
      h_hat = self.tanh(np.dot(self.weights['W_xh'], x) + np.dot(self.weights['W_hh'], r * h_previous) + self.weights['b_h'])

      # Update the hidden state
      h_out = np.multiply(z, h_previous) + np.multiply((1 - z), h_hat)

      # Predict the output for the given input
      preds = np.dot(self.weights['W_o'], h_out) + self.weights['b_o']

      # Store the computed parameters for backpropagation
      layer_params = [z, r, h_hat, h_out, h_previous, x]

      return preds, h_out, layer_params

    # Sigmoid activation function
    def sigmoid(self, x):
      return (1 / (1 + np.exp(-x)))

    # Derivative of sigmoid activation function
    def d_sigmoid(self, x):
      return np.multiply(self.sigmoid(x), 1 - self.sigmoid(x))

    # Tanh activation function
    def tanh(self, x):
      return ((np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)))

    # Derivative of Tanh activation function
    def d_tanh(self, x):
      return 1 - np.square(self.tanh(x))

    # Backpropagation logic for the (entire) GRU model
    def backpropagate(self, layer_params):
      self.gradients = {}

      # Weights and bias for Update gate
      self.gradients['dW_xz'] = np.random.randn(self.hidden_size, self.input_size)
      self.gradients['dW_hz'] = np.random.randn(self.hidden_size, self.hidden_size)
      self.gradients['db_z'] = np.zeros((self.hidden_size, 1))

      # Weights and bias for Reset gate
      self.gradients['dW_xr'] = np.random.randn(self.hidden_size, self.input_size)
      self.gradients['dW_hr'] = np.random.randn(self.hidden_size, self.hidden_size)
      self.gradients['db_r'] = np.zeros((self.hidden_size, 1))

      # Weights and bias for Candidate hidden state
      self.gradients['dW_xh'] = np.random.randn(self.hidden_size, self.input_size)
      self.gradients['dW_hh'] = np.random.randn(self.hidden_size, self.hidden_size)
      self.gradients['db_h'] = np.zeros((self.hidden_size, 1))

      # Weights and bias for output
      self.gradients['dx'] = np.zeros((self.input_size, self.num_sequences, self.sequence_length))
      self.gradients['dh_previous'] = np.zeros((self.hidden_size, self.num_sequences, self.sequence_length))

      # Go through every layer/cell in the GRU model
      for layer in range(self.sequence_length-1, -1, -1):
        self.backpropagate_(self.dh[:,:,layer], layer, layer_params[layer])

    def backpropagate_(self, dh, layer, layer_params):
      # Get the gradient of each computed parameter from layer parameters
      z, r, h_hat, h_out, h_previous, x = layer_params

      # Update gate gradient
      dz1 = np.multiply(dh, h_hat)
      dz2 = np.multiply(dh, h_previous)
      dz3 = dz1 + 1 - dz2
      dz4 = self.d_sigmoid(z)
      dz = np.multiply(dz3, dz4)

      # Candidate hidden state gradient
      dhhat1 = np.multiply(dh, z)
      dhhat2 = self.d_tanh(h_hat)
      dhhat = np.multiply(dhhat1, dhhat2)

      # Reset gate gradient
      dr1 = np.dot(np.transpose(self.weights["W_hh"]), dhhat)
      dr2 = np.multiply(dr1, r)
      dr3 = self.d_sigmoid(r)
      dr = np.multiply(dr2, dr3)

      # Hidden state gradient
      dh1 = np.dot(np.transpose(self.weights["W_hh"]), dhhat)
      dh2 = np.multiply(dh1, r)
      dh3 = np.multiply(1- z, dh)
      dh4 = np.dot(np.transpose(self.weights["W_hh"]), dr)
      dh5 = np.dot(np.transpose(self.weights["W_hh"]), dz)
      dh = dh2 + dh3 + dh4 + dh5

      # Output gradient
      dx1 = np.dot(np.transpose(self.weights["W_xh"]), dhhat)
      dx2 = np.dot(np.transpose(self.weights["W_xh"]), dz)
      dx3 = np.dot(np.transpose(self.weights["W_xh"]), dr)
      dx = dx1 + dx2 + dx3

      # Compute the final weight and bias gradients

      # Update gate weights and bias
      self.gradients['dW_xz'] += np.dot(dz, np.transpose(x))
      self.gradients['dW_hz'] += np.dot(dz, np.transpose(h_previous))
      self.gradients['db_z'] += np.sum(dz, axis=1, keepdims=True)

      # Reset gate weights and bias
      self.gradients['dW_xr'] += np.dot(dr, np.transpose(x))
      self.gradients['dW_hr'] += np.dot(dr, np.transpose(h_previous))
      self.gradients['db_r'] += np.sum(dr, axis=1, keepdims=True)

      # Weights and bias for Candidate hidden state
      self.gradients['dW_xh'] += np.dot(dh, np.transpose(x))
      self.gradients['dW_hh'] += np.dot(dh, np.transpose(np.multiply(h_previous, r)))
      self.gradients['db_h'] += np.sum(dh, axis=1, keepdims=True)

      # Output and hidden state gradients
      self.gradients['dx'][:,:,layer] = dx
      self.gradients['dh_previous'][:,:,layer] = dh

    # Prediction logic for the (entire) GRU model
    def predict(self, x_test):
      # Initialize state information
      h = np.zeros((self.hidden_size, x_test.shape[1], self.sequence_length))
      preds = np.zeros((self.output_size, x_test.shape[1], self.sequence_length))
      h_out = np.zeros((self.hidden_size, x_test.shape[1]))

      # Go through every layer/cell in the GRU model
      for layer in range(self.sequence_length):
        layer_preds, h_out = self.predict_(x_test[:,:,layer], h_out)
        h[:,:,layer] = h_out
        preds[:,:,layer] = layer_preds

      return preds

    # Prediction logic for the GRU model (for each individual layer/cell)
    def predict_(self, x, h_previous):

      # Update gate computation with sigmoid activation function
      z = self.sigmoid(np.dot(self.weights['W_xz'], x) + np.dot(self.weights['W_hz'], h_previous) + self.weights['b_z'])

      # Reset gate computation with sigmoid activation function
      r = self.sigmoid(np.dot(self.weights['W_xr'], x) + np.dot(self.weights['W_hr'], h_previous) + self.weights['b_r'])

      # Candidate hidden state computation with tanh activation function
      h_hat = self.tanh(np.dot(self.weights['W_xh'], x) + np.dot(self.weights['W_hh'], r * h_previous) + self.weights['b_h'])

      # Update the hidden state
      h_out = np.multiply(z, h_previous) + np.multiply((1 - z), h_hat)

      # Predict the output for the given input
      preds = np.dot(self.weights['W_o'], h_out) + self.weights['b_o']

      return preds, h_out