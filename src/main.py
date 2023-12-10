# General imports
import numpy as np
import matplotlib.pyplot as plt
import itertools
from tabulate import tabulate

# Suppress Runtime Warnings
import warnings
warnings.filterwarnings("ignore")

# Local imports
from GRU import GRU
from IndexData import IndexData

data_source = 'https://github.com/nikhilmanda9/ML-Stock-Market-Prediction/blob/main/all_stocks_5yr.parquet?raw=true'

# Create a log file and try different parameters for the GRU model
def hyperparameter_tuning():
  with open("ML-Test_Repo/gru_logs.txt", 'w') as logfile:
    # Different values for our parameters
    sequence_lengths = [3,4,5]

    learning_rates = [0.0001, 0.00001, 0.000001]
    epochs = [25, 50, 100]
    hidden_sizes = [5, 10, 20]

    # Store the parameters and errors after training the GRU model
    values = []

    # The cartesian product of all possible parameters
    for sequence_length, learning_rate, num_epochs, hidden_size in itertools.product(sequence_lengths,
                                                                                    learning_rates,
                                                                                    epochs,
                                                                                    hidden_sizes):
      # Train the GRU model with the given parameters
      # We keep the sequences in a Many-One structure (input sequences have multiple values
      # while output sequence will always be of length one)
      index_data_obj = IndexData(data_source,
                                sequence_length)

      gru = GRU(x_train=index_data_obj.x_train,
            y_train=index_data_obj.y_train,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            input_size=1,
            hidden_size=hidden_size,
            output_size=1,
            sequence_length=sequence_length)
      gru.fit()

      # Get the parameters, data, and errors for the model
      x_train = index_data_obj.x_train
      y_train = index_data_obj.y_train
      x_test = index_data_obj.x_test
      y_test = index_data_obj.y_test
      train_dates = index_data_obj.index_data['date'][:x_train.shape[1]]
      test_dates = index_data_obj.index_data['date'][x_train.shape[1]:x_train.shape[1]+x_test.shape[1]]
      train_indices = index_data_obj.index_data['index'][:x_train.shape[1]]
      test_indices = index_data_obj.index_data['index'][x_train.shape[1]:x_train.shape[1]+x_test.shape[1]]
      train_errs = gru.train_errs
      train_err = train_errs[-1]
      test_output = gru.predict(index_data_obj.x_test)
      test_err = np.sum(np.square((test_output - index_data_obj.y_test))) / (sequence_length * gru.num_sequences)

      values.append({'sequence_length': sequence_length,
                      'learning_rate': learning_rate,
                      'num_epochs': num_epochs,
                      'hidden_size': hidden_size,
                      'train_err': train_err,
                      'test_err': test_err,
                      'x_train': x_train,
                      'y_train': y_train,
                      'x_test': x_test,
                      'y_test': y_test,
                      'train_errs': train_errs,
                      'test_output': test_output,
                      'train_dates': train_dates,
                      'test_dates': test_dates,
                      'train_indices': train_indices,
                      'test_indices': test_indices,
                      'scaler': index_data_obj.scaler})
      
    tabulate_values = [list(d.values())[:6] for d in values]

    # Write the primary parameters and errors to the logfile
    logfile.write(tabulate(tabulate_values,
                  headers=['Sequence Length', 'Learning Rate', 'Number of Epochs',
                          'Hidden Size', 'Training Error', 'Testing Error'],
                  floatfmt=["", "", "", "", ".4f", ".4f"]))
  
  return values

# Extract the best parameters based on the lowest average Training and Testing Errors
def best_parameters(values):
  best_params = {}
  best_avg_err = 1e9

  for val_dict in values:
    avg_err = (val_dict['train_err'] + val_dict['test_err']) / 2
    if avg_err < best_avg_err:
      best_avg_err = avg_err
      best_params = val_dict

  # Print the best parameters and errors
  print("Sequence Length: {}\nLearning Rate: {}\nNumber of Epochs: {}\nHidden Size: {}\nTraining Error: {:.4f}\nTesting Error: {:.4f}".format(
      best_params['sequence_length'], 
      best_params['learning_rate'], 
      best_params['num_epochs'],
      best_params['hidden_size'],
      best_params['train_err'],
      best_params['test_err']))
  
  return best_params

# Plot the original S&P 500 Index data without sequences
def plot_original(filepath, best_params):
  fig = plt.subplots(figsize=(16, 5))

  # Plot training data
  plt.plot(best_params['train_dates'], best_params['train_indices'], color='r')
  
  # Plot testing data
  plt.plot(best_params['test_dates'], best_params['test_indices'], color='b')
  
  plt.title("Daily S&P 500 Index (Unsequenced)")
  plt.legend(['Train', 'Test'])
  plt.xlabel('Date')
  plt.ylabel('Index Price (USD)')
  plt.savefig(filepath)

# Plot the sequenced S&P 500 Index data
def plot_sequenced(filepath, best_params):
  fig2 = plt.subplots(figsize=(16, 5))

  # Plot training data
  x_train = best_params['scaler'].inverse_transform(best_params['x_train'][0])
  x_train = [np.mean(arr) for arr in x_train]
  plt.plot(best_params['train_dates'], x_train, color='r')

  # Plot testing data
  x_test = best_params['scaler'].inverse_transform(best_params['x_test'][0])
  x_test = [np.mean(arr) for arr in x_test]
  plt.plot(best_params['test_dates'], x_test, color='b')

  # Plot predicted data
  print(best_params['test_output'][0])
  predicted_data = best_params['scaler'].inverse_transform(best_params['test_output'][0])
  predicted_data = [np.max(arr) for arr in predicted_data]
  plt.plot(best_params['test_dates'], predicted_data, color='g')

  plt.title("Daily S&P 500 Index (sequenced)")
  plt.legend(['Train', 'Test', 'Predicted'])
  plt.xlabel('Date')
  plt.ylabel('Index Price (USD)')
  plt.savefig(filepath)

# Plot the training errors
def plot_errors(filepath, best_params):
  fig3 = plt.subplots(figsize=(5, 5))
  x = np.arange(0, best_params['num_epochs'])
  plt.plot(x, best_params['train_errs'], "r")

  plt.title("Training Errors Across {} Epochs".format(best_params['num_epochs']))
  plt.xlabel("Epochs")
  plt.ylabel("Error")
  plt.savefig(filepath)

if __name__ == '__main__':

  # Try out different configurations of parameters via hyperparameter tuning
  parameter_values = hyperparameter_tuning()

  # Extract the best parameters based on the lowest average Training and Testing Errors
  best_params = best_parameters(parameter_values)

  # Plot the original S&P 500 training and testing data without sequencing
  plot_original('ML-Stock-Market-Prediction/plots/original_plot.png', best_params)

  # Plot the S&P 500 training, testing and predicted sequenced data
  plot_sequenced('ML-Stock-Market-Prediction/plots/sequenced_plot.png', best_params)

  # Plot the training errors
  plot_errors('ML-Stock-Market-Prediction/plots/training_errors.png', best_params)