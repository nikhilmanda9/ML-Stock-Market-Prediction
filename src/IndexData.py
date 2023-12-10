import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Implementation of a class to predict the S&P 500 Index
class IndexData():

  def __init__(self, dataset, sequence_length):
    self.sequence_length = sequence_length

    # Load and preprocess the data from the given link (to a GitHub repo file)
    self.market_data = pd.read_parquet(dataset)
    self.preprocess_data()

    # Set the divisor values from https://ycharts.com/indicators/sp_500_divisor#:~:text=S%26P%20500%20Divisor%20is%20at,0.23%25%20from%20one%20year%20ago
    # The divisor is used to calculate the S&P 500 Index based on the total
    # market cap for the given quarter and is in the millions
    self.divisors = {2013: {'Q1': 8934.61, 'Q2': 8908.33, 'Q3': 8908.77, 'Q4': 8896.86},
                     2014: {'Q1': 8924.03, 'Q2': 8919.22, 'Q3': 8878.54, 'Q4': 8882.35},
                     2015: {'Q1': 8861.60, 'Q2': 8851.25, 'Q3': 8830.99, 'Q4': 8810.57},
                     2016: {'Q1': 8757.39, 'Q2': 8718.72, 'Q3': 8667.94, 'Q4': 8643.59},
                     2017: {'Q1': 8606.26, 'Q2': 8581.63, 'Q3': 8567.30, 'Q4': 8565.26},
                     2018: {'Q1': 8535.74, 'Q2': 8518.41, 'Q3': 8474.31, 'Q4': 8434.96}}

    # Calculate the S&P 500 Index
    self.extract_index_data()

    # Split the data into training and testing (we'll use the first ~80% of prices
    # as training and the remaining ~20% as testing)
    self.train = self.index_data[:1000].iloc[:,2:].values
    self.test = self.index_data[1000:].iloc[:,2:].values

    # Standardize the Index data
    self.standardize()

    # Generate input and target sequences for the training and testing data
    self.x_train, self.y_train, self.x_test, self.y_test = self.generate_sequences()

  # Preprocess the given dataset
  def preprocess_data(self):
    # Drop all unnecessary columns
    # We will only need the date, closing price, and volume
    self.market_data.drop(columns=['open', 'high', 'low', 'Name'], inplace=True)
    column_names = self.market_data.columns

    # Remove all rows with empty or missing values
    self.market_data.dropna(inplace=True)

    # Drop any duplicate rows
    self.market_data.drop_duplicates(inplace=True)

    # Convert the date column to datetime format
    self.market_data['date'] = pd.to_datetime(self.market_data['date'], format='%Y-%m-%d')

  # Calculate the S&P 500 Index using the closing price for every day
  def extract_index_data(self):
    # Calculate the numerator to calculate the Index and group by the date
    self.market_data['index_numerator'] = self.market_data['close'] * self.market_data['volume']
    self.market_data.drop(columns=['close', 'volume'], inplace=True)
    self.index_data = self.market_data.groupby(['date']).sum().reset_index()

    # Apply the appropriate divisor based on the quarter
    def apply_divisor(row):
      # Q1 for January-March
      if row['date'].month in [1,2,3]:
        quarter = 'Q1'
      # Q2 for April-June
      elif row['date'].month in [4,5,6]:
        quarter = 'Q2'
      # Q3 for July-September
      elif row['date'].month in [7,8,9]:
        quarter = 'Q3'
      # Q4 for October-December
      else:
        quarter = 'Q4'
      return float(row['index_numerator']) / (self.divisors[row['date'].year][quarter] * 1e4)

    # Apply the divisor for every date's numerator
    self.index_data['index'] = self.index_data.apply(apply_divisor, axis=1)

  # Standardize the index data
  def standardize(self):
    self.scaler = MinMaxScaler(feature_range=(0,1))
    self.train = self.scaler.fit_transform(self.train)
    self.test = self.scaler.fit_transform(self.test)

  # Generate input and target sequences for the training and testing data
  def generate_sequences(self):
    x_train_sequences, y_train_sequences, x_test_sequences, y_test_sequences = [], [], [], []

    # Training sequences
    for i in range(self.sequence_length, len(self.train) - 1):
      x_train_sequences.append(self.train[i-self.sequence_length : i,0])
      y_train_sequences.append(self.train[i : i+1,0])

    # Testing sequences
    for i in range(self.sequence_length, len(self.test) - 1):
      x_test_sequences.append(self.test[i-self.sequence_length : i,0])
      y_test_sequences.append(self.test[i : i+1,0])

    # Convert all sequences to numpy arrays
    x_train_sequences = np.array(x_train_sequences)
    y_train_sequences = np.array(y_train_sequences)
    x_test_sequences = np.array(x_test_sequences)
    y_test_sequences = np.array(y_test_sequences)

    # Reshape x data to fit as input to GRU
    x_train_sequences = np.reshape(x_train_sequences, (1, x_train_sequences.shape[0], x_train_sequences.shape[1]))
    x_test_sequences = np.reshape(x_test_sequences, (1, x_test_sequences.shape[0], x_test_sequences.shape[1]))

    return x_train_sequences, y_train_sequences, x_test_sequences, y_test_sequences