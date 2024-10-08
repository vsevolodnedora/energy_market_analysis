'''
    Stateless multi-output timeseries forcasting model using LSTM Pytorch.

    This code and its description were composed by ChatGPT o1 preview with
    my guidance and refactoring
'''
import copy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from datetime import datetime, timedelta
from itertools import product
from glob import glob
import pickle
import json
import os
import re

class Metadata:
    def __init__(self, file_path:str='./metadata.json'):
        self.file_path = file_path

    def dump(self, data:dict):
        """
        Updates a JSON log file with a dictionary of data.
        If the file exists, it reads the existing content, removes the file, then writes the updated data.
        If the file does not exist, it creates a new file and writes the data.

        :param filename: Name of the file to update.
        :param data: Dictionary containing the data to append.
        """
        existing_data = {}
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as file:
                existing_data = json.load(file)
            os.remove(self.file_path)  # Remove the existing file

        # Update the dictionary with new data
        existing_data.update(data)

        # Write the updated dictionary back to the file
        with open(self.file_path, 'w') as file:
            json.dump(existing_data, file, indent=4, default=str)  # Handling datetime objects if necessary

    def get(self, key:str):
        """
        Retrieves a value from a JSON log file based on the provided key.

        :param filename: Name of the JSON file to read.
        :param key: Key for the value to retrieve.
        :return: The value associated with the key, or None if key is not found.
        """
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as file:
                data = json.load(file)
            return data.get(key)
        else:
            return None  # File does not exist

    def check(self, key:str, val:any):
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as file:
                original_data = json.load(file)
            if not key in original_data:
                raise KeyError(f'Key {key} does not exist in metadata.json')
            self.compare_dicts({key:original_data[key]}, {key:val})
        else:
            raise FileNotFoundError(f"Metadata file does not exist: {self.file_path}")

    @staticmethod
    def compare_dicts(dict_original, dict_new, path='root'):
        for key in set(dict_original.keys()) | set(dict_new.keys()):
            if key not in dict_original:
                raise KeyError(f"Key '{key}' missing in original at path '{path}'")
            elif key not in dict_new:
                raise KeyError(f"Key '{key}' missing in new at path '{path}'")
            else:
                val1 = dict_original[key]
                val2 = dict_new[key]
                key_path = f"{path}->{key}"
                Metadata.compare_values(key_path, val1, val2)
    @staticmethod
    def compare_values(path, val1, val2):
        if isinstance(val1, dict) and isinstance(val2, dict):
            Metadata.compare_dicts(val1, val2, path)
        elif isinstance(val1, list) and isinstance(val2, list):
            Metadata.compare_lists(val1, val2, path)
        else:
            if val1 != val2:
                raise KeyError(f"Difference at '{path}': {val1} != {val2}")
    @staticmethod
    def compare_lists(list1, list2, path):
        if len(list1) != len(list2):
            raise KeyError(f"Difference in list length at '{path}': {len(list1)} != {len(list2)}")
        for idx, (item1, item2) in enumerate(zip(list1, list2)):
            item_path = f"{path}[{idx}]"
            if isinstance(item1, dict) and isinstance(item2, dict):
                Metadata.compare_dicts(item1, item2, item_path)
            elif isinstance(item1, list) and isinstance(item2, list):
                Metadata.compare_lists(item1, item2, item_path)
            else:
                if item1 != item2:
                    raise KeyError(f"Difference at '{item_path}': {item1} != {item2}")

class LSTMForecast(nn.Module):
    def __init__(self, window_size, input_size, hidden_size, num_layers, output_size, dropout):
        super(LSTMForecast, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size * window_size, output_size)


    def forward(self, x):
        '''
        h_0 and c_0 are the initial hidden state and cell state tensors for the LSTM.
        num_layers: The number of stacked LSTM layers.
        x.size(0): The batch size (number of sequences in the batch).

        :param x:
        :return:
        '''
        # Initialization of Hidden and Cell States:
        # By initializing h_0 and c_0 to zeros every time, we treat each sequence in the batch independently.
        # This is appropriate when sequences are not continuous across batches.
        device = x.device  # Get the device from the input tensor
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # Passing Input Through the LSTM Layer.
        # x: Input tensor of shape (batch_size, sequence_length, input_size).
        # By providing (h_0, c_0), we ensure that each batch starts with zeroed states.
        out, _ = self.lstm(x, (h_0, c_0))
        # out has (batch_size, sequence_length * hidden_size).
        # out.size(0): The batch size.
        # -1 Flattens the remaining dimensions (sequence_length * hidden_size).
        # Prepares the data for the fully connected layer by converting it from a sequence of hidden states to a single feature vector per sample.
        out = out.reshape(out.size(0), -1)
        # A linear layer mapping the flattened LSTM outputs to the desired output size (forecast horizon).
        # Each neuron in the output layer corresponds to a time step in the forecast horizon.
        out = self.fc(out)
        return out

def preprocess_dataframe(df:pd.DataFrame)->pd.DataFrame:
    df = df.copy()  # Make a copy to avoid SettingWithCopyWarning
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    # Create cyclical features for hour and day of week
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    # Drop the original 'hour' and 'dayofweek' columns
    df.drop(['hour', 'dayofweek'], axis=1, inplace=True)
    return df

# Function to create sequences for LSTM
def create_sequences(idx_target, features_scaled, window_size, horizon):
    X = []
    y = []
    for i in range(len(features_scaled) - window_size - horizon + 1):
        X.append(features_scaled[i: (i + window_size), :])
        y.append(features_scaled[(i + window_size) : (i + window_size + horizon), idx_target])
    return np.array(X), np.array(y)

def train_predict(pars:dict, df:pd.DataFrame,today:pd.Timestamp, output_dir:str):

    # train_cut = today + timedelta(hours=forecast_horizon) # to use weather forecasts

    # load latest dataframe
    if not os.path.isdir(output_dir):
        # train mode
        train = True
        print(f"Output directory {output_dir} does not exist. Training new model on {today}")
        os.mkdir(output_dir)
        log = Metadata(file_path=output_dir+"/metadata.json")
        log.dump({'training_datetime': today.isoformat(),
                  'start_date':pd.Timestamp(df.index[0]).isoformat(),
                  'end_date':pd.Timestamp(df.index[-1]).isoformat()})
        # save train-related metadata
        log.dump({'df_columns': df.columns.tolist()})
        log.dump({'pars': pars})
        # df = preprocess_dataframe(df[:train_cut])
        print("Metadata saved.")
    else:
        # predict mode
        train = False
        log = Metadata(file_path=output_dir+"/metadata.json")
        # check if metadata hasn't changed (error-proving)
        log.check('df_columns', df.columns.tolist())
        log.check('pars', pars)
        print("Metadata check successful.")

    if train:
        # Check if GPU is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Using device: {device} for Training')
    else:
        # Check if GPU is available
        device = torch.device("cpu")
        print(f'Using device: {device} for Inference')

    # ----------------- Data Preparation ----------------- #

    df = df[:today] # todo Use time-shifted weather forcast instead of cutting it out

    # add time-related features
    df = preprocess_dataframe(df=df)

    features = df.copy()

    # Split data into training and testing sets
    train_size = int(len(df) * 0.8)
    features_train = features.iloc[:train_size]
    features_test = features.iloc[train_size:]

    # Normalize features using StandardScaler
    if train:
        scaler = StandardScaler()
        features_train_scaled = scaler.fit_transform(features_train)
        with open(output_dir+'scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
    else:
        # Load the scaler from disk
        with open(output_dir+'scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
            features_train_scaled = scaler.transform(features_train)

    features_test_scaled = scaler.transform(features_test)

    # Scale the entire features DataFrame for future use
    features_scaled = scaler.transform(features)

    # Get the index of 'DA_auction_price' for later use
    feature_cols = list(features.columns)
    da_price_index = feature_cols.index(pars['target'])

    # Inverse transform the predictions and true values
    mean_da_price = scaler.mean_[da_price_index]
    scale_da_price = scaler.scale_[da_price_index]

    window_size = pars['window_size']
    horizon = pars['horizon']

    # Create sequences
    X_train, y_train = create_sequences(da_price_index, features_train_scaled, window_size, horizon)
    X_test, y_test = create_sequences(da_price_index, features_test_scaled, window_size, horizon)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).float()
    X_test_tensor = torch.from_numpy(X_test).float()
    y_test_tensor = torch.from_numpy(y_test).float()

    if train:
        # ----------------- LSTM Model Definition ----------------- #

        # Model parameters
        input_size = features_train.shape[1]
        output_size = horizon

        # Instantiate the model
        model = LSTMForecast(
            window_size, input_size,pars['hidden_size'], pars['num_layers'], output_size, pars['dropout']
        ).to(device)


        # ----------------- Training the Model ----------------- #

        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=pars['lr'])

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', factor=0.2, patience=5, min_lr=1e-5,verbose=True)

        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=pars['batch_size'], shuffle=True)

        # Variable to track the best model
        best_val_loss = float('inf')
        best_model_path = 'best.pth'

        # Lists to keep track of loss
        train_losses = []
        val_losses = []

        counter = 0
        early_stopping = 10
        num_epochs = pars['num_epochs']
        for epoch in range(num_epochs):
            model.train()
            total_train_loss = 0
            batches = 0
            for batch_X, batch_y in train_loader:
                # Move tensors to the configured device
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
                batches += 1
            average_train_loss = total_train_loss / batches
            train_losses.append(average_train_loss)

            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_test_tensor.to(device))
                val_loss = criterion(val_outputs, y_test_tensor.to(device))
            scheduler.step(val_loss)
            val_losses.append(val_loss.item())

            # Save the best model
            if val_loss.item() < best_val_loss:
                counter = 0
                best_val_loss = val_loss.item()
                torch.save(model, output_dir+best_model_path)
                print(f'Saved new best model with validation loss: {val_loss.item():.4f}')
            else:
                counter+=1
            # print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')

            # early stopping if validation loss is frozen or goes up
            if counter >= early_stopping:
                print(f"Early stopping triggered after {10} epochs with Val Loss capped at: {best_val_loss:.4f}")
                break

        print('Saving final model as .pth')
        model = model.to('cpu')
        torch.save(model, output_dir+'final.pth')
        # Save the losses to a .txt file
        with open(output_dir+'training_validation_loss.txt', 'w') as f:
            f.write('# Epoch\tTraining Loss\tValidation Loss\n')
            for i in range(len(train_losses)):
                f.write(f'{i+1}\t{train_losses[i]:.4f}\t{val_losses[i]:.4f}\n')

        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_dir+'/losses.png',dpi=300)
        # plt.show()

        # ----------------- Model Evaluation ----------------- #

        print('Saving final model as .pth')
        model = torch.load(output_dir+'best.pth', map_location=torch.device('cpu'))
        # or
        # model = LSTMForecast(input_size, hidden_size, num_layers, output_size)
        # model.load_state_dict(torch.load(output_dir+'lstm_forecast_model_state_dict.pth'))
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test_tensor).cpu().numpy()
            y_true = y_test_tensor.numpy()


        y_pred_inv = y_pred * scale_da_price + mean_da_price
        y_true_inv = y_true * scale_da_price + mean_da_price

        # Calculate metrics
        mae = mean_absolute_error(y_true_inv, y_pred_inv)
        mse = mean_squared_error(y_true_inv, y_pred_inv)
        r2 = r2_score(y_true_inv, y_pred_inv)
        msg = f'Test MSE: {mse:.4f} MAE: {mae:.4f} R2 Score: {r2:.4f} Val Loss: {best_val_loss:.4f}'
        print(msg)
        with open(output_dir+'metrics.txt', 'w') as f:
            f.write(msg)


        # ----------------- Plotting Results ----------------- #

        # Get the last sample from test set
        last_X = X_test_tensor[-1].unsqueeze(0)
        last_y_true = y_test_tensor[-1].numpy()
        # last_y_pred = model(last_X).detach().numpy()
        with torch.no_grad():
            last_y_pred = model(last_X).cpu().numpy()

        # Inverse transform
        last_y_true_inv = last_y_true * scale_da_price + mean_da_price
        last_y_pred_inv = last_y_pred * scale_da_price + mean_da_price

        # Get the corresponding historical data
        last_history = last_X.cpu().squeeze().numpy()[:, da_price_index]
        last_history_inv = last_history * scale_da_price + mean_da_price

        # Plot historical data and forecasts
        plt.figure(figsize=(14, 7))
        plt.plot(range(window_size), last_history_inv, label='History Window')
        plt.plot(range(window_size, window_size + horizon), last_y_true_inv, label='True')
        plt.plot(range(window_size, window_size + horizon), last_y_pred_inv.squeeze(), label='Predicted')
        plt.xlabel('Time Steps')
        plt.ylabel(pars['target'])
        plt.title(msg)
        plt.legend()
        plt.savefig(output_dir+'/performance.png',dpi=300)
        # plt.show()
    else:
        print('Loading best model')
        model = torch.load(output_dir+'best.pth',map_location=torch.device('cpu'))

    # -------- Predict for One Horizon Before Last Timestamp -------- #

    # Get the index to start the input sequence one horizon before the last timestamp
    last_index_before = len(features_scaled) - horizon - window_size
    last_input_seq_before = features_scaled[last_index_before : last_index_before + window_size, :]

    # Convert to tensor
    last_input_seq_before_tensor = torch.from_numpy(np.expand_dims(last_input_seq_before, axis=0)).float()

    # Make prediction
    with torch.no_grad():
        past_pred = model(last_input_seq_before_tensor).cpu().numpy()

    # Inverse transform
    past_pred_inv = past_pred * scale_da_price + mean_da_price

    # Get the timestamps
    past_timestamps = df.index[last_index_before + window_size : last_index_before + window_size + horizon]

    # Create DataFrame
    past_forecast_df = pd.DataFrame({
        'date': past_timestamps,
        pars['target']: past_pred_inv.squeeze()
    })


    # -------- Forecast for Horizon Beyond Last Timestamp -------- #

    # Generate future timestamps
    last_timestamp = df.index[-1]
    future_timestamps = pd.date_range(
        start=last_timestamp + pd.Timedelta(hours=1), periods=horizon, freq='h'
    )

    # Create a DataFrame for future timestamps
    future_df = pd.DataFrame(index=future_timestamps)

    future_df = preprocess_dataframe(df=future_df)

    # For other features, use the last known values
    last_known_values = df.iloc[-1]
    for col in df.columns:
        if col not in future_df.columns:
            future_df[col] = last_known_values[col]

    # Ensure the columns are in the same order
    future_df = future_df[features.columns]

    # Combine last window_size data points with future data
    combined_df = pd.concat([df.iloc[-window_size:], future_df])

    # Create features and scale them
    combined_features = combined_df.copy()
    combined_features_scaled = scaler.transform(combined_features)

    # Prepare the input sequence
    last_input_seq_after = combined_features_scaled[:window_size, :]
    last_input_seq_after = np.expand_dims(last_input_seq_after, axis=0)  # Shape (1, window_size, num_features)

    # Convert to tensor
    last_input_seq_after_tensor = torch.from_numpy(last_input_seq_after).float()

    # Make prediction
    with torch.no_grad():
        future_pred = model(last_input_seq_after_tensor).cpu().numpy()

    # Inverse transform the predictions
    future_pred_inv = future_pred * scale_da_price + mean_da_price

    # Create a DataFrame with predictions and timestamps
    future_forecast_df = pd.DataFrame({
        'date': future_timestamps,
        pars['target']: future_pred_inv.squeeze()
    })

    # -------- Combine and Save Forecasts -------- #

    fig,ax = plt.subplots(figsize=(14, 7),ncols=1,nrows=1)
    ax.plot(past_forecast_df['date'], past_forecast_df[pars['target']],color='gray',ls='--')
    ax.plot(future_forecast_df['date'], future_forecast_df[pars['target']],color='gray',ls='--')
    ax.plot(df.tail(n=window_size).index, df.tail(n=window_size)[pars['target']],color='black')
    ax.legend()
    ax.grid()
    plt.tight_layout()
    plt.savefig(output_dir+'forecast.png',dpi=300)
    # plt.show()

    # Combine the past forecast and future forecast
    combined_forecast_df = pd.concat([past_forecast_df, future_forecast_df], ignore_index=True)
    combined_forecast_df.set_index('date', inplace=True)
    # Save to CSV
    combined_forecast_df.to_csv(output_dir+'forecast.csv', index=True)
    print(f'Final forecast from {combined_forecast_df.index[0]} to {combined_forecast_df.index[-1]} is saved.')

def hyperparameter_grid_search(df:pd.DataFrame, horizon_size:int, today:pd.Timestamp, output_dir:str):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # Define parameter grid (parameters to iterate over)
    param_grid = {
        'window_size': [3*72, 2*72, 1*72],
        'hidden_size': [32, 48, 64],
        'num_layers': [1, 2, 3],
        'lr': [0.01, 0.008, 0.004, 0.001],
        'batch_size': [32, 48, 64],
        'dropout': [0.1, 0.2, 0.3],
    }

    # default parameter dictionary (some remain constant)
    pars = dict(
        target='DA_auction_price',
        window_size=3*72,   # Historical window size
        horizon=horizon_size, # Forecast horizon
        hidden_size = 32,
        num_layers = 2,
        dropout = 0.2,
        lr = 0.01,
        num_epochs = 40,
        batch_size = 64,
        early_stopping=15
    )

    # Generate all combinations of hyperparameters
    param_combinations = list(product(*param_grid.values()))
    param_names = list(param_grid.keys())

    for i, params in enumerate( param_combinations ):

        pars_run = copy.deepcopy(pars)
        # generate par dict for the run
        params_dict = dict(zip(param_names, params))
        for key, val in params_dict.items():
            pars_run[key] = val

        # run the model
        train_predict(
            pars=pars_run,df=df,today=today,
            output_dir=output_dir+f'run_{i}/'
        )

        # update the score file
        # Initialize a list to store dictionaries
        data_list = []
        current_runs = glob(output_dir + 'run*')
        for run in current_runs:
            # Initialize an empty dictionary to store the extracted values
            data_dict = {}
            # Open and read the text file
            with open(run+'/metrics.txt', 'r') as file:
                for line in file:
                    # Remove leading and trailing whitespaces
                    line = line.strip()

                    # Define a regex pattern to match the desired values
                    pattern = (r'MSE:\s*(-?[0-9.]+)\s+MAE:\s*(-?[0-9.]+)\s+R2 Score:\s*(-?[0-9.]+)\s+Val Loss:\s*(-?[0-9.]+)')

                    # Search for the pattern in the current line
                    match = re.search(pattern, line)

                    # If a match is found, extract the values and store them in the dictionary
                    if match:
                        data_dict['MSE'] = float(match.group(1))
                        data_dict['MAE'] = float(match.group(2))
                        data_dict['R2'] = float(match.group(3))
                        data_dict['Loss'] = float(match.group(4))
            data_dict['run']=run.split('/')[-1]
            data_list.append(data_dict)
        # Create a DataFrame from the list of dictionaries
        df_stats = pd.DataFrame(data_list)
        # Set the first column to be the file name
        df_stats.set_index('run', inplace=True)
        # Save the DataFrame to a CSV file
        csv_path = output_dir+'output_metrics.csv'
        df_stats.to_csv(csv_path, index=True)

if __name__ == '__main__':
    # todo Add tests
    pass



