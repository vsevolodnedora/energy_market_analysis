'''
    Stateless multi-output timeseries forcasting model using LSTM Pytorch.

    This code and its description were composed by ChatGPT o1 preview with
    my guidance and refactoring
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from datetime import datetime, timedelta
import pickle
import json
import os

output_dir = './res8/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

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
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
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
def create_sequences(da_price_index, features_scaled, window_size, horizon):
    X = []
    y = []
    for i in range(len(features_scaled) - window_size - horizon + 1):
        X.append(features_scaled[i: (i + window_size), :])
        y.append(features_scaled[(i + window_size) : (i + window_size + horizon), da_price_index])
    return np.array(X), np.array(y)

def train_predict(df:pd.DataFrame,today:pd.Timestamp, output_dir:str, train:bool):

    pars = dict(
        target='DA_auction_price',
        window_size=3*72,   # Historical window size
        horizon=72, # Forecast horizon
        hidden_size = 32,
        num_layers = 2,
        dropout = 0.3,
        lr = 0.01,
        num_epochs = 30,
        batch_size = 64,
        early_stopping=10,
    )

    # train_cut = today + timedelta(hours=forecast_horizon) # to use weather forecasts

    # load latest dataframe
    if not os.path.isdir(output_dir):
        # train mode
        print(f"Output directory {output_dir} does not exist. Training new model on {today}")
        os.mkdir(output_dir)
        log = Metadata(file_path=output_dir+"/metadata.json")
        log.dump({'training_datetime': today.isoformat()})
        # df = preprocess_dataframe(df[:train_cut])
    else:
        # predict mode
        log = Metadata(file_path=output_dir+"/metadata.json")

    log.dump(pars)

    # ----------------- Data Preparation ----------------- #

    # add time-related features
    df = preprocess_dataframe(df=df)
    features = df.copy()

    # Split data into training and testing sets
    train_size = int(len(df) * 0.8)
    features_train = features.iloc[:train_size]
    features_test = features.iloc[train_size:]

    # Normalize features using StandardScaler
    if train:
        with open(output_dir+'scaler.pkl', 'wb') as f:
            scaler = StandardScaler()
            features_train_scaled = scaler.fit_transform(features_train)
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
        model = LSTMForecast(window_size, input_size,
                             pars['hidden_size'], pars['num_layers'], output_size, pars['dropout'])


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
                val_outputs = model(X_test_tensor)
                val_loss = criterion(val_outputs, y_test_tensor)
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
        torch.save(model, output_dir+'final.pth')
        # Save the losses to a .txt file
        with open(output_dir+'training_validation_loss.txt', 'w') as f:
            f.write('# Epoch\tTraining Loss\tValidation Loss\n')
            for i in range(num_epochs):
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
        plt.show()

        # ----------------- Model Evaluation ----------------- #

        print('Saving final model as .pth')
        model = torch.load(output_dir+'best.pth')
        # or
        # model = LSTMForecast(input_size, hidden_size, num_layers, output_size)
        # model.load_state_dict(torch.load(output_dir+'lstm_forecast_model_state_dict.pth'))
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test_tensor).numpy()
            y_true = y_test_tensor.numpy()


        y_pred_inv = y_pred * scale_da_price + mean_da_price
        y_true_inv = y_true * scale_da_price + mean_da_price

        # Calculate metrics
        mae = mean_absolute_error(y_true_inv, y_pred_inv)
        mse = mean_squared_error(y_true_inv, y_pred_inv)
        r2 = r2_score(y_true_inv, y_pred_inv)
        msg = f'Test MSE: {mse:.4f} MAE: {mae:.4f} R2 Score: {r2:.4f}'
        print(msg)
        with open(output_dir+'metrics.txt', 'w') as f:
            f.write(msg)


        # ----------------- Plotting Results ----------------- #

        # Get the last sample from test set
        last_X = X_test_tensor[-1].unsqueeze(0)
        last_y_true = y_test_tensor[-1].numpy()
        last_y_pred = model(last_X).detach().numpy()

        # Inverse transform
        last_y_true_inv = last_y_true * scale_da_price + mean_da_price
        last_y_pred_inv = last_y_pred * scale_da_price + mean_da_price

        # Get the corresponding historical data
        last_history = last_X.squeeze().numpy()[:, da_price_index]
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
        plt.show()
    else:
        print('Loading best model')
        model = torch.load(output_dir+'best.pth')

    # -------- Predict for One Horizon Before Last Timestamp -------- #

    # Get the index to start the input sequence one horizon before the last timestamp
    last_index_before = len(features_scaled) - horizon - window_size
    last_input_seq_before = features_scaled[last_index_before : last_index_before + window_size, :]

    # Convert to tensor
    last_input_seq_before_tensor = torch.from_numpy(np.expand_dims(last_input_seq_before, axis=0)).float()

    # Make prediction
    with torch.no_grad():
        past_pred = model(last_input_seq_before_tensor).numpy()

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
        future_pred = model(last_input_seq_after_tensor).numpy()

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
    plt.show()

    # Combine the past forecast and future forecast
    combined_forecast_df = pd.concat([past_forecast_df, future_forecast_df], ignore_index=True)
    combined_forecast_df.set_index('date', inplace=True)
    # Save to CSV
    combined_forecast_df.to_csv(output_dir+'forecast.csv', index=True)
    print(f'Final forecast from {combined_forecast_df.index[0]} to {combined_forecast_df.index[-1]} is saved.')





