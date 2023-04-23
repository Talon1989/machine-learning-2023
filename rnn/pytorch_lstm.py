import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from CustomUtilities import create_sequential_dataset
from CustomUtilities import create_sequential_dataset_2
from CustomUtilities import print_graph
import torch
import torch.nn as nn
import torch.utils.data as data


torch.manual_seed(42)
dataframe = pd.read_csv('../data/airline_passengers.cvs', usecols=[1], engine='python').dropna()
dataset = dataframe.values.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)


# train_size = int(len(dataset) * 0.67)
X_train, X_test = train_test_split(dataset, train_size=1/2, shuffle=False)
sequence_length = 1

# X_train_seq, Y_train_next = create_sequential_dataset(dataset=X_train, look_back=sequence_length)
# X_test_seq, Y_test_next = create_sequential_dataset(dataset=X_test, look_back=sequence_length)
# X_train_seq = np.reshape(X_train_seq, [X_train_seq.shape[0], 1, X_train_seq.shape[1]])
# X_test_seq = np.reshape(X_test_seq, [X_test_seq.shape[0], 1, X_test_seq.shape[1]])

X_train_seq, Y_train_next = create_sequential_dataset_2(dataset=X_train, look_back=sequence_length)
X_test_seq, Y_test_next = create_sequential_dataset_2(dataset=X_test, look_back=sequence_length)


#  reshape input to [samples, time_steps, features]
X_train_seq = torch.tensor(X_train_seq)
Y_train_next = torch.tensor(Y_train_next)
X_test_seq = torch.tensor(X_test_seq)
Y_test_next = torch.tensor(Y_test_next)


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(in_features=50, out_features=1)

    def forward(self, x):
        x, _ = self.lstm(x)
        return self.linear(x)


model = Model()
optimizer = torch.optim.Adam(params=model.parameters())
loss_function = nn.MSELoss()
loader = data.DataLoader(data.TensorDataset(X_train_seq, Y_train_next), shuffle=True, batch_size=8)
epochs = 2_000
for e in range(1, epochs+1):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_function(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if e % 100 != 0:
        continue
    model.eval()
    with torch.no_grad():
        train_error = np.sqrt(loss_function(model(X_train_seq), Y_train_next))
        test_error = np.sqrt(loss_function(model(X_test_seq), Y_test_next))
    print('Epoch %d | Train RMSE %.4f | Test RMSE %.4f' % (e, train_error, test_error))


with torch.no_grad():
    predictions = model(X_test_seq)
    print_graph(np.squeeze(predictions), np.squeeze(Y_test_next), 'predicted data', 'true data', 'LSTM', 'month', False)


