import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import pandas as pd
from CustomUtilities import onehot_transformation, print_graph
from torch.utils.data import TensorDataset, DataLoader


dataset = pd.read_csv('data/iris.csv')
X_ = dataset.iloc[:, 0:-1].to_numpy()
y_ = dataset.iloc[:, -1].to_numpy()
y_ = LabelEncoder().fit_transform(y_)

X_train, X_test, y_train, y_test = train_test_split(X_, y_, train_size=1/2, shuffle=True, stratify=y_)


class Classifier:

    def __init__(self, n_input, h_shape, n_output, alpha=1/1_000, batch_size=2**5, dataloader_flag=False):
        self.n_input = n_input
        self.h_shape = h_shape
        self.n_output = n_output
        self.batch_size = batch_size
        self.dataloader_flag = dataloader_flag
        self.dnn = self._build_nn()
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(params=self.dnn.parameters(), lr=alpha)

    def _build_nn(self):
        model = nn.Sequential()
        for i in range(len(self.h_shape)):
            module = nn.Linear(self.h_shape[i-1] if i > 0 else self.n_input, self.h_shape[i])
            model.add_module(name='l_%d' % (i+1), module=module)
            model.add_module(name='a_%d' % (i+1), module=nn.ReLU())
        model.add_module(name='l_out', module=nn.Linear(self.h_shape[-1], self.n_output))
        model.add_module(name='a_out', module=nn.Sigmoid())
        return model

    def predict(self, x):
        x = torch.from_numpy(x).float()
        if len(x.shape) == 1:
            return torch.argmax(self.dnn(x))
        return torch.argmax(self.dnn(x), dim=1).numpy()

    def train(self, X, y, n_epochs=6_000):
        if self.dataloader_flag:
            self._train_w_dataloader(X, y, n_epochs)
        else:
            self._train_w_out_dataloader(X, y, n_epochs)

    def _train_w_out_dataloader(self, X, y, n_epochs):
        X = torch.from_numpy(X).float()
        Y = onehot_transformation(y)
        Y = torch.from_numpy(Y).float()
        losses, avg_losses = [], []
        for ep in range(1, n_epochs+1):
            batch = np.arange(len(y))
            loss_value = 0
            for idx in range(0, len(y), self.batch_size):
                minibatch = batch[idx: idx+self.batch_size]
                y_pred = self.dnn(X[minibatch])
                loss = self.criterion(y_pred, Y[minibatch])
                self.optimizer.zero_grad()
                loss.backward()
                loss_value += loss.detach.numpy()
                self.optimizer.step()
            losses.append(loss_value)
            avg_losses.append(np.sum(losses[-50:]) / len(losses[-50:]))
            if ep % 10 == 0:
                print('Episode %d | loss: %.3f' % (ep, loss_value))
            if ep % 1000 == 0:
                print_graph(losses, avg_losses, 'loss', 'avg loss', 'Pytorch Classifier on iris dataset')
        return self

    def _train_w_dataloader(self, X, y, n_epochs):
        Y = onehot_transformation(y)
        X = torch.from_numpy(X).float()
        Y = torch.from_numpy(Y).float()
        dataloader = DataLoader(dataset=TensorDataset(X, Y), batch_size=self.batch_size, shuffle=False)
        losses, avg_losses = [], []
        for ep in range(1, n_epochs+1):
            loss_value = 0
            for batch_x, batch_y in dataloader:
                y_pred = self.dnn(batch_x)
                loss = self.criterion(y_pred, batch_y)
                self.optimizer.zero_grad()
                loss.backward()
                loss_value += loss.detach().numpy()
                self.optimizer.step()
            losses.append(loss_value)
            avg_losses.append(np.sum(losses[-50:]) / len(losses[-50:]))
            if ep % 10 == 0:
                print('Episode %d | loss: %.3f' % (ep, loss_value))
            if ep % 1000 == 0:
                print_graph(losses, avg_losses, 'loss', 'avg loss', 'Pytorch Classifier on iris dataset')
        return self


classifier = Classifier(X_.shape[1], [2**4, 2**4, 2**5], 3, dataloader_flag=True)
classifier.train(X_train, y_train, n_epochs=3000)
y_pred = classifier.predict(X_test)
print('Accuracy: %.4f' % (np.sum(y_pred == y_test) / len(y_test)))

































































































































































































































































































































































































































































































































































































