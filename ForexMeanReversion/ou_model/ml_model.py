import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from ..config import HIDDEN_LAYERS, LEARNING_RATE, BATCH_SIZE, EPOCHS

class OUParamNN(nn.Module):
    def __init__(self, input_dim, output_dim=3, hidden_layers=[64,32,16]):
        super(OUParamNN, self).__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def train_nn(features, params):
    # features: NxF
    # params: Nx3 (theta, mu, sigma)
    X = torch.tensor(features, dtype=torch.float32)
    Y = torch.tensor(params, dtype=torch.float32)
    dataset = TensorDataset(X, Y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = OUParamNN(input_dim=features.shape[1], output_dim=3, hidden_layers=HIDDEN_LAYERS)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    for epoch in range(EPOCHS):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()
    return model

def predict_params(model, features):
    # features: NxF
    with torch.no_grad():
        X = torch.tensor(features, dtype=torch.float32)
        pred = model(X)
    return pred.numpy()