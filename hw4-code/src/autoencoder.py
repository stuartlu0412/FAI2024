import torch
from tqdm.auto import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

"""
Implementation of Autoencoder
"""
class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, encoding_dim: int) -> None:
        """
        Modify the model architecture here for comparison
        """
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.Linear(encoding_dim, encoding_dim//2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim//2, encoding_dim),
            nn.Linear(encoding_dim, input_dim),
        )
    
    def forward(self, x):
        #TODO: 5%
        return self.decoder(self.encoder(x))
    
    def fit(self, X, epochs=10, batch_size=32):
        #TODO: 5%
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        loss_function = nn.MSELoss()
        data_loader = DataLoader(
            dataset=TensorDataset(torch.tensor(X, dtype=torch.float32)),
            batch_size=batch_size,
            shuffle=True
        )

        for epoch in range(epochs):
            for X_batch in data_loader:
                optimizer.zero_grad()
                loss = loss_function(X_batch, self.forward(X_batch))
                loss.backward()
                optimizer.step()
    
    def transform(self, X):
        #TODO: 2%
        return self.encoder(X)
    
    def reconstruct(self, X):
        #TODO: 2%
        return self.decoder(self.encoder(X))


"""
Implementation of DenoisingAutoencoder
"""
class DenoisingAutoencoder(Autoencoder):
    def __init__(self, input_dim, encoding_dim, noise_factor=0.2):
        super(DenoisingAutoencoder, self).__init__(input_dim,encoding_dim)
        self.noise_factor = noise_factor
    
    def add_noise(self, x):
        #TODO: 3%
        mean = torch.zeros(x.size())
        std = torch.zeros(x.size()) + self.noise_factor
        return x + torch.normal(mean=mean, std=std)
    
    def fit(self, X, epochs=10, batch_size=32):
        #TODO: 4%

        optimizer = optim.Adam(self.parameters(), lr=0.001)
        loss_function = nn.MSELoss()
        data_loader = DataLoader(
                dataset=TensorDataset(torch.tensor(X, dtype=torch.float)),
                batch_size=batch_size,
                shuffle=False
        )

        for epoch in range(epochs):
            for batch in data_loader:
                batch_tensor = torch.cat([self.add_noise(x) for x in batch])
                optimizer.zero_grad()
                loss = loss_function(batch_tensor, self.forward(batch_tensor))
                loss.backward()
                optimizer.step()