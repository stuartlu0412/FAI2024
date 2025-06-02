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
    def __init__(
        self,
        input_dim: int,
        encoding_dim: int,
        optimizer_setting: str = "Adam",
        architecture: str = "sequential"
    ) -> None:
        """
        Modify the model architecture here for comparison
        """
        super().__init__()

        if architecture == "sequential":
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, encoding_dim),
                nn.Linear(encoding_dim, encoding_dim//2),
                nn.ReLU()
            )
            self.decoder = nn.Sequential(
                nn.Linear(encoding_dim//2, encoding_dim),
                nn.Linear(encoding_dim, input_dim),
            )
        elif architecture == "linear":
   
            self.encoder = nn.Linear(input_dim, encoding_dim)
            self.decoder = nn.Linear(encoding_dim, input_dim)
        else:
            raise ValueError(f"Invalid architecture: {architecture}")

        if optimizer_setting == "Adam":
            self.optimizer = optim.Adam(self.parameters(), lr=0.01)
        elif optimizer_setting == "SGD":
            self.optimizer = optim.SGD(self.parameters(), lr=0.01)
        else:
            raise ValueError(f"Invalid optimizer: {optimizer_setting}")

    def forward(self, x):
        #TODO: 5%
        return self.decoder(self.encoder(x))
    
    def fit(self, X, epochs=10, batch_size=32):
        #TODO: 5%
        loss_function = nn.MSELoss()
        data_loader = DataLoader(
            dataset=TensorDataset(torch.tensor(X, dtype=torch.float32)),
            batch_size=batch_size,
            shuffle=True
        )

        for epoch in tqdm(range(epochs)):
            for X_batch in data_loader:
                self.optimizer.zero_grad()
                batch_tensor = torch.cat(X_batch)
                loss = loss_function(batch_tensor, self.forward(batch_tensor))
                loss.backward()
                self.optimizer.step()
    
    def transform(self, X):
        #TODO: 2%
        return self.encoder(torch.tensor(X, dtype=torch.float32)).detach().numpy()
    
    def reconstruct(self, X):
        #TODO: 2%
        return self.decoder(torch.tensor(self.transform(X), dtype=torch.float32)).detach().numpy()


"""
Implementation of DenoisingAutoencoder
"""
class DenoisingAutoencoder(Autoencoder):
    def __init__(
        self,
        input_dim,
        encoding_dim,
        noise_factor=0.2,
        optimizer_setting: str = "Adam",
        architecture: str = "sequential"
    ):
        super().__init__(input_dim, encoding_dim, optimizer_setting = optimizer_setting, architecture = architecture)
        self.noise_factor = noise_factor
    
    def add_noise(self, x):
        #TODO: 3%
        mean = torch.zeros(x.size())
        std = torch.zeros(x.size()) + self.noise_factor
        return x + torch.normal(mean=mean, std=std)
    
    def fit(self, X, epochs=10, batch_size=32):
        #TODO: 4%

        loss_function = nn.MSELoss()
        data_loader = DataLoader(
                dataset=TensorDataset(torch.tensor(X, dtype=torch.float)),
                batch_size=batch_size,
                shuffle=False
        )

        for epoch in tqdm(range(epochs)):
            for batch in data_loader:
                batch_tensor = torch.cat([self.add_noise(x) for x in batch])
                self.optimizer.zero_grad()
                loss = loss_function(batch_tensor, self.forward(batch_tensor))
                loss.backward()
                self.optimizer.step()