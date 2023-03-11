from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import torch
import tqdm


class Dataset(torch.utils.data.Dataset):
    def __init__(self, interaction_matrix: np.array):
        self.interaction_matrix = interaction_matrix
        self.n, self.r = interaction_matrix.shape

    def __len__(self):
        return self.n

    def __getitem__(self, index: int):
        return torch.tensor(self.interaction_matrix[index, :], dtype=torch.float32)


class Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        return torch.norm(target - output) ** 2


class LossSparse(torch.nn.Module):
    def __init__(self, interaction_matrix: np.array, mu: float):
        super().__init__()
        S = np.zeros_like(interaction_matrix)
        S[interaction_matrix == 0] = 1
        S[interaction_matrix == 1] = mu
        self.S = torch.nn.Parameter(torch.tensor(S, dtype=torch.float32))

    def forward(self, output, target):
        return torch.norm(torch.mul(target - output, self.S)) ** 2


class AutoEncoder(torch.nn.Module):
    def __init__(self, layers_sizes: List[int]):
        super().__init__()
        self.encoder_layers = torch.nn.ModuleList()
        for i in range(len(layers_sizes) - 1):
            self.encoder_layers.append(
                torch.nn.Linear(layers_sizes[i], layers_sizes[i + 1])
            )
        self.decoder_layers = torch.nn.ModuleList()
        for i in range(len(layers_sizes) - 1, 0, -1):
            self.decoder_layers.append(
                torch.nn.Linear(layers_sizes[i], layers_sizes[i - 1])
            )
        self.non_linear = torch.nn.Tanh()

    def encode(self, x):
        for i in range(len(self.encoder_layers)):
            x = self.non_linear(self.encoder_layers[i](x))
        return x

    def decode(self, x):
        for i in range(len(self.decoder_layers)):
            x = self.non_linear(self.decoder_layers[i](x))
        return x

    def forward(self, x):
        return self.decode(self.encode(x))

    def save(self, path: Path):
        torch.save(self.state_dict(), path)

    def load(self, path: Path):
        self.load_state_dict(torch.load(path))


class AutoEncoderTrainer:
    def __init__(
        self,
        interaction_matrix,
        layers_size,
        learning_rate,
        weight_decay,
        batch_size,
        device,
    ):
        self.losses = []
        self.train_dataset = Dataset(interaction_matrix)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True
        )
        self.model = AutoEncoder(layers_size).to(device)
        self.loss = Loss().to(device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.device = device

    def train(self, epoch):
        self.model.train()
        for _ in range(epoch):
            for batch in self.train_loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(batch)
                loss = self.loss(output, batch)
                loss.backward()
                self.optimizer.step()
                self.losses.append(loss.item())

    def encode(self):
        self.model.eval()
        with torch.no_grad():
            encoded = self.model.encode(
                torch.tensor(self.train_dataset.interaction_matrix, dtype=torch.float32)
            )
        return encoded

    def scores(self):
        encoded = self.encode()
        scores = torch.sigmoid(encoded @ encoded.T).detach().cpu().numpy()
        np.fill_diagonal(scores, 0)
        return scores

    def plot_losses(self, path):
        plt.figure()
        plt.plot(self.losses)
        plt.savefig(path / "losses.png")
