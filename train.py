from pathlib import Path
import torch  # type: ignore
from torch import nn
from torch.utils.data import DataLoader  # type: ignore
from torchvision import datasets  # type: ignore
from torchvision.transforms import ToTensor  # type: ignore

from network import NeuralNetwork
from tqdm import tqdm  # type: ignore


class ImageClassifierTrainer:

    def __init__(self, weights_path: Path) -> None:
        self.weights_path = weights_path
        # Download training data from open datasets.
        self.training_data = datasets.FashionMNIST(
            root="data",
            train=True,
            download=True,
            transform=ToTensor(),
        )

        # Download test data from open datasets.
        self.test_data = datasets.FashionMNIST(
            root="data",
            train=False,
            download=True,
            transform=ToTensor(),
        )

        batch_size = 64

        # Create data loaders.
        self.train_dataloader = DataLoader(self.training_data, batch_size=batch_size)
        self.test_dataloader = DataLoader(self.test_data, batch_size=batch_size)

        for X, y in self.test_dataloader:
            print(f"Shape of X [N, C, H, W]: {X.shape}")
            print(f"Shape of y: {y.shape} {y.dtype}")
            break

        # Get cpu, gpu or mps device for training.
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"Using {self.device} device")

        # Define model

        self.model = NeuralNetwork().to(self.device)
        print(self.model)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)

    def train_epochs(self, epochs: int):
        if self.weights_path.exists():
            self.model.load_state_dict(torch.load(self.weights_path))
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            self.train()
            self.test()
        torch.save(self.model.state_dict(), self.weights_path)
        print("Done!")

    def train(self):
        size = len(self.train_dataloader.dataset)  # type:ignore
        self.model.train()
        for batch, (X, y) in tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader)):
            X, y = X.to(self.device), y.to(self.device)

            # Compute prediction error
            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        print(f"Loss: {loss.item()}")

    def test(self):
        size = len(self.test_dataloader.dataset)  # type: ignore
        num_batches = len(self.test_dataloader)
        self.model.eval()
        test_loss, correct = 0.0, 0.0
        with torch.no_grad():
            for X, y in self.test_dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
