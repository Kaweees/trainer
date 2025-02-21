import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from trainer.training import send_metrics_to_dashboard


class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)


class TrainerState:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.iteration = 0
        self.loss = 0.0
        self.accuracy = 0.0
        self.bits_memorized = 0.0
        self.bits_per_second = 0.0
        self.skills = {
            "Translation": 20,
            "Summarization": 15,
            "Reasoning": 10,
            "Coding": 5,
            "Comprehension": 25,
        }
        self.start_time = time.time()


def get_gpu_utilization():
    if not torch.cuda.is_available():
        return [0.0] * 1024

    try:
        import pynvml

        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        utils = []
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            utils.append(util.gpu / 100.0)
        return utils + [0.0] * (1024 - len(utils))
    except (ImportError, pynvml.NVMLError) as e:
        print(f"Failed to get GPU utilization: {e}")
        return [0.0] * 1024


def train():
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load MNIST dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(
        "data", train=True, download=True, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Initialize model and optimizer
    model = MNISTNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Initialize trainer state
    trainer_state = TrainerState(model, optimizer)
    update_frequency = 10  # Update dashboard every 10 batches

    print("Starting MNIST training...")

    for epoch in range(10):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            accuracy = correct / len(data)

            # Update trainer state
            trainer_state.iteration += 1
            trainer_state.loss = loss.item()
            trainer_state.accuracy = accuracy
            trainer_state.bits_memorized = trainer_state.iteration * 0.5
            current_time = time.time()
            trainer_state.bits_per_second = trainer_state.bits_memorized / (
                current_time - trainer_state.start_time
            )
            trainer_state.gpu_utilization = get_gpu_utilization()
            # Update dashboard
            if batch_idx % update_frequency == 0:
                send_metrics_to_dashboard(trainer_state)

            if batch_idx % 100 == 0:
                print(
                    f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}"
                )
