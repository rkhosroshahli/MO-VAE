from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.nn import functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models import VAE
from optim import AlignedMTLBalancer


# Define the multi-task learning objective
def multi_task_loss(recon_loss, kl_loss, condition_value):
    # Calculate the weighted sum of the reconstruction loss and kl loss
    loss = recon_loss + kl_loss
    return loss


# Define the condition value for stabilizing gradients
def condition_value(loss1, loss2):
    return torch.abs(loss1 - loss2) / (torch.abs(loss1) + torch.abs(loss2) + 1e-8)


def mse_recon_loss(data, logits):
    reconstructed_data = logits[0]
    loss = F.mse_loss(data, reconstructed_data, reduction='mean')
    return loss


# Beta-KL divergence loss
def bkl_loss(data, logits):
    mu, log_var = logits[1], logits[2]

    beta = 1.0
    loss = beta * -0.5 * torch.mean(1 + log_var - mu ** 2 - torch.exp(log_var))
    return loss


def train_step(model, train_dataset, optimizer, balancer, device):
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    criteria = {"reconstruction": mse_recon_loss, "kl": bkl_loss}
    # Train the VAE model
    model.train()
    loss_total, task_losses = 0, defaultdict(float)
    for batch_idx, (data, _) in enumerate(train_loader):

        balancer.step_with_model(
            data=data.to(device),
            model=model,
            criteria=criteria
        )
        # Backward pass
        optimizer.zero_grad()
        optimizer.step()
        losses = balancer.losses
        # if hasattr(balancer, 'info') and balancer.info is not None:
        #     fmtl_metrics.write(utils.strfy(balancer.info) + "\n")
        #     fmtl_metrics.flush()

        loss_total += sum(losses.values())
        for task_id in losses:
            task_losses[task_id] += losses[task_id]
    avg_total_loss = loss_total / len(train_loader)
    for task_id in task_losses:
        task_losses[task_id] /= len(train_loader)
    return avg_total_loss, task_losses


def main(args):
    # Define the device (GPU or CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the CIFAR10 dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Create data loaders
    batch_size = 128
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the VAE model
    model = VAE().to(device)

    # Initialize the optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1, verbose=True)

    balancer = AlignedMTLBalancer()

    train_metrics = []
    for epoch in range(args.epochs):
        avg_train_loss, avg_task_losses = train_step(model, train_dataset, optimizer, balancer, device)

        # Print the loss at each epoch
        print(f"Epoch: {epoch}, ", f"avg_train_loss: {avg_train_loss}, ", end=' ')
        for task_id in avg_task_losses:
            print('loss_{}: {:.6f}'.format(task_id, avg_task_losses[task_id]), end=', ')

        train_metrics.append({'train_loss': avg_train_loss, 'task_losses': avg_task_losses})
        # Update the scheduler
        scheduler.step()
        print(f'current LR: {scheduler.get_last_lr()}')
        #print()

    epochs = np.arange(1, args.epochs + 1)
    loss_values = [entry['train_loss'] for entry in train_metrics]

    plt.plot(epochs, loss_values, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train total loss')
    plt.grid(True)
    plt.show()

    loss_values = [entry['task_losses']['reconstruction'] for entry in train_metrics]
    plt.plot(epochs, loss_values, marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train reconstruction loss')
    plt.grid(True)
    plt.show()

    loss_values = [entry['task_losses']['kl'] for entry in train_metrics]
    plt.plot(epochs, loss_values, marker='^')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train KL-divergence loss')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()
    main(args)
