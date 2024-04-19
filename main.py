from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models import VAE


def main(args):
    # Define the device (GPU or CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the CIFAR10 dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Create data loaders
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the VAE model
    vae = VAE().to(device)

    # Define the multi-task learning objective
    def multi_task_loss(recon_loss, kl_loss, condition_value):
        # Calculate the weighted sum of the reconstruction loss and kl loss
        loss = recon_loss + kl_loss
        return loss

    # Define the condition value for stabilizing gradients
    def condition_value(loss1, loss2):
        return torch.abs(loss1 - loss2) / (torch.abs(loss1) + torch.abs(loss2) + 1e-8)

    # Initialize the optimizer and scheduler
    optimizer = optim.Adam(vae.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Train the VAE model
    for epoch in range(args.epochs):
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(device)
            # Forward pass
            reconstructed, mu, log_var = vae(x)
            # Reconstruction loss (mean-square error)
            recon_loss = torch.mean((reconstructed - x) ** 2)
            # Beta-KL divergence loss
            beta = 1.0
            kl_loss = beta * -0.5 * torch.mean(1 + log_var - mu ** 2 - torch.exp(log_var))
            # Calculate the condition value
            cond = torch.exp(-torch.abs(recon_loss - kl_loss))
            # Compute the multi-task loss
            loss = multi_task_loss(recon_loss, kl_loss, condition_value)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            # Stabilize the gradients
            for param in vae.parameters():
                param.grad *= cond
            optimizer.step()
        # Update the scheduler
        scheduler.step()
        # Print the loss at each epoch
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    main(args)
