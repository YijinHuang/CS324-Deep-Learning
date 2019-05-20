from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os

import torch
from torch.utils.data import Dataset, DataLoader
from cnn_model import VGG, VGG_CNF


# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_EPOCHS_DEFAULT = 500
EVAL_FREQ_DEFAULT = 25
OPTIMIZER_DEFAULT = 'ADAM'
DATA_DIR_DEFAULT = ''

FLAGS = None


def generate_CIFAR10():
    import torchvision
    from torchvision import transforms
    train_preprocess = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_set = torchvision.datasets.CIFAR10(root='../CIFAR10', train=True, transform=train_preprocess)
    test_set = torchvision.datasets.CIFAR10(root='../CIFAR10', train=False, transform=test_preprocess)
    return train_set, test_set


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e., the average of correct predictions
    of the network.
    Args:
        predictions: 2D float array of size [number_of_data_samples, n_classes]
        labels: 2D int array of size [number_of_data_samples, n_classes] with one-hot encoding of ground-truth labels
    Returns:
        accuracy: scalar float, the accuracy of predictions.
    """
    return accuracy


def train(net, trainset, testset, n_channels, layers, n_classes,
          epochs, learning_rate, batch_size, eval_freq=EVAL_FREQ_DEFAULT):
    """
    Performs training and evaluation of MLP model.
    NOTE: You should the model on the whole test set each eval_freq iterations.
    """
    record_epochs, accs, losses = [], [], []
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, drop_last=False)
    model = net(n_channels, layers, n_classes).cuda()

    cross_entropy = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, epochs+1):
        epoch_loss = 0
        for step, train_data in enumerate(train_loader):
            X, y = train_data
            X, y = X.cuda(), y.cuda()

            y_pred = model(X)
            loss = cross_entropy(y_pred, y)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % eval_freq == 0:
            correct = 0
            total = 0
            with torch.no_grad():
                for test_data in test_loader:
                    X, y = test_data
                    X, y = X.cuda(), y.cuda()

                    y_pred = model(X)
                    total += y.size(0)
                    _, predicted = torch.max(y_pred.data, 1)
                    correct += (predicted == y).sum().item()
            acc = round(correct / total, 4)
            avg_loss = round(epoch_loss / step+1, 6)
            print('epoch: {}, loss: {}, test acc: {}'.format(
                epoch, avg_loss, acc
            ))
            record_epochs.append(epoch)
            accs.append(acc)
            losses.append(avg_loss)

    return record_epochs, accs, losses


def main():
    """
    Main function
    """
    trainset, testset = generate_CIFAR10()
    print('train dataset size: {}'.format(len(trainset)))
    print('test dataset size: {}'.format(len(testset)))
    layers = VGG_CNF['B']
    train(
        net=VGG,
        trainset=trainset,
        testset=testset,
        n_channels=3,
        layers=layers,
        n_classes=10,
        epochs=MAX_EPOCHS_DEFAULT,
        learning_rate=LEARNING_RATE_DEFAULT,
        batch_size=128,
        eval_freq=EVAL_FREQ_DEFAULT
    )


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_EPOCHS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()

    main()
