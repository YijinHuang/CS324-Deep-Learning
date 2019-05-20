from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
import numpy as np

import torch
from torch.utils.data import DataLoader

from dataset import PalindromeDataset
from vanilla_rnn import VanillaRNN


# default constants
INPUT_LENGTH = 5
INPUT_DIM = 10
NUM_CLASSES = 10
NUM_HIDDEN = 128
OUTPUT_DIM = 10
NUM_BATCH_SIZE = 128
LEARNING_RATE = 0.001
TRAIN_STEPS = 200
MAX_NORM = 10.0
TEST_SIZE = 100


def train(input_length, print_log):

    # Initialize the model that we are going to use
    model = VanillaRNN(input_length, INPUT_DIM, NUM_HIDDEN, OUTPUT_DIM, NUM_BATCH_SIZE).cuda()

    # Initialize the dataset and data loader (leave the +1)
    dataset = PalindromeDataset(input_length+1)
    data_loader = DataLoader(dataset, NUM_BATCH_SIZE, num_workers=1)

    # Setup the loss and optimizer
    cross_entropy = torch.nn.CrossEntropyLoss()  # fixme
    optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE)

    record_epochs, accs, losses = [], [], []
    for step, train_data in enumerate(data_loader):
        X, y = train_data
        X, y = X.cuda().long(), y.cuda()

        y_pred = model(X)
        loss = cross_entropy(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # the following line is to deal with exploding gradients
        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=MAX_NORM)

        if step % 10 == 0 and print_log:
            correct = 0
            total = 0
            with torch.no_grad():
                for i, test_data in enumerate(data_loader):
                    X, y = test_data
                    X, y = X.cuda().long(), y.cuda()

                    y_pred = model(X)
                    total += y.size(0)
                    _, predicted = torch.max(y_pred.data, 1)
                    correct += (predicted == y).sum().item()

                    if (i+1) % TEST_SIZE == 0:
                        break

            acc = round(correct / total, 4)
            avg_loss = round(loss.item(), 6)
            print('step: {}, loss: {}, test acc: {}'.format(
                step, avg_loss, acc
            ))
            record_epochs.append(step)
            accs.append(acc)
            losses.append(avg_loss)

        if step == TRAIN_STEPS:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    correct = 0
    total = 0
    with torch.no_grad():
        for i, test_data in enumerate(data_loader):
            X, y = test_data
            X, y = X.cuda().long(), y.cuda()

            y_pred = model(X)
            total += y.size(0)
            _, predicted = torch.max(y_pred.data, 1)
            correct += (predicted == y).sum().item()

            if (i+1) % TEST_SIZE == 0:
                break
        acc = round(correct / total, 4)

    print('Done training.')
    return record_epochs, accs, losses, acc

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--input_length', type=int, default=10, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)

    config = parser.parse_args()
    # Train the model
    train()