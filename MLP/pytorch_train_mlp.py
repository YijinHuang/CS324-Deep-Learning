from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os

import torch
from pytorch_mlp import MLP, CIFAR_MLP
from torch.utils.data import Dataset, DataLoader
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 200
EVAL_FREQ_DEFAULT = 10

FLAGS = None

# Custom constants
N_SAMPLES = 2000
N_FEATURES = 2
N_CLASSES = 4
TRAIN_SAMPLE_RATE = 0.8


class Data(Dataset):
    def __init__(self, X, y):
        super(Data, self).__init__()
        self.X = X
        self.y = y

    def __getitem__(self, ids):
        return self.X[ids], self.y[ids]

    def __len__(self):
        return self.X.shape[0]


def generate_moon(n_samples):
    X, y = datasets.make_moons(n_samples, True)
    return wrap_dataset(X, y)


def generate_data(n_samples, n_features, n_classes):
    X, y = datasets.make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1
    )
    return wrap_dataset(X, y)


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


def wrap_dataset(X, y, train_sample_rate=TRAIN_SAMPLE_RATE):
    X = X.astype(np.float32)
    y = y.astype(np.longlong)

    X = StandardScaler().fit_transform(X)

    sample_num = int(len(X) * train_sample_rate)
    X_train = X[:sample_num]
    X_test = X[sample_num:]
    y_train = y[:sample_num]
    y_test = y[sample_num:]

    raw_dataset = (X_train, X_test, y_train, y_test)
    demo(X_train, X_test, y_train, y_test)

    return Data(X_train, y_train), Data(X_test, y_test), raw_dataset


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
    pred = np.argmax(predictions)
    accuracy = (pred == targets).sum() / len(targets)
    return accuracy


def train(net, trainset, testset, n_features, n_hidden, n_classes,
          epochs, learning_rate, batch_size, eval_freq=EVAL_FREQ_DEFAULT):
    """
    Performs training and evaluation of MLP model.
    NOTE: You should the model on the whole test set each eval_freq iterations.
    """
    record_epochs, accs, losses = [], [], []
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, drop_last=False)
    model = net(n_features, n_hidden, n_classes).cuda()

    cross_entropy = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

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


def demo(X_train, X_test, y_train, y_test):
    figure = plt.figure()

    cm_bright = ListedColormap(['#FF0000', '#0000FF', '#00FF00', '#00FFFF'])
    train_fig = plt.subplot(1, 2, 1)
    train_fig.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
    test_fig = plt.subplot(1, 2, 2, sharex=train_fig, sharey=train_fig)
    test_fig.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright)
    plt.show()


def main():
    """
    Main function
    """
    # trainset, testset = generate_moon(N_SAMPLES, N_FEATURES, N_CLASSES)
    # n_hidden = [int(hidden) for hidden in DNN_HIDDEN_UNITS_DEFAULT.split()]
    # train(
    #     trainset,
    #     testset,
    #     N_FEATURES,
    #     n_hidden,
    #     N_CLASSES,
    #     MAX_EPOCHS_DEFAULT,
    #     LEARNING_RATE_DEFAULT
    # )
    trainset, testset = generate_CIFAR10()
    print('train dataset size: {}'.format(len(trainset)))
    print('test dataset size: {}'.format(len(testset)))
    n_hidden = [int(hidden) for hidden in '768 192'.split()]
    epochs, accs, losses = train(
        net=CIFAR_MLP,
        trainset=trainset,
        testset=testset,
        n_features=3*32*32,
        n_hidden=n_hidden,
        n_classes=10,
        epochs=MAX_EPOCHS_DEFAULT,
        learning_rate=LEARNING_RATE_DEFAULT,
        batch_size=256
    )


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_EPOCHS_DEFAULT,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    FLAGS, unparsed = parser.parse_known_args()
    main()
