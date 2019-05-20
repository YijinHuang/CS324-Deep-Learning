from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import random
import os
import bunch

from matplotlib import pyplot
from sklearn.datasets import make_moons

from .mymlp import MLP
from .modules import CrossEntropy, Optimizer

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 200
EVAL_FREQ_DEFAULT = 10

FLAGS = {
    'dnn_hidden_units': DNN_HIDDEN_UNITS_DEFAULT,
    'learning_rate': LEARNING_RATE_DEFAULT,
    'max_steps': MAX_EPOCHS_DEFAULT,
    'eval_freq': EVAL_FREQ_DEFAULT
}
FLAGS = bunch.Bunch(FLAGS)


def data_generate(size):
    X, y = make_moons(size, True)
    return X, y


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
    accuracy = len(np.where(predictions == targets)[0]) / len(targets)
    return accuracy


def train(mlp, optimizer, loss_func, train_X, train_y, test_X, test_y, batch_size):
    """
    Performs training and evaluation of MLP model.
    NOTE: You should the model on the whole test set each eval_freq iterations.
    """
    train_targets = np.argmax(train_y, axis=1)
    test_targets = np.argmax(test_y, axis=1)

    size = len(train_X)
    train_scores = []
    train_losses = []
    test_scores = []
    test_losses = []
    for epoch in range(1, FLAGS.max_steps + 1):
        epoch_loss = 0
        candidates = [i for i in range(size)]
        random.shuffle(candidates)
        epoch_loss = 0
        steps = size // batch_size
        for step in range(steps):
            batch = [candidates.pop() for _ in range(batch_size)]
            x_batch = train_X[batch]
            y_batch = train_y[batch]

            out = mlp.forward(x_batch)
            cel = loss_func.backward(out, y_batch)
            mlp.backward(cel)
            optimizer.optimize()

        train_out = mlp.forward(train_X)
        test_out = mlp.forward(test_X)
        train_predictions = np.argmax(train_out, axis=1)
        test_predictions = np.argmax(test_out, axis=1)

        train_loss = loss_func.forward(train_out, train_y)
        test_loss = loss_func.forward(test_out, test_y)
        train_acc = accuracy(train_predictions, train_targets)
        test_acc = accuracy(test_predictions, test_targets)

        if epoch % FLAGS.eval_freq == 0:
            print('epoch {}: loss: {} acc: {}'.format(
                epoch,
                round(test_loss, 6),
                round(test_acc, 4)
            ))

            train_scores.append(train_acc)
            train_losses.append(train_loss)
            test_scores.append(test_acc)
            test_losses.append(test_loss)

    return train_scores, train_losses, test_scores, test_losses


def main():
    """
    Main function
    """
    TRAIN_SIZE = 800
    TEST_SIZE = 200
    X, y = data_generate(TRAIN_SIZE+TEST_SIZE)
    label = np.zeros((len(y), 2))
    label[np.arange(len(y)), y] = 1
    train_data = X[:TRAIN_SIZE]
    train_label = label[:TRAIN_SIZE]
    test_data = X[TRAIN_SIZE:]
    test_label = label[TRAIN_SIZE:]

    hidden_units = list(map(int, FLAGS.dnn_hidden_units.split(',')))
    mlp = MLP(2, hidden_units, 2)
    optimizer = Optimizer(mlp, FLAGS.learning_rate, optimizer='RMSprop')
    loss_func = CrossEntropy()

    train(mlp, optimizer, loss_func, train_data, train_label, test_data, test_label, 32)
    print('Accuracy curve shown in notebook')


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
