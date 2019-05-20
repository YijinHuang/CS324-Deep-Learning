import time
import random
import numpy as np
from matplotlib import pyplot as plt

# random.seed(10)
classes = [1, -1]


class Perceptron(object):

    def __init__(self, n_inputs, max_epochs=1e2, learning_rate=1e-4):
        """
        Initializes perceptron object.
        Args:
            n_inputs: number of inputs.
            max_epochs: maximum number of training cycles.
            learning_rate: magnitude of weight changes at each training cycle
        """
        self.n_inputs = n_inputs
        self.max_epochs = int(max_epochs)
        self.learning_rate = learning_rate

    def forward(self, input):
        """
        Predict label from input 
        Args:
            input: array of dimension equal to n_inputs.
        """
        label = np.sign(np.dot(input, self.w))
        return label

    def train(self, training_inputs, labels, optimizer='adam', batch_size=10):
        """
        Train the perceptron
        Args:
            training_inputs: list of numpy arrays of training points.
            labels: arrays of expected output value for the corresponding point in training_inputs.
        """
        self.train_size = int(len(training_inputs) * 0.75)
        x_train = training_inputs[:self.train_size]
        y_train = labels[:self.train_size]
        x_val = training_inputs[self.train_size:]
        y_val = labels[self.train_size:]

        self._normalization(x_train, x_val)
        self.w = np.random.standard_normal(self.n_inputs)
        self.best_w = 0
        # for optimizer
        self.v = 0
        self.m = 0

        if optimizer == 'momentum':
            optimize = self.optimize_with_momentum
        elif optimizer == 'adam':
            optimize = self.optimize_adam
        elif optimizer == 'RMSprop':
            optimize = self.optimize_RMSProb
        else:
            optimize = self.basic_optimize

        best_score = 0
        scores = []
        losses = []
        for epoch in range(1, self.max_epochs + 1):
            epoch_loss = 0
            candidates = [i for i in range(self.train_size)]
            random.shuffle(candidates)
            for step in range(self.train_size // batch_size):
                batch = [candidates.pop() for _ in range(batch_size)]
                x_batch = x_train[batch]
                y_batch = y_train[batch]

                for x, y in zip(x_batch, y_batch):
                    loss = self.loss(x, y)
                    epoch_loss += loss
                    if loss > 0:
                        grad = self.backward(x, y, loss)
                        self.w = optimize(grad, step)

            epoch_loss /= self.train_size
            val_acc = self.score(x_val, y_val)
            if val_acc > best_score:
                self.best_w = self.w
                best_score = val_acc
            if epoch % 10 == 0:
                print('epoch {}: loss {} val acc {}'.format(epoch, epoch_loss, val_acc))
            scores.append(val_acc)
            losses.append(epoch_loss)
        print('best val acc {}'.format(best_score))
        print('load best weight...')
        self.w = self.best_w
        return scores, losses

    def score(self, x, y):
        x_pred = np.where(self.forward(x) > 0, 1, -1)
        x_true = len(np.where(x_pred == y)[0])
        return x_true / len(y)

    def loss(self, x, y):
        return max(0, 1 - y * self.forward(x))

    def backward(self, x, y, loss):
        return (-y * x) if loss > 0 else 0

    def basic_optimize(self, grad, step):
        return self.w - self.learning_rate * grad

    def optimize_with_momentum(self, grad, step, gamma=0.9):
        self.v = gamma * self.v + self.learning_rate * grad
        return self.w - self.v

    def optimize_RMSProb(self, grad, step, beta=0.9, e=1e-8):
        self.v = beta * self.v + (1 - beta) * grad**2
        w = self.w - self.learning_rate * grad / (self.v**0.5 + e)
        return w

    def optimize_adam(self, grad, step, beta1=0.9, beta2=0.999, e=1e-8):
        self.m = beta1 * self.m + (1 - beta1) * grad
        self.v = beta2 * self.v + (1 - beta2) * grad**2
        m_ = self.m / (1 - beta1**step)
        v_ = self.v / (1 - beta2**step)
        w = self.w - self.learning_rate * m_ / (v_**0.5 + e)
        return w

    def _normalization(self, x_train, x_val):
        self.x_mean = np.mean(x_train, axis=0)
        self.x_std = np.std(x_train, axis=0)

        self.normalization(x_train)
        self.normalization(x_val)

    def normalization(self, xs):
        for i in range(len(xs)):
            x = xs[i]
            xs[i] = (x - self.x_mean) / (2 * self.x_std + 0.0001)


def data_generator(mean1, mean2, cov1, cov2, train_num, test_num):
    total_num = train_num+test_num
    x1 = np.random.multivariate_normal(mean1, cov1, total_num)
    x2 = np.random.multivariate_normal(mean2, cov2, total_num)
    y1 = [1 for _ in range(total_num)]
    y2 = [-1 for _ in range(total_num)]

    x1_train = x1[:train_num]
    y1_train = y1[:train_num]
    x2_train = x2[:train_num]
    y2_train = y2[:train_num]

    x1_test = x1[train_num:]
    y1_test = y1[train_num:]
    x2_test = x2[train_num:]
    y2_test = y2[train_num:]

    data_train = random_input(x1_train, x2_train, y1_train, y2_train)
    data_test = random_input(x1_test, x2_test, y1_test, y2_test)

    return data_train, data_test


def random_input(x1, x2, y1, y2):
    pack1 = list(zip(x1, y1))
    pack2 = list(zip(x2, y2))
    train_pack = pack1 + pack2
    random.shuffle(train_pack)

    xs = []
    ys = []
    for x, y in train_pack:
        xs.append(x)
        ys.append(y)

    return (np.array(xs), np.array(ys))


def main(data_train, data_test):
    max_epochs = 100

    perceptron = Perceptron(2, max_epochs=max_epochs)
    epochs = [i for i in range(1, int(max_epochs) + 1)]
    accs, losses = perceptron.train(data_train[0], data_train[1], optimizer='momentum')

    acc_plot = plt.plot(epochs, accs)
    plt.ylim([0, 1.1])
    plt.show()
    plt.close()
    losses = plt.plot(epochs, losses)
    plt.show()
    plt.close()

    perceptron.normalization(data_test[0])
    plt.grid(True)
    plt.scatter(data_test[0][:,0], data_test[0][:,1])
    line_x = [-1, 0, 1]
    slope = -perceptron.w[0]/perceptron.w[1]
    line_y = [slope*x for x in line_x]
    plt.plot(line_x, line_y, color='r')
    plt.show()

    acc = perceptron.score(data_test[0], data_test[1])
    print('Final test accuracy: {}'.format(acc))


if __name__ == "__main__":
    mean1 = [1, 1]
    mean2 = [7, 5]
    cov1 = [[1, 0], [0, 1]]
    cov2 = [[1, 0], [0, 1]]

    data_train, data_test = data_generator(mean1, mean2, cov1, cov2, 160, 40)
    main(data_train, data_test)
