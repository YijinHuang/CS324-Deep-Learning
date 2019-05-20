import numpy as np


class Linear(object):
    def __init__(self, in_features, out_features):
        """
        Module initialisation.
        Args:
            in_features: input dimension
            out_features: output dimension
        TODO
        1) Initialize weights self.params['weight'] using normal distribution with mean = 0 and 
        std = 0.0001.
        2) Initialize biases self.params['bias'] with 0. 
        3) Initialize gradients with zeros.
        """
        params = {}
        params['weight'] = np.random.normal(0, 0.01, (in_features, out_features))
        params['bias'] = np.zeros(out_features)

        self.params = params
        self.grads = {}
        self.x = None
        self.out = None

    def forward(self, x):
        """
        Forward pass (i.e., compute output from input).
        Args:
            x: input to the module
        Returns:
            out: output of the module
        Hint: Similarly to pytorch, you can store the computed values inside the object
        and use them in the backward pass computation. This is true for *all* forward methods of *all* modules in this class
        """
        W, b = self.params['weight'], self.params['bias']
        self.out = np.dot(x, W) + b
        self.x = x
        return self.out

    def backward(self, dout):
        """
        Backward pass (i.e., compute gradient).
        Args:
            dout: gradients of the previous module
        Returns:
            dx: gradients with respect to the input of the module
        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to
        layer parameters in self.grads['weight'] and self.grads['bias'].
        """
        self.grads['weight'] = np.dot(np.transpose(self.x), dout) / self.x.shape[0]
        self.grads['bias'] = np.mean(dout, axis=0)
        dx = np.dot(dout, np.transpose(self.params['weight']))
        return dx


class ReLU(object):
    def forward(self, x):
        """
        Forward pass.
        Args:
            x: input to the module
        Returns:
            out: output of the module
        """
        out = np.maximum(0, x)
        self.x = x
        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
            dout: gradients of the previous module
        Returns:
            dx: gradients with respect to the input of the module
        """
        dx = np.where(self.x > 0, dout, 0)
        return dx


class SoftMax(object):
    def forward(self, x):
        """
        Forward pass.
        Args:
            x: input to the module
        Returns:
            out: output of the module

        TODO:
        Implement forward pass of the module.
        To stabilize computation you should use the so-called Max Trick
        https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

        """
        b = x.max()
        y = np.exp(x-b)
        out = y / np.reshape(y.sum(axis=1), (-1, 1))
        self.out = out
        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
            dout: gradients of the previous module
        Returns:
            dx: gradients with respect to the input of the module
        """
        # reference: https://github.com/danielsabinasz/TensorSlow/blob/master/tensorslow/gradients.py
        dx = (dout - np.reshape(np.sum(dout * self.out, 1), [-1, 1])) * self.out
        return dx


class CrossEntropy(object):
    def forward(self, x, y):
        """
        Forward pass.
        Args:
            x: input to the module
            y: labels of the input
        Returns:
            out: cross entropy loss
        """
        out = np.sum(-np.log(np.maximum(x, 1e-8))*y) / x.shape[0]
        return out

    def backward(self, x, y):
        """
        Backward pass.
        Args:
            x: input to the module
            y: labels of the input
        Returns:
            dx: gradient of the loss with respect to the input x.
        """
        return -y/(np.maximum(x, 1e-8))


class Optimizer:
    def __init__(self, model, learning_rate, optimizer=''):
        self.lr = learning_rate
        self.trainable_layer = []
        for layer in model.layers:
            if isinstance(layer, Linear):
                self.trainable_layer.append(layer)

        if optimizer == 'momentum':
            self.update = self.optimize_with_momentum
        elif optimizer == 'RMSprop':
            self.update = self.optimize_RMSProb
        else:
            self.update = self.basic_optimize

    def optimize(self):
        weight_variables = {'v': 0}
        bias_variables = {'v': 0}
        for layer in self.trainable_layer:
            layer.params['weight'] -= self.update(layer.grads['weight'], weight_variables)
            layer.params['bias'] -= self.update(layer.grads['bias'], bias_variables)

    def basic_optimize(self, grad, variables):
        return self.lr * grad

    def optimize_with_momentum(self, grad, variables, gamma=0.9):
        variables['v'] = gamma * variables['v'] + self.lr * grad
        return variables['v']

    def optimize_RMSProb(self, grad, beta=0.9, e=1e-8):
        variables['v'] = beta * variables['v'] + (1 - beta) * grad**2
        return self.lr * grad / (variables['v']**0.5 + e)


if __name__ == "__main__":
    # softmax = SoftMax()
    # print(softmax.forward(np.array([[1, 2, 3]])))
    # print(softmax.backward(np.array([[0.2, 0.3, 0.1]])))
    ce = CrossEntropy()
    print(ce.forward([[0.1, 0.2, 0.7], [0.7, 0.2, 0.1]], [[1, 0, 0], [0, 0, 1]]))
    print(ce.backward(np.array([[0.1, 0.2, 0.7], [0.7, 0.2, 0.1]]), np.array([[1, 0, 0], [0, 0, 1]])))
