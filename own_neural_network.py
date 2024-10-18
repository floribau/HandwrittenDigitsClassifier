import numpy as np
from tensorflow.keras import datasets


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(0 > x, 0, 1)


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x)


def cross_entropy_loss(y, y_hat):
    return -np.sum(y * np.log(y_hat))


def he_distribution(n_in):
    return np.sqrt(2.0 / n_in)


def preprocessing(x_train, y_train, x_test, y_test):
    # normalize input data
    x_train, x_test = x_train / 255, x_test / 255
    # flatten input data
    x_train, x_test = x_train.reshape(x_train.shape[0], -1), x_test.reshape(x_test.shape[0], -1)

    # one-hot encode labels
    x_train, x_test = x_train.reshape(x_train.shape[0], -1), x_test.reshape(x_test.shape[0], -1)
    y_train, y_test = np.eye(10)[y_train], np.eye(10)[y_test]

    return x_train, y_train, x_test, y_test


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.a1 = None
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.w1 = np.random.randn(self.input_size, self.hidden_size) * he_distribution(self.input_size)
        self.w2 = np.random.randn(self.hidden_size, self.output_size) * he_distribution(self.hidden_size)

        self.b1 = np.zeros((1, self.hidden_size))
        self.b2 = np.zeros((1, self.output_size))

    def forward(self, x):
        z1 = np.dot(x, self.w1) + self.b1
        self.a1 = relu(z1)
        z2 = np.dot(self.a1, self.w2) + self.b2
        a2 = softmax(z2)
        return a2

    def backward(self, x, y, y_hat, learning_rate=0.01):
        # Gradients for output layer
        dz2 = y_hat - y
        dw2 = np.dot(self.a1.T, dz2)

        # Gradients for hidden layer
        dz1 = np.dot(dz2, self.w2.T) * relu_derivative(self.a1)
        dw1 = np.dot(x.T, dz1)

        # Update weights and biases
        self.w2 -= learning_rate * dw2
        self.b2 -= learning_rate * dz2
        self.w1 -= learning_rate * dw1
        self.b1 -= learning_rate * dz1

    def train(self, x, y, epochs=10, learning_rate=0.01):
        for epoch in range(epochs):
            sum_loss = 0
            for i in range(x.shape[0]):
                # flatten inputs
                x_sample = x[i:i+1]
                y_sample = y[i]

                y_hat = self.forward(x_sample)
                self.backward(x_sample, y_sample, y_hat, learning_rate)
                sum_loss += cross_entropy_loss(y_sample, y_hat)
            print(f"Epoch {epoch + 1}/{epochs}, cross entropy loss: {sum_loss/len(x)}")

    def predict(self, x):
        return self.forward(x)

    def test(self, x, y):
        accuracy = 0
        for i in range(x.shape[0]):
            # flatten inputs
            x_sample = x[i:i+1]
            y_sample = y[i]
            y_hat = self.predict(x_sample)[0]
            if np.argmax(y_hat) == np.argmax(y_sample):
                accuracy += 1.0
        return accuracy / x.shape[0]


nn = NeuralNetwork(28 * 28, 128, 10)
# load data
mnist = datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = x_train[:5000], y_train[:5000]
print("Starting preprocessing")
x_train, y_train, x_test, y_test = preprocessing(x_train, y_train, x_test, y_test)

print("Starting training")
nn.train(x_train, y_train, epochs=3)
print("Finished training")


print("\nStarting testing")
print(f"Accuracy: {nn.test(x_test, y_test)}")
