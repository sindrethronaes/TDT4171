import numpy as np
import statistics


def func(X: np.ndarray) -> np.ndarray:
    """
    The data generating function.
    Do not modify this function.
    """
    return 0.3 * X[:, 0] + 0.6 * X[:, 1] ** 2


def noisy_func(X: np.ndarray, epsilon: float = 0.075) -> np.ndarray:
    """
    Add Gaussian noise to the data generating function.
    Do not modify this function.
    """
    return func(X) + np.random.randn(len(X)) * epsilon


def get_data(n_train: int, n_test: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generating training and test data for
    training and testing the neural network.
    Do not modify this function.
    """
    X_train = np.random.rand(n_train, 2) * 2 - 1
    y_train = noisy_func(X_train)
    X_test = np.random.rand(n_test, 2) * 2 - 1
    y_test = noisy_func(X_test)

    return X_train, y_train, X_test, y_test


class Artificial_Neural_Network:
    """ Note: This model supports only 3 layers. 
    Can be generalized further to n layers if necessary"""

    # Network components
    input_Layer = 0
    hidden_Layer = 0
    output_Layer = 0
    weights = []
    gradients = []

    # To hold activation-values and output-values for each node
    products = []
    outputs = []

    # Layers specify how many nodes in each layer in a 3-layer model.
    # E.q: [2, 2, 1] has 2 input nodes, 2 hidden nodes and 1 output node
    def __init__(self, layers: list[int]) -> None:

        # Initialize size of layers
        self.input_Layer = layers[0]
        self.hidden_Layer = layers[1]
        self.output_Layer = layers[-1]

        # Initialize weights for each arc in the network
        self.weights.append(np.random.randn(
            self.input_Layer, self.hidden_Layer) / np.sqrt(self.input_Layer))
        self.weights.append(np.random.randn(
            self.hidden_Layer, self.output_Layer) / np.sqrt(self.hidden_Layer))

        # Initialize gradients
        self.gradients = [None] * len(self.weights)

    # Sigmoid activation function for hidden nodes
    def sigmoid_Activation(self, x: np.ndarray) -> int:
        return 1 / (1 + np.exp(-x))

    # Derivative of sigmoid function for backpropagation
    def sigmoid_Derivative(self, x: np.ndarray) -> int:
        return self.sigmoid_Activation(x) * (1 - self.sigmoid_Activation(x))

    # Unit-Step function for output-node
    def linear_Activation(self, x: np.ndarray) -> int:
        if np.sum(x) > 0:
            return 1
        else:
            return 0

    # Derivative of Unit-Step function for backpropagation
    def linear_Derivative(self, x: np.ndarray) -> int:
        return 1

    # Forward propagation algorithm aka actual regression
    def forward_Propagation(self, x: np.ndarray):

        # Feed INPUT LAYER values and store values for future use
        self.products.append(x)
        input = x

        # Calculate HIDDEN LAYER values
        for i in range(self.hidden_Layer):

            # Matrix multiplication of Inputs and Weights,
            # then store the product
            product = np.dot(input, self.weights[i])
            self.products.append(product)

            # Calculate the input for the next layer from the activation function of this layer,
            # then store the result
            input = self.sigmoid_Activation(product)
            self.outputs.append(input)

        # Calculate OUTPUT LAYER value
        product = np.dot(input, self.weights[-1].T)
        output = self.linear_Activation(product)

        # Returns the output from the forward propagation in the network
        return output

    # Backward propagation aka weight adjustment with gradient descent algorithm
    def backward_Propagation(self, x: np.ndarray, outputs: np.ndarray, target: np.ndarray, learning_rate: float):

        # Resets gradients before each backward propagation
        self.gradients = [None] * len(self.weights)

        # Calculate error
        output_Error = target - outputs

        # Calculate the delta for the output layer (DOES THIS WORK??)
        output_delta = output_Error * self.linear_Derivative(self.outputs[-1])

        # Calculate the gradient for the output layer,
        # then store the result
        output_gradients = np.dot(
            self.outputs[-1].reshape(-1, 1), output_delta.reshape(-1, 1).T)
        self.gradients[-1] = output_gradients

        # Calculate the delta for the hidden layer
        hidden_delta = np.dot(
            output_delta, self.weights[-1].T) * (self.outputs[-1] * (1 - self.outputs[-1]))

        # Calculate the gradient for the hidden layer,
        # then store the result
        hidden_gradients = np.dot(
            self.products[-2].reshape(-1, 1), hidden_delta.reshape(-1, 1).T)
        self.gradients[-2] = hidden_gradients

        # Actual update of all weights in the network
        for i in range(len(self.weights)):
            # print(self.weights[i])
            self.weights[i] = self.weights[i] - \
                learning_rate * self.gradients[i]

        # Resets activation-values and output-values for each node for next forward propagation
        self.products = []
        self.outputs = []


# Calculates the Squared Mean Error of a network given a data set
def calculate_MSE(network: Artificial_Neural_Network, x: np.ndarray, y: np.ndarray):

    # Calculate the output for the network
    output = network.forward_Propagation(x)

    # MSE is the mean of the squares of the errors (observed - predicted)
    MSE = (1 / len(x)) * sum(((y - output)**2))

    # Return the MSE
    return MSE


def train_Model(network: Artificial_Neural_Network, learning_Rate: float, epochs: int, x: np.ndarray, y: np.ndarray) -> None:

    # How many iterations used to train the network
    for i in range(int(epochs)):

        # Forward propagation
        output = network.forward_Propagation(x)

        # Backward propagation
        network.backward_Propagation(x, output, y, learning_Rate)

        # Calculate error
        error = calculate_MSE(network, x, y)


if __name__ == "__main__":
    np.random.seed(0)
    X_train, y_train, X_test, y_test = get_data(n_train=280, n_test=120)

    # TODO: Your code goes here.

    y_train = y_train.reshape(y_train.shape[0], 1)
    y_test = y_test.reshape(y_test.shape[0], 1)

    learning_Rate = 0.0025

    # Network Construction with 2 input nodes, 2 hidden nodes and 1 output node
    shape = [2, 2, 1]
    ann = Artificial_Neural_Network(shape)

    # Train the network
    learning_Rate = 0.0025
    # train_Model(ann, learning_Rate, 1e+5, X_train, y_train)

    # MSE: TRAINING
    mse_Training = calculate_MSE(ann, X_train, y_train)

    # MSE: TEST
    mse_Test = calculate_MSE(ann, X_test, y_test)

    # Print in Terminal
    print("MSE for Training Set is: " + str(mse_Training))
    print("MSE for Testing Set is: " + str(mse_Test))
