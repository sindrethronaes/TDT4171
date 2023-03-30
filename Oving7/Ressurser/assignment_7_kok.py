import numpy as np
import math

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

class Model:
    def __init__(self, layers: list[int]) -> None:

        self.activation_function_hidden_layers      = lambda z: 1 / (1 + np.exp(-z))
        self.activation_function_hidden_layers_der  = lambda z: self.activation_function_hidden_layers(z) * (1 - self.activation_function_hidden_layers(z))
        self.activation_function_output_layer       = lambda z: z

        self.I = layers[0]                      # Input 
        self.neurons_pr_layer = layers[1:-1]    # Hidden layers
        self.ws = []                            # All weights

        # Initiate weights (Report)
        prev_neurons = self.I
        for curr_neurons in layers[1:]:
            w_shape = (prev_neurons, curr_neurons)
            w = np.random.normal(0, 1/np.sqrt(prev_neurons), (w_shape)) # Normal distrebution dependant on number of inputs
            self.ws.append(w)

            prev_neurons = curr_neurons

        # Initiate/allocate gradients (Are set for every backpropagation)
        self.gradients = [None for i in range(len(self.ws))] 

        # Allocating variables to save forwardpass information [makes it easier to calculate the backward pass]
        self.zs = [] 
        self.activations = []

    def forward(self, X: np.ndarray):
        self.zs = []
        self.activations = []

        # Input ["First layer"]
        activation = X
        self.activations.append(activation)         # Save intermediate steps to make BackProp easier

        # Hidden Layers ["Hidden Layers"]
        for i in range(len(self.neurons_pr_layer)):
            z = activation @ self.ws[i]
            activation = self.activation_function_hidden_layers(z)

            self.zs.append(z)                       # Save intermediate steps to make BackProp easier
            self.activations.append(activation)     # Save intermediate steps to make BackProp easier

        # Output ["Output Layer"]
        z = activation @ self.ws[-1]
        output = self.activation_function_output_layer(z)
        return output
    
    def backward(self, X: np.ndarray, outputs: np.ndarray, targets: np.ndarray):
        self.gradients = []                     # Delete old gradient
        train_size = X.shape[0]

        delta = -(targets - outputs)
        gradient = self.activations[-1].T @ delta / train_size
        self.gradients.insert(0,gradient)       # The same as append, just on the other side of the list

        for i in range(1,len(self.neurons_pr_layer)+1):
            activation_derivative = self.activation_function_hidden_layers_der(self.zs[-i])
            delta = (delta @ self.ws[-i].T) * activation_derivative
            gradient = self.activations[-1-i].T @ delta / train_size
            self.gradients.insert(0,gradient)   # The same as append, just on the other side of the list

class Trainer:
    def __init__(self,
                 model: Model,
                 learning_rate: float,
                 epochs: int,
                 X_train: np.ndarray,
                 Y_train: np.ndarray) -> None:
        
        self.model = model
        self.X = X_train
        self.Y = Y_train

        self.learning_rate = learning_rate
        
        self.accuracy_history = []

        self.look_back = 30
        
        self.epochs = epochs
        self.milestones = [math.floor(self.epochs/100)*i for i in range(1, 101)]
        
    def train(self):

        improving = True
        best_mse = math.inf
        epoch = 0

        while improving and epoch <= self.epochs:
            # Forward and Backward pass
            outputs = self.model.forward(self.X)
            targets = self.Y
            self.model.backward(self.X, outputs, targets) # Calculates gradient

            # Gradient decent step
            for i in range(len(self.model.gradients)):
                gradient = self.model.gradients[i]
                self.model.ws[i] = self.model.ws[i] - (self.learning_rate*gradient)

            # Calculate accuracy
            mse = accuracy(self.model, self.X, self.Y)
            self.accuracy_history.append(mse)
            best_mse = np.min([mse, best_mse])

            # Determin whether or not to stop training
            if epoch < self.look_back:
                improving = True   
            elif (mse == best_mse) or (best_mse in self.accuracy_history[-self.look_back: -1]):
                improving = True
            else:
                print("Early stopping since training accuracy did not improve")
                improving = False

            # Print Training Progress
            if epoch in self.milestones:
                print_progress(math.floor(epoch/epochs*100))
                if epoch != self.epochs:
                    print("")
                    print(f"Current mean square error of the network (training data): {mse}")

            # Increacing global step by one
            epoch += 1    


def accuracy(model: Model, X: np.ndarray, output: np.ndarray):
    predicted = model.forward(X)
    MSE = ((output - predicted)**2).mean()
    return MSE

def print_progress(percentage: int):  
    clear_console()
    print("Training in progress")
    len_chop = 2
    progress = "["
    for i in range(math.floor(100/len_chop)):
        if i < percentage/len_chop:
            progress += "="
        else:
            progress += " "
    progress += "] "+str(percentage)+"%"
    print(progress)

def clear_console():
    print("\033[H\033[J", end="")


if __name__ == "__main__":
    np.random.seed(0)
    X_train, y_train, X_test, y_test = get_data(n_train=280, n_test=120)
    y_train = y_train.reshape(y_train.shape[0], 1)
    y_test  = y_test.reshape(y_test.shape[0], 1)
            
    layers = [X_train.shape[1], 2, y_train.shape[1]] # Network topology
    model  = Model(layers=layers)

    learning_rate = 1e-2 # (Report)
    epochs = 1e+5
    trainer = Trainer(model=model,
                      learning_rate=learning_rate,
                      epochs=epochs,
                      X_train=X_train,
                      Y_train=y_train)
    trainer.train()
    
    mse_train = accuracy(model=model, X=X_train, output=y_train)
    mse_test  = accuracy(model=model, X=X_test,  output=y_test)

    print("")
    #print(f"Training history: {trainer.accuracy_history}")
    print(f"Training MSE {mse_train}")
    print(f"Test     MSE {mse_test}")