import pickle
from typing import Dict, List, Any, Union
import numpy as np
# Keras
import tensorflow as tf
from tensorflow import keras
from keras.utils import pad_sequences


def load_data() -> Dict[str, Union[List[Any], int]]:
    path = "keras-data.pickle"
    with open(file=path, mode="rb") as file:
        data = pickle.load(file)

    return data


def preprocess_data(data: Dict[str, Union[List[Any], int]]) -> Dict[str, Union[List[Any], np.ndarray, int]]:
    """
    Preprocesses the data dictionary. Both the training-data and the test-data must be padded
    to the same length; play around with the maxlen parameter to trade off speed and accuracy.
    """
    maxlen = data["max_length"]//16
    data["x_train"] = pad_sequences(data['x_train'], maxlen=maxlen)
    data["y_train"] = np.asarray(data['y_train'])
    data["x_test"] = pad_sequences(data['x_test'], maxlen=maxlen)
    data["y_test"] = np.asarray(data['y_test'])

    return data


def train_model(data: Dict[str, Union[List[Any], np.ndarray, int]], model_type="feedforward") -> float:
    """
    Build a neural network of type model_type and train the model on the data.
    Evaluate the accuracy of the model on test data.

    :param data: The dataset dictionary to train neural network on
    :param model_type: The model to be trained, either "feedforward" for feedforward network
                        or "recurrent" for recurrent network
    :return: The accuracy of the model on test data
    """

    # TODO build the model given model_type, train it on (data["x_train"], data["y_train"])
    #  and evaluate its accuracy on (data["x_test"], data["y_test"]). Return the accuracy

    # Define Size & Length of Input, Output
    input_dim = data["vocab_size"]
    output_dim = data["x_test"].shape[1]
    input_length = data["max_length"]

    # Create a feedforward model
    if model_type == "feedforward":
        # Initialize the model
        model = tf.keras.Sequential()

        # Define Input-Layer & Output-Layer
        model.add(tf.keras.layers.Embedding(
            input_dim, output_dim, input_length=input_length//16))

        # Create shaping-layers with dense and flatten. I chose Re-Lu for the hidden layer and Sigmoid for the output layer
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, "relu"))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(1, "sigmoid"))

        # "Save" the model
        model.compile(optimizer = tf.keras.optimizers.Adadelta(learning_rate=0.1), loss = "binary_crossentropy", metrics = ["accuracy"])

    # Create a recurrent model
    if model_type == "recurrent":
        # Initializes the model
        model = tf.keras.Sequential()

        # Define Input-Layer & Output-Layer
        model.add(tf.keras.layers.Embedding(
            input_dim, output_dim, input_length=input_length//16))

        # Create shaping-layers with LSTM and flatten. Re-LU is used for the hidden layer while Sigmoid is used for the output layer
        model.add(tf.keras.layers.LSTM(128, "relu"))

        # Regularization technique
        model.add(tf.keras.layers.Dropout(0.2))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(1, "sigmoid"))

        # "Save" the model
        model.compile(optimizer = tf.keras.optimizers.Adadelta(learning_rate=0.1), loss = "binary_crossentropy", metrics = ["accuracy"])

    # Training data definition
    training_data = (data["x_train"], data["y_train"])
    testing_data = (data["x_test"], data["y_test"])

    # How many forwards/backwards of all training data
    epochs = 1

    # Do I want to measure the progress of the training? -> Yes = 1, No = 0
    verbose = 1

    # Train the model
    model.fit(*training_data, 10000, epochs,
              verbose)

    loss, accuracy = model.evaluate(*testing_data, verbose)

    return accuracy


def main() -> None:
    print("1. Loading data...")
    keras_data = load_data()
    print("2. Preprocessing data...")
    keras_data = preprocess_data(keras_data)
    print("3. Training feedforward neural network...")
    fnn_test_accuracy = train_model(keras_data, model_type="feedforward")
    print('Model: Feedforward NN.\n'
          f'Test accuracy: {fnn_test_accuracy:.3f}')
    print("4. Training recurrent neural network...")
    rnn_test_accuracy = train_model(keras_data, model_type="recurrent")
    print('Model: Recurrent NN.\n'
          f'Test accuracy: {rnn_test_accuracy:.3f}')


if __name__ == '__main__':
    main()
