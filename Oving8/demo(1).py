"""
Train a simple convolutional neural network on the MNIST dataset.
The model gets > 99% test accuracy after some epochs, but there is still a lot of margin for parameter tuning.
"""

"""
To run on Mac Silicone and utilize GPU (M1 or better), you need to pip install the following:
- tensorflow-macos: This will install the CPU version of TF for macos; I have tested on v. 2.9.0 
- tensorflow-metal: This will install the GPU addons; I have tested on v. 0.5.0
"""

import numpy as np
import tensorflow as tf
from typing import Tuple


def prepare_data() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Prepare the data: It will end up as shuffled and split between train and test sets.
    Data is normalized to (0, 1) and coded as floats.
    Shape of each image in the data is (28, 28, 1), since we only have one color channel.
    Labels are one-hot encoded.

    :return: Two tuples: First with training data (x, y), second with test (x, y)
    """

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()          # Load the data from Keras
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255.     # Reshape and normalize
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255.

    """
    Convert class vectors to binary class matrices
    This is the so-called 'one-hot' encoding of the labels.
    It is a common way to represent categorical data in ML.
    So, if the original label was 3, it will now be [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    Note that the length of the vector = number of classes, and that only one element is 1.
    """
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    """
    Going home
    """
    return (x_train, y_train), (x_test, y_test)


def define_model() -> tf.keras.models.Sequential:
    """
    Define the model  -- Here we go with a very simple thing, e.g., no dropouts.
    Still it gets 99% accuracy when run for sufficient number of epochs (say 50).

    Defaulting to 32 filters, kernel size (3, 3) without much thinking.
    Max-pooling over (2,2) also without much thinking behind it.

    :return: A Keras sequential-model that already has been compiled and is ready for training
    """

    # Use tf.keras and the Sequential class to define the model layer-by-layer.
    # This hides all the dirty details from us, yet gives the flexibility needed to make an OK model.
    model = tf.keras.models.Sequential()

    """
    First layer will just declare the input size. 
    """
    model.add(  # Add a layer to the sequential model
        tf.keras.layers.InputLayer(  # This layer just defines the input size. It is not a 'real' layer, and not needed.
            input_shape=(28, 28, 1),  # Input is an image of size (28, 28, 1); 1 is because we have one color channel.
            name='Input',  # Name for easy reference
        )
    )

    """
    First conv-layer: Input to this layer is the previous layer:
    An  image of size (28, 28, 1) where the 1 is because we have one color channel. 
    Result is (28, 28, 32): 28 * 28 due to padding and no stride, 32 is the number of filters I have.
    
    We can think of this as giving the model 32 ways to represent interesting local patterns. 
    One channel may of course simply copy the raw data (who knows), but there is also sufficient space in 
    terms of the number of filters for e.g. line detection or smoothing (or both).
    
    Number of learnable parameters per filter are 3 * 3 (layer-size) weights + 1 bias = 10 params/filter,
    thus 320 parameters in total. 
    """
    model.add(                          # Add a layer to the sequential model
        tf.keras.layers.Conv2D(         # The layer is a convolutional 2D layer
            32,                         # Number of filters
            (3, 3),                     # 'window size', i.e., the size of the kernel's receptive field
            input_shape=(28, 28, 1),    # For the first layer we need to tell the system what shape the input data is
            activation='relu',          # Activation function
            padding='same',             # Padding to determine how to behave around the edges.
            name='Conv2d-1',            # Name for easy reference
        )
    )

    """
    Second conv-layer: Output shape is same as input (28, 28, 32) due to padding & stride + same no. filters.
    The point here is that we gradually let each value in the representation consider larger parts of the image: 
    In the first conv-layer, a value at (i, j, filter) would look at the area of the image inside the rectangle 
    defined as between the points (i-1, j-1) and (i+1, j+1). Now we get a 'view' in the area defined from 
    (i-2, j-2) to (i+2, j+2).
    Notice the use of kernel_regularizer here. We simply add L2 regularization to the weights of the layer.      
    """
    model.add(
        tf.keras.layers.Conv2D(
            32, (3, 3), activation='relu', padding='same', name='Conv2d-2',
            kernel_regularizer=tf.keras.regularizers.l2(),    # Regularize the weights using std L2
        )
    )

    """
    Third conv-layer: Output shape is same as input: (28, 28, 32), again due to padding & stride + same no. filters
    The number of weights is same as in the second conv-layer, since shapes are the same.
    """
    model.add(
        tf.keras.layers.Conv2D(
            32, (3, 3), activation='relu', padding='same', name='Conv2d-3',
            kernel_regularizer=tf.keras.regularizers.l2()
        )
    )

    """
    First max-pooling: Going from (28, 28, 32) and down to (14, 14, 32) because max-pooling over (2, 2).
    The point of the max-pooling we do here is to strengthen the 'localized' information in each filter. 
    So, if an image is shifted one pixel to the left (or right or up or down), the max-pooling will recognize that and 
    ensure that the same activations are sent on forward. 
    Max-pooling does not have trainable parameters.  
    """
    model.add(
        tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2), name='Pool2d-1',
        )
    )

    """
    Fourth conv-layer: 
    Output shape is same as input, (14, 14, 32), due to padding & stride + same no. filters.
    """
    model.add(
        tf.keras.layers.Conv2D(
            32, (3, 3), activation='relu', padding='same', name='Conv2d-4',
            kernel_regularizer=tf.keras.regularizers.l2()
        )
    )

    """
    Second max-pooling. 
    Again reduction by a factor of 2 in first two dims, leaving us with (7, 7, 32).
    """
    model.add(
        tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2), name='Pool2d-2',
        )
    )

    """
    Flatten takes the representation of (7, 7, 32) into a vector. Size will be 7 * 7 * 32 =  1568 per observation.
    This is just a 'reshape' of the representation from (*, 7, 7, 32) to (*, 1568), not a layer with learnable 
    parameters. We do it because we want to end up with something that can be fed into a dense layer.
    """
    model.add(
        tf.keras.layers.Flatten(name='Flatten')
    )

    """
    Simple dense layer taking us down to a representation of size 128. 
    Think of this as giving the model a way to compress its representation. It used to be 1568 real values, 
    now it is going to be only 128. The compression is by a weighted sum + rectified linear unit (relu) activation.  
    This will require a *lot* of parameters, namely 1568 * 128 weights + 128 biases =  200.832 parameters. 
    Over-fitting can be easily handled using some of the standard regularization techniques. 
    Surprisingly, it also works well without that for the simple data we use in this demo, so I go with that. 
    """
    model.add(
        tf.keras.layers.Dense(
            128, activation='relu', name='Compress',
        )
    )

    """
    Finally we are going to have 10 outputs, one per class. Hopefully the representation of size 128 in the previous 
    layer suffices to give info about what class the image should be allocated to. Here we make a weighted sum of 
    those 128 real-values. Note the softmax activation function. This ensures that the output over this layer sums to 1, 
    so we can think of it as if a high value for output <c> means that class <c> is considered 'likely'. 
    Number of parameters here will be 128 * 10 weights + 10 biases = 1290 parameters.  
    """
    model.add(
        tf.keras.layers.Dense(
            10,                         # Number of outputs equals number of classes
            activation='softmax',       # Softmax activation function ensures a sum of 1 over the outputs
            name='Output'               # Name for easy reference
        )
    )

    """
    Compile the model. Define the loss as categorical crossentropy and define the optimizer + metrics to survey
    The categorical cross entropy loss is useful for the classification we are doing here.
    If I was doing this 'for real', I'd have spent time on finding good optimizer and a learning-rate to go with it. 
    Here I simply choose the optimizer somewhat arbitrarily, and rely on default params for that one.
    """
    model.compile(
        loss='categorical_crossentropy',    # Loss function we want to optimize. Standard choice for classification
        optimizer='adadelta',               # Optimizer to use. I chose this one somewhat arbitrarily
        metrics=['accuracy'],               # Metrics to survey during training. Here we are interested in accuracy
    )

    """
    If we have files saved with weights then let's try to load them. This makes it possible to have a 'hot-start' 
    for training. Not a required feature, but nice for demos so that I don't have to run more than a few epochs to 
    get OK results.
    """
    try:
        model.load_weights('./weights.h5')
        print('Weights loaded successfully')
    except IOError:
        print('Weight loading did not work')

    """
    Dump description using model.summary(). This is always very useful to check that the model you have actually 
    built is indeed the one you *tried* to build. There is also interesting info here, like the number of trainable 
    parameters.
    """
    model.summary()
    return model


def learn_model_from_data(no_epochs: int = 50) -> tf.keras.models.Sequential:
    """
    This is basically all that needs doing to define and learn the model from data:
    1) Define model structure
    2) Get hold of MNIST data. Here I use simple numpy arrays. Much more efficient representations are possible, but
        makes no difference for me with the simple data we use in this demo.
    3) Do the training for a given number of epochs.

    :param: no_epochs: The number of training epochs.
    :return: The learned model in the form of a Keras Sequential.
    """

    """
    Define model structure
    """
    model = define_model()

    """
    Get hold of data: Two tuples, one for training and one for test data. Each holding two numpy arrays: (x, y).
    """
    train_data, test_data = prepare_data()

    """ 
    Fit model based on training data
    tf.keras abstracts all the tricky parts away: Our model simply has a .fit method 'sklearn-style'. 
    We only need to supply 
    -- training data: Since it is a tuple in my code I unwrap it when passing it on.
    -- batch-size: I have chosen it to use my GPU efficiently, no tweaking or hyper-parameter search.
    -- number of epochs: Typically I'd use something large(r) combined with early stopping. 
        For demo I typically choose something small. One epoch takes ~ 2 sec. on GPU
    -- validation_data: The test-set that is supplied to calculate the generalization-error. 
    """
    model.fit(
        *train_data,                # Unpack the tuple. Now the two first submitted parameters are numpy arrays x and y.
        batch_size=16384,           # This is a good batch-size for my GPU.
        epochs=no_epochs,           # I would typically use something large if trained from scratch, but for demo I use
                                    # something small and rely on loading pretrained weights from file.
        verbose=1,                  # I like to see progress; verbose == 0 would keep it quiet.
        validation_data=test_data   # This is the data-set that is used to calculate the generalization-error while
                                    # the model trains. Take care here: If I were to use this for anything more than
                                    # just showing the value (e.g., early stopping, hyperparameter search) I'd have
                                    # to use a SEPARATE validation-set for that. The test-set should NEVER be used
                                    # for anything other than the final evaluation of the model.
    )

    """
    Check quality and report to screen. 
    Since I know this works well I don't care about thorough examination of results here.
    In a more real situation I'd received the history-object returned from model.fit and analyzed it more carefully.  
    """
    loss, accuracy = model.evaluate(*test_data, verbose=0)
    print(f'\nTest loss: {loss:.6f}')
    print(f'Test accuracy: {accuracy * 100:.2f}%')

    """
    Save the learned weights and go home
    """
    model.save_weights('./weights.h5')
    return model


if __name__ == '__main__':
    model_ = learn_model_from_data(no_epochs=5)
