# VGG-19 with dropout Training
  This code is used to train and create a VGG19 with dropout model which is later saved to use in facial expression recognition.
  
# Getting Started
You need to collect all the requirements and notedown what are needed.

# Prerequisites
Some of the prerequisites might include:
Downloading RAFDB and FER2013 datasets with labels
Python 3.6 or higher
OpenCV
Matplotlib
Scikit-image
MTCNN
TensorFlow and Keras

# Installing
Install the required packages 
pip install opencv-python matplotlib scikit-image mtcnn pillow tensorflow==2.4.0 tkinter
This will install OpenCV, Matplotlib, pytorch, Scikit-image, MTCNN, Pillow, TensorFlow version 2.4.0, and Tkinter packages.

# Usage
Install the required libraries and packages mentioned above.
Install the necessary packages, including TensorFlow and Keras.
Prepare your dataset.Then create a directory containing two subdirectories: one for training data and one for validation data. Each subdirectory should contain subdirectories for each expression class, with each image of a particular class stored in its respective class subdirectory.
Update the paths to the training and validation data directories in the code.
Define the expression classes in the classes variable.
Set the img_height and img_width variables to the dimensions of your input images.
Run the code.
When you run the code, it will load and preprocess the training and validation data using the ImageDataGenerator class. It will then define the VGG19-based CNN, compile it, and train it on the data. After training, it will save the trained model to a file called my_vgg19_dropout_model.h5

# Explination
Sure, here's an explanation of the code you provided:

1. The first few lines import the necessary packages for building and training a convolutional neural network for image classification using the Keras library.

2. The next few lines define the input shape of the images to be fed into the network, as well as the directories where the training and validation data are stored.

3. The `train_datagen` and `val_datagen` objects use the `ImageDataGenerator` class to apply data augmentation to the training data (randomly shearing, zooming, and flipping the images) and to rescale the pixel values of both the training and validation data by dividing them by 255 to make them fall within the range of [0, 1].

4. The `train_data` and `val_data` objects are created by calling the `flow_from_directory` method on the `train_datagen` and `val_datagen` objects, respectively. These objects will read the images from the specified directories, apply the preprocessing steps defined by the data generators, and generate batches of image data to be fed into the neural network during training.

5. The `classes` list contains the names of the emotion classes that the network will be trained to classify.

6. The `vgg19` model is created by calling the `VGG19` function from the `tensorflow.keras.applications.vgg19` module, passing in the `include_top=False` argument to exclude the fully connected layer at the top of the network, and the `input_shape` argument to specify the shape of the input images.

7. The `for` loop sets all the layers of the `vgg19` model to be non-trainable, meaning that their weights will not be updated during training.

8. The `model` object is created by calling the `Sequential` class, which allows you to stack layers on top of each other to form a neural network. The `vgg19` model is added as the first layer, followed by a `Flatten` layer to convert the output of the convolutional layers into a 1D vector, a `Dense` layer with 256 units and ReLU activation, a `Dropout` layer to randomly drop out 50% of the units during training to prevent overfitting, and a final `Dense` layer with a softmax activation function to output the probabilities of the input image belonging to each of the emotion classes.

9. The `compile` method is called on the `model` object to configure the training process, specifying the loss function to be categorical cross-entropy (since this is a multi-class classification problem), the optimizer to be Adam with a learning rate of 0.0001, and the metric to be accuracy.

10. The `fit` method is called on the `model` object to train the network on the training data, specifying the number of epochs to be 8 and the validation data to be used to evaluate the performance of the network on data it has not seen before.

11. Finally, the `save` method is called on the `model` object to save the trained network as an HDF5 file.

# Known Issues
As far as the code provided is concerned, there seem to be no known issues. However, it is always possible that issues arise when the code is run on different systems or with different input data. It is recommended to test the code thoroughly and report any issues that may arise.

# License 
This project license belongs to me , but it can be used for educational uses.


