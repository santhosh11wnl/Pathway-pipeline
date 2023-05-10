# Facial Expression Recognition
This is a Python application that uses deep learning to classify facial expressions in images. The application is built using the VGG19 deep learning model with dropout regularization, which was trained on a dataset of facial expressions consisting of seven classes: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.The application uses the MTCNN library for face detection, and the skimage and OpenCV libraries for image processing. It also includes a graphical user interface (GUI) built with the tkinter library, allowing users to browse for an image and get a classification of the facial expression.To use the application, simply run the script and click the "Browse for an image" button. Select an image file and the application will display the image with the predicted facial expression label. If the image size is too small, an error message will be displayed.

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
Tkinter (for the GUI)

# Installing
Install the required packages 
pip install opencv-python matplotlib scikit-image mtcnn pillow tensorflow==2.4.0 tkinter
This will install OpenCV, Matplotlib, Scikit-image, MTCNN, Pillow, TensorFlow version 2.4.0, and Tkinter packages.

# Usage
Install the required libraries and packages mentioned above.
Download the my_vgg19_dropout_model.h5 file and place it in the project directory.
You need to split dataset for into 3 directories of training,testing and validating.
Place the images that you want to classify in the "testing" directory.
Run the main.py file.
A tkinter window will appear that will allow you to browse and select an image to classify.
Once you have selected an image, the predicted emotion will be displayed along with the image.

# Known Issues
As far as the code provided is concerned, there seem to be no known issues. However, it is always possible that issues arise when the code is run on different systems or with different input data. It is recommended to test the code thoroughly and report any issues that may arise.

# License 
This project license belongs to me , but it can be used for educational uses.


