# Fashion-MNIST Image Classification with Convolutional Neural Networks

This project demonstrates image classification using Convolutional Neural Networks (CNNs) on the Fashion-MNIST dataset. It trains a CNN model to classify images of fashion items into different categories.

## Dataset

The Fashion-MNIST dataset is a collection of 60,000 28x28 grayscale images of 10 different fashion item categories, with 6,000 images per category. The dataset is divided into 50,000 training images and 10,000 test images. Each image is labeled with one of the following categories: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, and Ankle boot.

The dataset is preprocessed by reshaping the images to (28, 28, 1) and scaling the pixel values between 0 and 1 (float32) to normalize the data before training the model.

## Requirements

To run the project, you need to have the following dependencies installed:

- Python (version 3.6 or later)
- NumPy (```pip install numpy```)
- TensorFlow (```pip install tensorflow```)
- Matplotlib (```pip install matplotlib```)

## Getting Started

To get started with the project, follow these steps:

1. Clone the project repository from GitHub.
2. Install the required dependencies as mentioned in the "Requirements" section.
3. Open a terminal or command prompt and navigate to the project directory.
4. Run the script `fashion_mnist_classification.py` using the command: `python fashion_mnist_classification.py`.
5. The script will load the Fashion-MNIST dataset, preprocess the data, define the CNN model architecture, compile the model, and train the model.
6. During training, the accuracy and loss curves will be plotted using Matplotlib.
7. After training, the model will be evaluated on the test set, and the test accuracy will be displayed in the terminal.

## Model Architecture

The project uses a CNN model architecture consisting of the following layers:

1. **Conv2D**: 32 filters of size 3x3 with ReLU activation function and input shape (28, 28, 1).
2. **MaxPooling2D**: Max pooling with pool size 2x2.
3. **Flatten**: Flatten the output from the previous layer.
4. **Dense**: Fully connected layer with 64 units and ReLU activation function.
5. **Dense**: Fully connected layer with 10 units (output layer).

The model is compiled with the Adam optimizer, sparse categorical cross-entropy loss function, and accuracy as the evaluation metric.

## Results

The project plots the accuracy and loss curves during training using Matplotlib. The curves show the training and validation accuracy/loss over epochs, allowing you to analyze the model's performance.

After training, the model is evaluated on the test set, and the test accuracy is displayed in the terminal.

## License

This project is licensed under the [MIT License](LICENSE).

Feel free to modify and adapt the code according to your needs.

If you have any questions or suggestions, please feel free to contact me.

**Author:** Sajad Waloo
**Email:** sajadwaloo786@gmail.com

---
