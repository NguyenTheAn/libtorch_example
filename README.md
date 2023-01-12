# Libtorch Example with Custom Dataset <br>
This project is an example about how to use libtorch to train a classification model. It include 2 model: <br>
- Handwritten digit recognition
- Dog and Cat classification


The custom dataset is in the [dataset](./dataset/) folder. The [Mnist](./dataset/mnist.hpp) dataset loads information from a csv filee and the [Dog and Cat](./dataset/dogncat.hpp) dataset loads image from a folder. [model](./model) folder contain Fully Connected Neural Network ([FCN](./model/FCN.hpp)) and Convolutional Neural Network ([CNN](./model/CNN.hpp))