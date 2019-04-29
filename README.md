# VisualNN

VisualNN is an interface for visualising written in python using plotly. It is flexible and can be used for visualising various neural network or deep learning architectures like multi-layered perceptron and convolutional neural networks. The library also allows capability to visualise a Keras model, saved model or saved architecture directly. This paper describes the implementation of the VisualNN library and the way it works. The project was a part of the Software Engineering course at BTIS Pilani K. K. Birla Goa Campus.


**Installation**

You can clone this repository on your system, go to the home directory and run the following command in terminal:
```bash
sudo python3 setup.py install
````

You can run some examples in the following way:

Multi-layer Perceptron:
```bash
$ python3 fceg.py
```
Convolutional Neural Network:
```bash
$ python3 convnet_example.py
```

-----
### Using the library

You can create your own visualizations in the following ways:

**Visualising a Multi-layer Perceptron**

```python
from fc import MultiLayerPerceptron

model = MultiLayerPerceptron(layer_sizes=layerList,showgrid=True)
model.plot()
```
layerList: list of sizes of the layers of MLP (see example).

**Visualising a Convolutional Neural Network**

```python
from convnet import ConvNet2D

model = ConvNet2D(layers_conv, layers_dense)
model.plot()
```
layers_conv: list of dimensions of the convolutional layers (see example) \
layers_dense: list of size of fully connected layers (see example)

It is also possible to load the model directly from Keras in the following ways:

**Loading from Keras Sequential Model**
```python
import keras_loader

# model is a Keras object
# model = Sequential()
vnnmodel = keras_loader.kerasToVnn(model)
vnnmodel.plot()
```

**Loading from saved Keras model and saved Keras architecture**
```python
import keras_loader

# From .h5 file
vnnmodel1 = keras_loader.loadFromFile(filepath)
vnnmodel1.plot()

# From .json file
vnnmodel2 = keras_loader.loadFromJSON(filepath)
vnnmodel2.plot()
```
