# VisualNN

VisualNN is an interface for visualising written in python using plotly. It is flexible and can be used for visualising various neural network or deep learning architectures like multi-layered perceptron and convolutional neural networks. The library also allows capability to visualise a Keras model, saved model or saved architecture directly. This paper describes the implementation of the VisualNN library and the way it works. The project was a part of the Software Engineering course at BTIS Pilani K. K. Birla Goa Campus.


Installing the library:

`sudo python3 setup.py install`

*Visualising MultiLayer Perceptron:*

``` 
from fc import MultiLayerPerceptron
model = MultiLayerPerceptron(layer_sizes=layerList,showgrid=True)
model.plot()
```
layerList -> list of sizes of the layers of MLP

*Visualising Convolutional Model:*

```
from convnet import ConvNet2D
model = ConvNet2D(layers_conv, layers_dense)
model.plot()
```
layers_conv -> list of dimensions of the convolutional layers

layers_dense -> list of size of fully connected layers

*Loading from keras model:*

```
import keras_loader
vnnmodel = keras_loader.kerasToVnn(model)
vnnmodel.plot()
```
model -> keras model, works for MLP and convolutional models

*Loading from saved keras model and saved keras architecture:*
```
import keras_loader
vnnmodel1 = keras_loader.loadFromFile(filepath)
vnnmodel1.plot()

vnnmodel2 = keras_loader.loadFromJSON(filepath)
vnnmodel2.plot()
```
