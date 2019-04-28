#!/usr/bin/env python

from keras.models import load_model
from keras.models import model_from_json
from fc import MultiLayerPerceptron

def loadFromKerasModel(model):
    layers = []
    if  len(model.input_shape) == 2:
        allDense = 1
        layers.append(['Dense',model.input_shape[1:]])
    else:
        allDense = 0
        layers.append(['Conv2D',model.input_shape[1:]])

    for layer in model.layers:
        layerClass = layer.__class__.__name__
        if layerClass == 'Dense' or layerClass == 'Conv2D' or layerClass == 'Flatten' or layerClass == 'MaxPooling2D':
            if layerClass == 'MaxPooling2D':
                layers.append(['Conv2D', layer.output_shape[1:]])
            elif layerClass == 'Flatten':
                layers.append(['Dense', layer.output_shape[1:]])
            else:
                layers.append([layerClass, layer.output_shape[1:]])
            
            if layerClass != 'Dense':
                allDense = 0
    return layers, allDense

def kerasToVnn(model):
    layers, allDense = loadFromKerasModel(model)
    if allDense:
        units = []
        for layer in layers:
            units.append(layer[1][0])
        vnnModel = MultiLayerPerceptron(layer_sizes=units,showgrid=True)
    else:
        layers_conv = []
        layers_dense = []
        for layer in layers:
            print(layer)
            if layer[0] == 'Conv2D':
                layers_conv.append(layer[1])
            else:
                layers_dense.append(layer[1][0])
        print(layers_dense,layers_conv)
        vnnModel = ConvNet2D(layers_conv, layers_dense)
    return vnnModel

def loadFromFile(filepath):
    model = load_model(filepath)
    return kerasToVnn(model)

def loadFromJSON(filepath):
    json_file = open(filepath, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    return kerasToVnn(loaded_model)