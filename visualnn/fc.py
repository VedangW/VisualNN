#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.models import model_from_json
# from keras_loader import *
from convnet import ConvNet2D
from plotly.offline import init_notebook_mode, plot

def get_line(x, p1, p2):
    """ Function to get a line
        y = mx + c, given two points
        that the line passes through.
        
        Parameters
        ----------
        x: np.ndarray
            Corresponding points on x-axis
        p1: tuple
            First point, pair of (x1, y1)
        p2: tuple
            Second point, pair of (x2, y2)
    """
    x1, y1 = p1
    x2, y2 = p2
    
    # Handle division by zero error
    if x2 == x1:
        return np.linspace(y1, y2, len(x))
    
    m = (y2-y1)/(x2-x1)
    c = y1 - m*x1
    
    return m*x + c

class Dense:
    
    def __init__(self, num_neurons, x_coord=1, 
                 offset=2, x_offset=1, n_color='blue',
                 showgrid=False):
        """ Class to represent a layer of the MLP class.
            
            Parameters
            ----------
            num_neurons: int
                No. of neurons in layer.
            x_coord: int or float
                x-coordinate of the layer.
            offset: int or float
                Distance between two neurons of the layer.
            x_offset: int or float
                Distance between x_axis and the lowest
                neuron in the layer.
            n_color: string (Plotly color)
                Color of the neurons. 
        """
        
        self.offset = offset
        self.x_offset = x_offset
        self.x_coord = x_coord
        self.num_neurons = num_neurons
        self.n_color = n_color
        self.showgrid = showgrid
        
        self.data = list()
        self.x_list = list()
        self.y_list = list()
        
        # Highest value of y for any neuron + offset
        self.max_y = self.x_offset + self.num_neurons*self.offset
        
        # Set coordinates
        self._set_coords()
        # Set data
        self._set_data()
        
    # Getter and setter methods
    
    @property
    def x_list(self):
        return self.__x_list
    
    @x_list.setter
    def x_list(self, new):
        self.__x_list = new
        
    @property
    def y_list(self):
        return self.__y_list
    
    @y_list.setter
    def y_list(self, new):
        self.__y_list = new
        
    @property
    def data(self):
        return self.__data
    
    @data.setter
    def data(self, new):
        self.__data = new
        
    @property
    def x_offset(self):
        return self.__x_offset
    
    @x_offset.setter
    def x_offset(self, new):
        self.__x_offset = new
    
    @property
    def offset(self):
        return self.__offset
    
    @offset.setter
    def offset(self, new):
        self.__offset = new
    
    @property
    def x_coord(self):
        return self.__x_coord
    
    @x_coord.setter
    def x_coord(self, new):
        self.__x_coord = new
        
    # Methods
        
    def _set_layout(self):
        """ Sets the layout if the layer needs
            to be plotted. """
        
        # Set range for x
        self.x_min = self.x_coord - 1
        self.x_max = self.x_coord + 1
        
        # Set range for y
        self.y_min = 0
        self.y_max = self.max_y
        
        # Set the layout
        self.layout = {'xaxis': {'range': [self.x_min, self.x_max], 
                                 'autorange': False, 
                                 'zeroline': False,
                                 'showgrid': self.showgrid,
                                 'showticklabels': self.showgrid},
                       'yaxis': {'range': [self.y_min, self.y_max], 
                                 'autorange': False, 
                                 'zeroline': False,
                                 'showgrid': self.showgrid,
                                 'showticklabels': self.showgrid},
                       'title': 'MLP Layer',
                       'hovermode': 'closest'}

    def _set_coords(self):
        """ Sets the x and y coordinates of the
            neurons in the layer. Note that this
            is set even if the object doesn't need
            to be plotted. """
        
        self.coords = list()
        
        initial = self.x_offset
        for i in range(self.num_neurons):
            self.x_list.append(self.x_coord)
            self.y_list.append(initial + i*self.offset)
        
    def _set_data(self):
        """ Sets the data attribute of the class.
            Note that this is set even if the object
            doesn't need to be plotted. """
    
        # Set data for the neurons
        self.data += [dict(x=self.x_list, y=self.y_list,
                    mode='markers',
                    marker=dict(color=self.n_color, size=30))]
        self.data += [dict(x=self.x_list, y=self.y_list,
                    mode='markers',
                    marker=dict(color='white', size=20))]
        
    def plot(self):
        """ For plotting the MLP layer. Mostly used
            for testing or such. """
        
        # Set the layout
        self._set_layout()
        
        # Create figure and plot
        self.figure = dict(data=self.data, layout=self.layout)
        plot(self.figure)

def get_standard_layout(x_min, x_max, y_min, y_max, name, 
                        showgrid):
    """ Get the standard layout for any plot. 
        
        Parameters
        ----------
        x_min: int or float
            Lower bound of x range.
        x_max: int or float
            Upper bound of x range.
        y_min: int or float
            Lower bound of y range
        y_max: int or float
            Upper bound of y range.
        name: string
            Name of plot
        showgrid: bool
            To show the grid or not.
    """ 
    standard_layout = {'xaxis': {'range': [x_min, x_max], 
                                 'autorange': False, 
                                 'zeroline': False,
                                 'showgrid': showgrid,
                                 'showticklabels': showgrid},
                       'yaxis': {'range': [y_min, y_max], 
                                 'autorange': False, 
                                 'zeroline': False,
                                 'showgrid': showgrid,
                                 'showticklabels': showgrid},
                       'title': name,
                       'hovermode': 'closest',
                       'showlegend': False}
    
    return standard_layout

def get_trace(x, y, color='blue'):
    """ Get trace for a line. 
        
        Parameters
        ----------
        x: list
            List of x-coords
        y: list
            List of y-coords
        color: string (Plotly color)
            Color of trace. 
    """
    trace = [dict(x=x, y=y,
                   mode='lines',
                   line=dict(width=2, color=color)
                  )
            ]
    
    return trace


def connect(layer_curr, layer_next):
    """ Connect two layers in the neural network.
    
        Parameters
        ----------
        layer_curr: MultiLayerPerceptronLayer
            Current layer
        layer_next: MultiLayerPerceptronLayer
            Next layer
            
        Returns
        -------
        data: list of dicts
            The data list.
    """
    data = list()
    
    # Points in current and next layer in tuple form
    curr_points = [(layer_curr.x_list[i], layer_curr.y_list[i]) 
                   for i in range(len(layer_curr.x_list))]
    next_points = [(layer_next.x_list[i], layer_next.y_list[i]) 
                   for i in range(len(layer_next.x_list))]
    
    # Set of points on the x-axis
    x = np.linspace(layer_curr.x_coord, layer_next.x_coord, 2)
    
    # Get y-coordinates of x for all lines
    y_s = list()
    for i in curr_points:
        for j in next_points:
            y_s.append(get_line(x, i, j))
     
    # Add traces to data
    for i in range(len(y_s)):
        data += get_trace(x, y_s[i])
    
    data += layer_curr.data
    data += layer_next.data
    
    return data
    
def connect_layers(layers, name, showgrid):
    """ Connect all layers in the MLP. """
    data_s = [layer.data for layer in layers]
    
    # Range of x in layout
    x_min = layers[0].x_coord - 1
    x_max = layers[-1].x_coord + 1
    
    # Range of y in layout
    y_min = 0
    y_max = max([layer.max_y for layer in layers])
    
    # Get standard layout
    layout = get_standard_layout(x_min, x_max, y_min, y_max, 
                                 name, showgrid)

    # Get the connection traces
    data = list()
    for i in range(len(layers)-1):
        data += connect(layers[i], layers[i+1])
    
    return data, layout

class MultiLayerPerceptron:
    
    def __init__(self, layer_sizes, n_color='blue', b_color='red', 
                 showgrid=False, name='Multi-Layer Perceptron'):
        """ Class for visual representation of a multi-layer 
            perceptron.
            
            Parameters
            ----------
            layer_sizes: list
                Sizes of each layer
            n_color: string (plotly color)
                Color of the neurons
            b_color: string (plotly color)
                Color of the bias
            showgrid: bool
                Set true to show the grid
            name: string
                Title of plot.
        """
        
        self.name = name
        self.layer_sizes = layer_sizes
        self.n_color = n_color
        self.b_color = b_color
        self.showgrid = showgrid
        
        self._assign_x_coords()
        self._assign_x_offsets()
        
    def _assign_x_coords(self):
        """ Assign the x coordinates to the 
            layers in the network.
        """
        # x coordinate of layer is # of layer + 1
        self.x_coords = list()
        for i in range(len(self.layer_sizes)):
            self.x_coords.append(i+1)
        
    def _assign_x_offsets(self):
        """ Assign the x offsets to the 
            layers in the network.
        """
        
        # Set x_offset of largest layer to 1
        max_layer = max(self.layer_sizes)
        max_layer_x_offset = 1
        
        # Set x_offset of other layers
        self.x_offsets = list()
        for i in range(len(self.layer_sizes)):
            self.x_offsets.append(max_layer - self.layer_sizes[i] + max_layer_x_offset)
        
    def plot(self, show_bias=False):
        """ Plot the network.
            
            Parameters
            ----------
            show_bias: bool
                Set true to show bias neurons.
        """
        self.show_bias = show_bias
        
        # Build the layers
        self.layers = list()
        for i in range(len(self.layer_sizes)):
            self.layers.append(Dense(self.layer_sizes[i], 
                                     x_coord=self.x_coords[i], 
                                     x_offset=self.x_offsets[i]))
        
        # Connect the layers
        data, layout = connect_layers(self.layers, self.name, self.showgrid)
        figure = dict(data=data, layout=layout)
        plot(figure)


