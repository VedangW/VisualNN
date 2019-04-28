import numpy as np
import plotly.graph_objs as go

from plotly.offline import plot

class Conv2DLeNetStyle:
    
    def __init__(self, x_init, y_mid, layer_shape=(10, 20), 
                 n_channels=10, col1=(128, 0, 128), col2=(45, 0, 65),
                 x_shift=2, y_shift=5, transparency=0.9):
        
        self.x_init = x_init
        self.y_mid = y_mid
        self.width, self.height = layer_shape
        
        self.n_h = np.max([self.width, self.height])
        self.n_w = int(self.n_h/2)
        
        self.n_c = n_channels
        
        if self.n_c % 2 == 0:
            self.col1 = col1
            self.col2 = col2
        else:
            self.col1 = col2
            self.col2 = col1
            
        self.transparency = transparency
        
        self.x_shift = x_shift
        self.y_shift = y_shift

        self.x_mid_init_box = self.x_init - (self.x_shift * self.n_c/2)
        self.y_mid_init_box = self.y_mid + (self.y_shift * self.n_c/2)

        self.x0 = self.x_mid_init_box - self.n_w/2
        self.x1 = self.x_mid_init_box + self.n_w/2
        self.y0 = self.y_mid_init_box - self.n_h/2
        self.y1 = self.y_mid_init_box + self.n_h/2
        
        self.upper_right_corner = (self.x1, self.y1)
        
        self.layer_text = str(self.n_c) + '@' + str(self.height) + 'x' + str(2*self.width)
        
        self.data = [go.Scatter(
                x=[self.x0],
                y=[int(15*self.y1/14)],
                text=[self.layer_text],
                mode='text',
            )]
        
    def _color_layer(self, colors):
        r, g, b = colors
        
        trans_str = 'rgba(' + str(r) + ', ' +  \
            str(g) + ', ' + str(b) + ', ' + \
            str(self.transparency) + ')'
        line_str = 'rgba(' + str(r) + ', ' + \
            str(g) + ', ' + str(b) + ', ' + '1.0)'

        return line_str, trans_str
        
    def _conv_layer_shapes(self,):
        x0, y0, x1, y1 = self.x0, self.y0, self.x1, self.y1

        colors = {0: self._color_layer(self.col1),
                  1: self._color_layer(self.col2)}

        shapes = list()
        for i in range(self.n_c):
            choice = i % 2

            lcolor, fcolor = colors[choice]

            shapes.append({
                'type': 'rect',
                'x0': x0,
                'y0': y0,
                'x1': x1,
                'y1': y1,
                'line': {
                    'color': lcolor,
                    'width': 2,
                },
                'fillcolor': fcolor,
            })

            
            if i != self.n_c - 1:
                x0 += self.x_shift
                x1 += self.x_shift
                y0 -= self.y_shift
                y1 -= self.y_shift

        self.lower_right_corner = (x1, y0)
        
        self.x0_final = x0
        self.x1_final = x1
        self.y0_final = y0
        self.y1_final = y1
        
        self.final_centre = ((self.x0_final + self.x1_final)/2, 
                             (self.y0_final + self.y1_final)/2)
        
        self.x0_final += self.n_w/2
        self.y0_final += self.n_h/2
        
        self.x1_final -= self.n_w/4
        self.y1_final -= self.n_h/4
        
        shapes.append({
                'type': 'rect',
                'x0': self.x0_final,
                'y0': self.y0_final,
                'x1': self.x1_final,
                'y1': self.y1_final,
                'line': {
                    'color': 'rgba(230, 0, 230, 1)',
                    'width': 2,
                },
                'fillcolor': 'rgba(230, 0, 230, 0.9)',
            })
                
        return shapes
        
    def layer_shapes(self,):
        return self._conv_layer_shapes()
    
    def layer_data(self,):
        return self.data
    
    def corner_points(self,):
        return self.final_centre, (self.x1_final, self.y0_final), (self.x1_final, self.y1_final)



class DenseLeNetStyle:
    
    def __init__(self, x_init, y_mid, width, height, num_neurons, color=(128, 0, 128), transparency=0.9):
        self.x_init = x_init
        self.y_mid = y_mid
        self.width = width
        self.height = height
        self.color = color
        self.num_neurons = num_neurons
        self.transparency = transparency
        
        self.x0 = int(self.x_init - self.width/2)
        self.x1 = int(self.x_init + self.width/2)
        self.y0 = int(self.y_mid - self.height/2)
        self.y1 = int(self.y_mid + self.height/2)
        
        self.data = [go.Scatter(
                x=[self.x_init],
                y=[self.y_mid + self.height*17/28],
                text=['Dense (' + str(self.num_neurons) + ')'],
                mode='text',
            )]
        
    def corner_points(self,):
        lower_left = (self.x0, self.y0)
        lower_right = (self.x1, self.y0)
        upper_left = (self.x0, self.y1)
        upper_right = (self.x1, self.y1)
        
        return (lower_left, lower_right, upper_left, upper_right)
        
    def _color_layer(self, colors):
        r, g, b = colors

        trans_str = 'rgba(' + str(r) + ', ' +  \
            str(g) + ', ' + str(b) + ', ' + \
            str(self.transparency) + ')'
        line_str = 'rgba(' + str(r) + ', ' + \
            str(g) + ', ' + str(b) + ', ' + '1.0)'

        return line_str, trans_str
    
    def layer_data(self,):
        return self.data

    def layer_shapes(self,):
        self.line_str, self.trans_str = self._color_layer(self.color)
        
        shapes = [{'type': 'rect',
                    'x0': self.x0,
                    'y0': self.y0,
                    'x1': self.x1,
                    'y1': self.y1,
                    'line': {
                        'color': self.line_str,
                        'width': 2,
                    },
                    'fillcolor': self.trans_str}]
        
        return shapes


class ConvNet2D:
    
    def __init__(self, layers_conv, layers_dense, scaling_factor=2):
        prods = np.array([layers_conv[i][0]*layers_conv[i][2]/3 for i in range(len(layers_conv))])

        y_mid = np.max(prods)
        x_shift = 2*y_mid/100
        y_shift = 5*y_mid/100

        x_curr = 150

        conv_layers = list()
        for i in range(len(layers_conv)):
            n_w, n_h, n_c = layers_conv[i]
            if i != 0:
                x_curr += 11/10*n_w + x_shift*n_c/2
            conv_layers.append(Conv2DLeNetStyle(x_curr, y_mid, 
                                            layer_shape=(int(n_w/2), n_h), 
                                            n_channels=n_c, x_shift=x_shift, 
                                            y_shift=y_shift))


        all_data = []
        all_shapes = []

        for i in range(len(conv_layers)):
            all_data += conv_layers[i].layer_data()
            all_shapes += conv_layers[i].layer_shapes()

        for i in range(len(conv_layers) - 1):
            _, p1_1, p2_1 = conv_layers[i].corner_points()
            fc_2, _, _ = conv_layers[i+1].corner_points()
            all_shapes += [{
                    'type': 'line',
                    'x0': p1_1[0],
                    'y0': p1_1[1],
                    'x1': fc_2[0],
                    'y1': fc_2[1],
                    'line': {
                        'color': 'rgb(0, 0, 0)',
                        'width': 1,
                    },
                }]
            all_shapes += [{
                    'type': 'line',
                    'x0': p2_1[0],
                    'y0': p2_1[1],
                    'x1': fc_2[0],
                    'y1': fc_2[1],
                    'line': {
                        'color': 'rgb(0, 0, 0)',
                        'width': 1,
                    },
                }]


        max_height = max(layers_dense) * scaling_factor
        const_width = max_height/100 * 5

        x_init = conv_layers[-1].lower_right_corner[0]

        dense_layers = []
        for i in range(len(layers_dense)):
            x_init += 3*const_width
            dense_layers.append(DenseLeNetStyle(x_init, y_mid, const_width, layers_dense[i], layers_dense[i]))

        for i in range(len(dense_layers)):
            all_data += dense_layers[i].layer_data()
            all_shapes += dense_layers[i].layer_shapes()


        for i in range(len(dense_layers) - 1):
            _, lr, _, ur = dense_layers[i].corner_points()
            ll, _, ul,_ = dense_layers[i+1].corner_points()
            all_shapes += [{
                    'type': 'line',
                    'x0': lr[0],
                    'y0': lr[1],
                    'x1': ll[0],
                    'y1': ll[1],
                    'line': {
                        'color': 'rgb(0, 0, 0)',
                        'width': 1,
                    },
                }]
            all_shapes += [{
                    'type': 'line',
                    'x0': ur[0],
                    'y0': ur[1],
                    'x1': ul[0],
                    'y1': ul[1],
                    'line': {
                        'color': 'rgb(0, 0, 0)',
                        'width': 1,
                    },
                }]

        all_shapes += [{
            'type': 'line',
            'x0': conv_layers[-1].lower_right_corner[0],
            'y0': conv_layers[-1].lower_right_corner[1],
            'x1': dense_layers[0].corner_points()[0][0],
            'y1': dense_layers[0].corner_points()[0][1],
            'line': {
                'color': 'rgb(0, 0, 0)',
                'width': 1,
            },
        }]

        all_shapes += [{
            'type': 'line',
            'x0': conv_layers[-1].upper_right_corner[0],
            'y0': conv_layers[-1].upper_right_corner[1],
            'x1': dense_layers[0].corner_points()[2][0],
            'y1': dense_layers[0].corner_points()[2][1],
            'line': {
                'color': 'rgb(0, 0, 0)',
                'width': 1,
            },
        }]

        layout = {
            'xaxis': {'range': [0, int(2*y_mid)], 'showgrid': False, 'showticklabels': False},
            'yaxis': {'range': [0, int(2*y_mid)], 'showgrid': False, 'showticklabels': False},
            'shapes': all_shapes
        }

        self.fig = {
            'data': all_data,
            'layout': layout,
        }
        
    def plot(self,):
        plot(self.fig)