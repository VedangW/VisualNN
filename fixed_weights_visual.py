#!/usr/bin/python

class McCullochPittsVisual:
    
    def __init__(self, X, W, tau, p, out, fps, color='blue'):
        """ Class to create a visual of the Mc-pitts model.
            Specific constants adhere only to this
            particular model. Not compatible with other
            models.
            
            Parameters
            ----------
            X: list
                [x1, x2, x3], the inputs.
            W: list
                [w1, w2, w3], the weights.
            tau: float or int
                Threshold of model
            p: float or int
                Weighted sum
            out: binary
                1 or 0, the output
            fps: int
                Number of points per edge.
                Decides the speed of point
                traversal.
        """
        self.xm = 0
        self.xM = 4
        self.ym = 0
        self.yM = 6
        self.color = color
        
        self.fps = fps
        
        self.x1, self.x2, self.x3 = X
        self.w1, self.w2, self.w3 = W
        self.p = p
        self.tau = tau
        self.out = out
        
        self._set_layout()
        self._set_data_and_frames()
        
    def _set_layout(self):
        """ Sets the layout of the graph. """
        self.layout = dict(xaxis = dict(range=[self.xm, self.xM], autorange=False, zeroline=False),
                        yaxis = dict(range=[self.ym, self.yM], autorange=False, zeroline=False),
                        title = 'McCulloch-Pitts Model', hovermode='closest',
                        updatemenus = [{'type': 'buttons',
                                       'buttons': [{'label': 'Play',
                                                    'method': 'animate',
                                                    'args': [None]}]}])
    
    def _set_data_and_frames(self):
        """ Sets the data and frames. """
        N = self.fps
        
        # Equations of lines and markers
        
        x = np.linspace(1, 2, 100)
        y1 = get_line(x, (1, 1), (2, 3))
        y2 = get_line(x, (1, 3), (2, 3))
        y3 = get_line(x, (1, 5), (2, 3))

        x_ = np.linspace(2, 3, 100)
        y_ = get_line(x_, (2, 3), (3, 3))

        xx = np.linspace(1, 2, N)
        yy1 = get_line(xx, (1, 1), (2, 3))
        yy2 = get_line(xx, (1, 3), (2, 3))
        yy3 = get_line(xx, (1, 5), (2, 3))

        xx_ = np.linspace(2, 3, N)
        yy_ = get_line(xx_, (2, 3), (3, 3))
        
        # Data
        
        # Edge from (1, 1) to (2, 3)
        self.data = [dict(x=x, y=y1, 
                   mode='lines', 
                   line=dict(width=2, color=self.color)
                  ),
              dict(x=x, y=y1, 
                   mode='lines', 
                   line=dict(width=2, color=self.color)
                  )
            ]
        
        # Edge from (1, 3) to (2, 3)
        self.data += [dict(x=x, y=y2, 
                   mode='lines', 
                   line=dict(width=2, color=self.color)
                  ),
              dict(x=x, y=y2, 
                   mode='lines', 
                   line=dict(width=2, color=self.color)
                  )
            ]
        
        # Edge from (1, 5) to (2, 3)
        self.data += [dict(x=x, y=y3, 
                   mode='lines', 
                   line=dict(width=2, color=self.color)
                  ),
              dict(x=x, y=y3, 
                   mode='lines', 
                   line=dict(width=2, color=self.color)
                  )
            ]

        # Edge from (2, 3) to (3, 3)
        self.data += [dict(x=x_, y=y_,
                   mode='lines', 
                   line=dict(width=2, color=self.color)
                  ),
              dict(x=x_, y=y_,
                   mode='lines', 
                   line=dict(width=2, color=self.color)
                  )
            ]

        # Points at the ends of edges
        self.data += [dict(x=[1, 1, 1, 2, 3], y=[1, 3, 5, 3, 3],
                    mode='markers+text',
                    text=['x1 = ' + str(self.x1), 
                          'x2 = ' + str(self.x2), 
                          'x3 = ' + str(self.x3), 
                          'p = ' + str(self.p), 
                          'f(p) = ' + str(self.out)],
                    textposition='bottom center',
                    marker=dict(color=self.color, size=30))]

        # For visual effect
        self.data += [dict(x=[1, 1, 1, 2, 3], y=[1, 3, 5, 3, 3],
                    mode='markers',
                    marker=dict(color='white', size=20))]

        # Frames
        
        self.running_sum = 0
        
        # Transition on edge 1
        self.frames = [dict(data=[dict(x=[xx[k]], 
                                y=[yy1[k]], 
                                mode='markers+text',
                                text=[str(self.x1) + ' x ' + str(self.w1) + ' = ' + str(self.x1*self.w1)],
                                textposition='bottom right',
                                marker=dict(color=self.color, size=10)
                                )
                          ]) for k in range(N)]
        
        self.running_sum += self.x1*self.w1

        # Transition on edge 2
        self.frames += [dict(data=[dict(x=[xx[k]], 
                                y=[yy2[k]], 
                                mode='markers+text',
                                text=[str(self.running_sum) + ' + ' + 
                                      str(self.x2) + ' x ' + 
                                      str(self.w2) + ' = ' + 
                                      str(self.running_sum + self.x2*self.w2)],
                                textposition='top right',
                                marker=dict(color=self.color, size=10)
                                )
                          ]) for k in range(N)]

        self.running_sum += self.x2*self.w2

        # Transition on edge 3
        self.frames += [dict(data=[dict(x=[xx[k]], 
                                y=[yy3[k]], 
                                mode='markers+text',
                                text=[str(self.running_sum) + ' + ' + 
                                      str(self.x3) + ' x ' + 
                                      str(self.w3) + ' = ' + 
                                      str(self.running_sum + self.x3*self.w3)],
                                textposition='top right',
                                marker=dict(color=self.color, size=10)
                                )
                          ]) for k in range(N)]

        self.running_sum += self.x3*self.w3

        # Transition on edge 4
        self.frames += [dict(data=[dict(x=[xx_[k]],
                                y=[yy_[k]],
                                mode='markers+text',
                                text=[str(self.running_sum) + '/' + str(self.tau) + ' = ' + str(self.out)],
                                marker=dict(color=self.color, size=10)
                                )
                          ]) for k in range(N)]
        
    def visualise(self):
        """ Call this function to create the visualisation. """
        self.figure = dict(data=self.data, layout=self.layout, frames=self.frames)
        iplot(self.figure)