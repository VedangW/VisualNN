#!/usr/bin/python

class McCullochPitts:
    
    def __init__(self, X, W, tau, fps=2):
        """ Class that performs the computation part
            of the McCullochPitts model.
            
            Parameters
            ----------
            X: list
                [x1, x2, x3], the inputs.
            W: list
                [w1, w2, w3], the weights.
            tau: float or int
                Threshold of model
            fps: int
                Number of points per edge.
                Decides the speed of point
                traversal.
            """
        self.X = X
        self.W = W
        self.tau = tau
        self.fps = fps
        
        self._propagate()
        
    def _propagate(self):
        """ Forward propagation of model. """
        self.p = np.dot(self.X, self.W)
        self.out = self._activation()
    
    def _activation(self):
        """ Application of activation function. """
        if self.p > self.tau:
            return 1
        return 0
    
    def visualise(self, color='blue'):
        """ Call this function to create a visualisation. """
        self.color = color
        
        mc_pitts_visual = McCullochPittsVisual(self.X, 
                                               self.W, 
                                               self.tau, 
                                               self.p, 
                                               self.out, 
                                               self.fps, 
                                               self.color)
        mc_pitts_visual.visualise()