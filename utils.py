#!/usr/bin/python

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