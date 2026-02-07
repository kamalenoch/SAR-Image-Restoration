import numpy as np
def enl(roi):
    return (roi.mean()**2) / roi.var()
