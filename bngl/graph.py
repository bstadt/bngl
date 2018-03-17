import numpy as np

class LinearGraph:
    def __init__(self):
        self.layers = []
        return

    def register_layer(self, layer):
        self.layers.append(layer)
