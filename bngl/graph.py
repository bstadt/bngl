import numpy as np

class LinearGraph:
    def __init__(self):
        self.layers = []
        return

    def add_operation(self, layer):
        #TODO check if layer is operation
        self.layers.append(layer)
        return

    def train_on_batch(self, batch):
        batch_loss = 0
        for x in batch:
            for layer in self.layers:
                x = layer.do_operation(x)
            batch_loss += x

            final_output_size = np.prod(self.layers[-1].output_shape)
            dloss_dout = np.ones((final_output_size, ))
            for idx in range(len(self.layers)-1, -1, -1):
                layer = self.layers[idx]
                if layer.trainable:
                    layer.register_weight_gradient(dloss_dout)
                dout_din = layer.get_input_gradient()
                dloss_dout = dloss_dout @ dout_din

        for layer in self.layers:
            if layer.trainable:
                layer.update()

        return batch_loss
