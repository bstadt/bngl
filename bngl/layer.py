import numpy as np
from operation import Operation

class FullyConnected(Operation):
    def __init__(self,
                 input_shape,
                 output_shape,
                 update_fn,
                 initializer_fn=None):

        weight_shape = (output_shape, input_shape)
        if initializer_fn is None:
            trainable_parameters = np.random.multivariate_normal(0, 1, weight_shape)
        else
            trainable_parameters = initializer_fn()
            if trainable_parameters.shape != weight_shape:
                raise ValueError('initializer function must return shape: ', weight_shape, ' but returned shape: ', trainable_parameters.shape)

        def operation_fn(x, params):
            params = params.reshape(weight_shape)
            return params @ x

        def input_gradient_fn(last_input, params):
            return params.reshape(weight_shape)

        def weight_gradient_fn(dloss_dout, last_input):
            dout_dweight = np.stack([last_input for _ in range(output_shape)])
            dloss_dweight = np.flatten(dloss_dout.T * dout_dweight)
            return dloss_dweight

        super(FullyConnected, self).__init__(input_shape,
                                              output_shape,
                                              operation_fn=operation_fn,
                                              input_gradient_fn=input_gradient_fn,
                                              weight_gradient_fn=weight_gradient_fn,
                                              update_fn=update_fn,
                                              trainable=True,
                                              trainable_parameters=trainable_parameters)
