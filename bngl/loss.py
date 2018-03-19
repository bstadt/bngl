import numpy as np
from .operation import Operation

class SoftmaxCrossEntropy(Operation):
    def __init__(self, input_shape):

        def operation_fn(x, params):
            m = np.max(x)
            smax = np.exp(x-m)/(np.sum(np.exp(x-m)))

            #NOTE this is done for numerical stability in the log
            smax = np.maximum(smax, np.ones_like(smax) * 1e-7)
            return (-1 * x.T @ np.log(smax)).reshape(-1, 1)

        def input_gradient_fn(x, params):
            m = np.max(x)
            smax = np.exp(x-m)/(np.sum(np.exp(x-m)))
            return (smax - x).reshape(1, -1)

        super(SoftmaxCrossEntropy, self).__init__(input_shape,
                                                  (1, 1),
                                                  operation_fn=operation_fn,
                                                  input_gradient_fn=input_gradient_fn,
                                                  trainable=False)
