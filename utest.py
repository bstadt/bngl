import sys
import warnings
import numpy as np
from bngl.activation import Relu
from bngl.graph import LinearGraph
from bngl.operation import Operation
from bngl.loss import SoftmaxCrossEntropy, MSE
from bngl.layer import FullyConnected1D, Bias1D

def operation_tests():
    failed_tests = []

    #initializer warns when trainable_parameters is None and layer is trainable
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        try:
            my_operation = Operation((1, 1),
                                     (1, 1),
                                     operation_fn=lambda x, y: np.identity(1).reshape(-1, 1),
                                     input_gradient_fn=lambda x, y: np.identity(1).reshape(-1, 1),
                                     update_fn=lambda x, y: 1,
                                     trainable=True)

            if not len(w)==1:
                failed_tests.append('initializer does not warn when trainable parameters is None and layer is trainable')
        except:
            failed_tests.append('initializer does not warn when trainable parameters is None and layer is trainable')


    #initializer warns when trainable_parameters is not None and layer is not trainable
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        try:
            my_operation = Operation((1, 1),
                                     (1, 1),
                                     operation_fn=lambda x, y: np.identity(1).reshape(-1, 1),
                                     input_gradient_fn=lambda x, y: np.identity(1).reshape(-1, 1),
                                     trainable_parameters = [.1],
                                     trainable=False)
            if not len(w)==1:
                failed_tests.append('initializer does not warn when trainable_parameters are passed to non trainable layer')
        except:
            failed_tests.append('initializer does not warn when trainable_parameters are passed to non trainable layer')



    #initializer fails when operation_fn does not return numpy array
    passed = False
    try:
        my_operation = Operation((1, 1),
                                 (1, 1),
                                 operation_fn=lambda x, y: 1,
                                 input_gradient_fn=lambda x, y: np.identity(1).reshape(-1, 1),
                                 trainable=False)
    except TypeError:
        passed=True
    except:
        pass
    if not passed:
        failed_tests.append('Initializer does not fail when operation_fn does not return numpy array')


    #initializer fails when operation_fn does not return correct shape
    passed = False
    try:
        my_operation = Operation((1, 1),
                                 (1, 1),
                                 operation_fn=lambda x, y: np.identity(2),
                                 input_gradient_fn=lambda x, y: np.identity(1).reshape(-1, 1),
                                 trainable=False)
    except ValueError:
        passed=True
    except:
        pass
    if not passed:
        failed_tests.append('Initializer does not fail when operation_fn returns array of wrong shape')


    #initializer fails when gradient_fn does not return np.ndarray
    passed = False
    try:
        my_operation = Operation((2, 1),
                                 (2, 1),
                                 operation_fn=lambda x, y: np.arange(2).reshape(-1, 1),
                                 input_gradient_fn=lambda x, y: 1,
                                 trainable=False)
    except TypeError:
        passed=True
    except:
        pass
    if not passed:
        failed_tests.append('Initializer does not fail when gradient_fn does not return numpy array')


    #initializer fails when gradient_fn does not return correct shape
    passed = False
    try:
        my_operation = Operation((2, 1),
                                 (2, 1),
                                 operation_fn=lambda x, y: np.arange(2).reshape(-1, 1),
                                 input_gradient_fn=lambda x, y: np.identity(3),
                                 trainable=False)
    except ValueError:
        passed=True
    except:
        pass
    if not passed:
        failed_tests.append('Initializer does not fail when gradient_fn returns array of wrong shape')


    #initializer fails when update_fn is none and layer is trainable
    passed = False
    try:
        my_operation = Operation((1, 1),
                                 (1, 1),
                                 operation_fn=lambda x, y: np.identity(1).reshape(-1, 1),
                                 input_gradient_fn=lambda x, y: np.identity(1).reshape(-1, 1),
                                 trainable_parameters=[1.],
                                 trainable=True)
    except ValueError:
        passed=True
    except:
        pass
    if not passed:
        failed_tests.append('Initializer does not fail when update_fn is none and layer is trainable')


    #initializer warns when update_fn is not none and layer is not trainable
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        try:
            my_operation = Operation((1, 1),
                                     (1, 1),
                                     operation_fn=lambda x, y: np.identity(1).reshape(-1, 1),
                                     input_gradient_fn=lambda x, y: np.identity(1).reshape(-1, 1),
                                     update_fn=lambda x, y: 1,
                                     trainable=False)
            if not len(w)==1:
                failed_tests.append('Initializer does not warn when update_fn is not none and layer is not trainable')

        except:
            failed_tests.append('Initializer does not warn when update_fn is not none and layer is not trainable')


    #do_operation fails when passed non numpy array
    passed = False
    try:
        my_operation = Operation((2, 1),
                                 (2, 1),
                                 operation_fn=lambda x, y: x,
                                 input_gradient_fn=lambda x, y: np.identity(2),
                                 trainable=False)

        my_operation.do_operation(1)

    except TypeError:
        passed=True
    except:
        pass
    if not passed:
        failed_tests.append('do_operation does not fail when passed data not of type numpy.ndarray')


    #do_operation fails when passed incorrect input shape
    passed = False
    try:
        my_operation = Operation((2, 1),
                                 (2, 1),
                                 operation_fn=lambda x, y: x,
                                 input_gradient_fn=lambda x, y: np.identity(4),
                                 trainable=False)

        my_operation.do_operation(np.identity(3))

    except ValueError:
        passed=True
    except:
        pass
    if not passed:
        failed_tests.append('do_operation does not fail when passed incorrect input shape')


    #do_operation sets last input
    passed = False
    try:
        my_operation = Operation((2, 1),
                                 (2, 1),
                                 operation_fn=lambda x, y: x,
                                 input_gradient_fn=lambda x, y: np.identity(2),
                                 trainable=False)

        my_operation.do_operation(np.arange(2).reshape(-1, 1))
        delta_mat = np.abs(my_operation.last_input - np.arange(2).reshape(-1, 1))
        if np.allclose(np.sum(delta_mat), 0.):
            passed = True
    except:
        pass
    if not passed:
        failed_tests.append('do_operation does not set last operation input')


    #get_input_gradient gets input gradient
    passed = False
    try:
        my_operation = Operation((2, 1),
                                 (2, 1),
                                 operation_fn=lambda x, y: x,
                                 input_gradient_fn=lambda x, y: np.identity(2) * x,
                                 trainable=False)

        my_operation.do_operation(np.arange(2).reshape(-1, 1))
        expected_out = np.identity(2) * np.arange(2).reshape(-1, 1)
        delta_mat = np.abs(my_operation.get_input_gradient() - expected_out)
        if np.allclose(np.sum(delta_mat), 0.):
            passed=True
    except:
        pass
    if not passed:
        failed_tests.append('get_gradient does not get gradient')

    #for use in tests below
    def test_weight_grad_fn(dloss_dout, last_input):
        dloss_dout = np.array(dloss_dout)
        last_input = np.squeeze(last_input)
        dout_dweight = np.stack([[last_input[0], last_input[1], 0, 0],
                                 [0, 0, last_input[0], last_input[1]]])
        return dloss_dout @ dout_dweight

    def test_update_fn(gradients, trainable_parameters):
        trainable_parameters = np.array(trainable_parameters).reshape((2, 2))
        gradient = np.sum(gradients, axis=0).reshape((2, 2))
        updated_params = trainable_parameters - .5 * gradient
        return updated_params.flatten()


    #register_weight_gradient registers gradient
    passed = False
    try:
        my_operation = Operation((2, 1),
                                 (2, 1),
                                 operation_fn=lambda x, y: np.array(y).reshape(2, 2) @ x,
                                 input_gradient_fn=lambda x, y: np.array(y).reshape(2, 2),
                                 weight_gradient_fn=test_weight_grad_fn,
                                 update_fn=test_update_fn,
                                 trainable_parameters=np.array([1., 0., 0., 1.]),
                                 trainable=True)

        #required to populate last_input
        my_operation.do_operation(np.array([1, 2]).reshape(-1, 1))
        my_operation.register_weight_gradient(np.array([1, 1]).reshape(1, 2))
        expected_out = np.array([1, 2, 1, 2])
        delta_mat = np.abs(my_operation.weight_gradients[0] - expected_out)
        if np.allclose(np.sum(delta_mat), 0.):
            passed = True

    except:
        pass
    if not passed:
        failed_tests.append('register_gradient does not register gradient')

    #register_weight_gradient accumulates gradients
    passed = False
    try:
        my_operation = Operation((2, 1),
                                 (2, 1),
                                 operation_fn=lambda x, y: np.array(y).reshape(2, 2) @ x,
                                 input_gradient_fn=lambda x, y: np.array(y).reshape(2, 2),
                                 weight_gradient_fn=test_weight_grad_fn,
                                 update_fn=test_update_fn,
                                 trainable_parameters=np.array([1., 0., 0., 1.]),
                                 trainable=True)

        #required to populate last_input
        my_operation.do_operation(np.array([1, 2]).reshape(-1, 1))
        my_operation.register_weight_gradient(np.array([1, 1]).reshape(1, 2))
        my_operation.do_operation(np.array([1, 2]).reshape(-1, 1))
        my_operation.register_weight_gradient(np.array([1, 1]).reshape(1, 2))

        if len(my_operation.weight_gradients) == 2:
            passed = True

    except:
        pass
    if not passed:
        failed_tests.append('register_gradient does not register gradient')


    #register_weight_gradient handles bad input data type
    passed = False
    try:
        my_operation = Operation((2, 1),
                                 (2, 1),
                                 operation_fn=lambda x, y: np.array(y).reshape(2, 2) @ x,
                                 input_gradient_fn=lambda x, y: np.array(y).reshape(2, 2),
                                 weight_gradient_fn=test_weight_grad_fn,
                                 update_fn=test_update_fn,
                                 trainable_parameters=np.array([1., 0., 0., 1.]),
                                 trainable=True)

        #required to populate last_input
        my_operation.do_operation(np.array([1, 2]).reshape(-1, 1))
        my_operation.register_weight_gradient('cat')

    except TypeError:
        passed = True
    except:
        pass

    if not passed:
        failed_tests.append('register_gradient does not handle bad input dtype')



    #register_weight_gradient throws error if do_operation is not done first
    passed = False
    try:
        my_operation = Operation((2, 1),
                                 (2, 1),
                                 operation_fn=lambda x, y: np.array(y).reshape(2, 2) @ x,
                                 input_gradient_fn=lambda x, y: np.array(y).reshape(2, 2),
                                 weight_gradient_fn=test_weight_grad_fn,
                                 update_fn=test_update_fn,
                                 trainable_parameters=np.array([1., 0., 0., 1.]),
                                 trainable=True)

        #required to populate last_input
        my_operation.register_weight_gradient(np.array([1, 2]).reshape(-1, 1))

    except ValueError:
        passed = True
    except:
        pass

    if not passed:
        failed_tests.append('register_gradient does not fail when last_input is None')


    #update updates
    passed = False
    try:
        my_operation = Operation((2, 1),
                                 (2, 1),
                                 operation_fn=lambda x, y: np.array(y).reshape(2, 2) @ x,
                                 input_gradient_fn=lambda x, y: np.array(y).reshape(2, 2),
                                 weight_gradient_fn=test_weight_grad_fn,
                                 update_fn=test_update_fn,
                                 trainable_parameters=np.array([1., 0., 0., 1.]),
                                 trainable=True)

        my_operation.do_operation(np.array([1, 2]).reshape(-1, 1))
        my_operation.register_weight_gradient(np.array([1, 1]).reshape(1, 2))
        my_operation.update()
        delta = np.sum(abs(my_operation.trainable_parameters - [.5, -1, -.5, 0]))
        if np.allclose(delta, 0.):
            passed = True

    except:
        pass
    if not passed:
        failed_tests.append('update gradient does not correctly update weights')

    #update autoclears gradients by default
    passed = False
    try:
        my_operation = Operation((2, 1),
                                 (2, 1),
                                 operation_fn=lambda x, y: np.array(y).reshape(2, 2) @ x,
                                 input_gradient_fn=lambda x, y: np.array(y).reshape(2, 2),
                                 weight_gradient_fn=test_weight_grad_fn,
                                 update_fn=test_update_fn,
                                 trainable_parameters=np.array([1., 0., 0., 1.]),
                                 trainable=True)

        my_operation.do_operation(np.array([1, 2]).reshape(-1, 1))
        my_operation.register_weight_gradient(np.array([1, 1]).reshape(1, 2))
        my_operation.update()
        if len(my_operation.weight_gradients)==0:
            passed=True

    except:
        pass

    if not passed:
        failed_tests.append('update gradient does not clear gradients by default')

    #update autoclear_gradients=False does not clear gradients
    passed = False
    try:
        my_operation = Operation((2, 1),
                                 (2, 1),
                                 operation_fn=lambda x, y: np.array(y).reshape(2, 2) @ x,
                                 input_gradient_fn=lambda x, y: np.array(y).reshape(2, 2),
                                 weight_gradient_fn=test_weight_grad_fn,
                                 update_fn=test_update_fn,
                                 trainable_parameters=np.array([1., 0., 0., 1.]),
                                 trainable=True)

        my_operation.do_operation(np.array([1, 2]).reshape(-1, 1))
        my_operation.register_weight_gradient(np.array([1, 1]).reshape(1, 2))
        my_operation.update(autoclear_gradients=False)
        if len(my_operation.weight_gradients)==1:
            passed=True

    except:
        pass

    if not passed:
        failed_tests.append('update gradient clears gradients even when autoclear_gradients=False')

    return failed_tests


def layer_tests():
    failed_tests = []

    #initializer throws error when input_shape is not rank 2
    passed = False
    try:
        my_fc = FullyConnected1D((2,),
                                 (2, 1),
                                 lambda x, y: y-.5*np.sum(x, axis=0))
    except ValueError:
        passed = True
        pass
    except:
        pass
    if not passed:
        failed_tests.append('FullyConnected1D initializer does not throw error when input_shape is not rank 2')

    #initializer throws error when output_shape is not rank 2
    passed = False
    try:
        my_fc = FullyConnected1D((2, 1),
                                 (2,),
                                 lambda x, y: y-.5*np.sum(x, axis=0))
    except ValueError:
        passed = True
        pass
    except:
        pass
    if not passed:
        failed_tests.append('FullyConnected1D initializer does not throw error when output_shape is not rank 2')


    #initializer throws error when input_shape is not (-1, 1)
    passed = False
    try:
        my_fc = FullyConnected1D((2, 2),
                                 (2, 1),
                                 lambda x, y: y-.5*np.sum(x, axis=0))
    except ValueError:
        passed = True
        pass
    except:
        pass
    if not passed:
        failed_tests.append('FullyConnected1D initializer does not throw error when input_shape is not (-1, 1)')


    #initializer throws error when output_shape is not (-1, 1)
    passed = False
    try:
        my_fc = FullyConnected1D((2, 1),
                                 (2, 2),
                                 lambda x, y: y-.5*np.sum(x, axis=0))
    except ValueError:
        passed = True
        pass
    except:
        pass
    if not passed:
        failed_tests.append('FullyConnected1D initializer does not throw error when output_shape is not (-1, 1)')

    #initializer_fn initializes weights
    passed = False
    try:
        my_fc = FullyConnected1D((2, 1),
                                 (2, 1),
                                 lambda x, y: y-.5*np.sum(x, axis=0),
                                 lambda : np.array([.1 for _ in range(4)]))

        delta =  my_fc.trainable_parameters - np.array([.1 for _ in range(4)])
        if np.allclose(delta, 0.):
            passed = True
    except:
        pass
    if not passed:
        failed_tests.append('FullyConnected1D initializer_fn does not correctly initialize weights')


    #initializer throws error when initializer_fn does not return np.array
    passed = False
    try:
        my_fc = FullyConnected1D((2, 1),
                                 (2, 1),
                                 lambda x, y: y-.5*np.sum(x, axis=0),
                                 lambda : 'cat')
    except TypeError:
        passed = True
        pass
    except:
        pass
    if not passed:
        failed_tests.append('FullyConnected1D initializer does not throw error when initializer_fn returns non-list')

    #initializer throws error when initializer_fn does not return correct shape
    passed = False
    try:
        my_fc = FullyConnected1D((2, 1),
                                 (2, 1),
                                 lambda x, y: y-.5*np.sum(x, axis=0),
                                 lambda : np.array([.1 for _ in range(3)]))
    except ValueError:
        passed = True
        pass
    except:
        pass
    if not passed:
        failed_tests.append('FullyConnected1D doesnt throw error when initializer_fn returns incorrect number of trainable params')


    #FullyConnected1D completes 1 iteration of gradient descent
    passed = False
    try:
        my_fc = FullyConnected1D((2, 1),
                                 (2, 1),
                                 lambda x, y: y-.5*np.sum(x, axis=0),
                                 lambda : np.array([1., 0., 0., 1.]))

        my_fc.do_operation(np.array([1, 2]).reshape(-1, 1))
        my_fc.register_weight_gradient(np.array([1, 1]).reshape(1, 2))
        my_fc.update()
        delta = np.sum(abs(my_fc.trainable_parameters - [.5, -1, -.5, 0]))
        if np.allclose(delta, 0.):
            passed = True
    except:
        pass
    if not passed:
        failed_tests.append('FullyConnected1D does not complete 1 iteration of gradient descent')

    #Bias1D throws error if input_shape rank is not 2
    passed = False
    try:
        my_b = Bias1D((2,),
                      lambda x, y: y-.5*np.sum(x, axis=0))

    except ValueError:
        passed = True
    except:
        pass
    if not passed:
        failed_tests.append('Bias1D does not throw error when input is not rank 2')

    #Bias1D throws error if input_shape is not (-1, 1)
    passed = False
    try:
        my_b = Bias1D((1,2),
                      lambda x, y: y-.5*np.sum(x, axis=0))

    except ValueError:
        passed = True
    except:
        pass
    if not passed:
        failed_tests.append('Bias1D does not throw error when input is not (-1, 1)')

    #Bias1D throws error if initializer_fn doesnt return np.ndarray
    passed = False
    try:
        my_b = Bias1D((2,1),
                      lambda x, y: y-.5*np.sum(x, axis=0),
                      initializer_fn=lambda:'cat')

    except TypeError:
        passed = True
    except:
        pass
    if not passed:
        failed_tests.append('Bias1D does not throw error when initializer fn returns non np.ndarray')
    #Bias1D throws error if initializer_fn doesnt return correct number of params
    passed = False
    try:
        my_b = Bias1D((2,1),
                      lambda x, y: y-.5*np.sum(x, axis=0),
                      initializer_fn=lambda: np.array([1]).reshape(1,))

    except ValueError:
        passed = True
    except:
        pass
    if not passed:
        failed_tests.append('Bias1D does not throw error when initializer fn returns incorrect number of params')

    #Bias1D computs correctly
    passed = False
    try:
        my_b = Bias1D((2,1),
                      lambda x, y: y-.5*np.sum(x, axis=0),
                      initializer_fn=lambda: np.array([1., 1.]).reshape(2,))
        out = my_b.do_operation(np.array([1., 1.]).reshape(-1, 1))
        delta = out - np.array([2, 2]).reshape(-1, 1)
        if np.allclose(delta, 0.):
            passed = True
    except:
        pass
    if not passed:
        failed_tests.append('Bias1D does not compute correctly')


    #Bias1D computes input gradient correctly
    passed = False
    try:
        my_b = Bias1D((2,1),
                      lambda x, y: y-.5*np.sum(x, axis=0),
                      initializer_fn=lambda: np.array([1., 1.]).reshape(2,))
        my_b.do_operation(np.array([1., 1.]).reshape(-1, 1))
        out = my_b.get_input_gradient()
        delta = out - np.identity(2)
        if np.allclose(delta, 0.):
            passed = True
    except:
        pass
    if not passed:
        failed_tests.append('Bias1D does not compute input gradient correctly')

    #Bias1D can perform 1 full iteration of gradient descent
    passed = False
    try:
        my_b = Bias1D((2,1),
                      lambda x, y: y-.5*np.sum(x, axis=0),
                      initializer_fn=lambda: np.array([1., 1.]).reshape(2,))
        my_b.do_operation(np.array([1., 1.]).reshape(-1, 1))
        my_b.register_weight_gradient(np.array([1., 1.]).reshape((1, 2)))
        my_b.update()
        delta = my_b.trainable_parameters - [.5, .5]
        if np.allclose(delta, 0.):
            passed = True
    except:
        pass
    if not passed:
        failed_tests.append('Bias1D does not complete 1 iteration of gradient descent correctly')

    return failed_tests


def activation_tests():
    failed_tests = []

    #Relu does Relu
    passed = False
    try:
        my_relu = Relu((2, 1))
        out = my_relu.do_operation(np.array([2, -1]).reshape(-1, 1))
        delta = out - np.array([2, 0]).reshape(-1, 1)
        if np.allclose(delta, 0.):
            passed = True
    except:
        pass
    if not passed:
        failed_tests.append('Relu does not compute Relu')

    #Relu coputes input_gradient_fn correctly
    passed = False
    try:
        my_relu = Relu((2, 1))
        my_relu.do_operation(np.array([2, -1]).reshape(-1, 1))
        input_grad = my_relu.get_input_gradient()
        expected = np.array([[1., 0.],[0., 0.]])
        delta = input_grad - expected
        if np.allclose(delta, 0.):
            passed = True
    except:
        pass
    if not passed:
        failed_tests.append('Relu does not compute input gradient correctly')

    return failed_tests


def loss_tests():
    failed_tests = []

    #Softmax Cross Entropy Computes Correctly
    passed = False
    try:
        my_smce = SoftmaxCrossEntropy((2, 2, 1))
        x = np.array([2, 1]).reshape(-1, 1)
        y = np.array([2, 1]).reshape(-1, 1)
        out = my_smce.do_operation(np.stack([x, y]))
        expected_out = -1 * (2*np.log(1/(1+1/np.e)) + np.log((1/np.e)/(1+1/np.e)))
        delta = out - expected_out
        if np.allclose(delta, 0.):
            passed = True
    except:
        pass
    if not passed:
        failed_tests.append('SoftmaxCrossEntropy does not compute correctly')



    #Softmax Cross Entropy Computes Gradient Correctly
    passed = False
    try:
        my_smce = SoftmaxCrossEntropy((2, 2, 1))
        x = np.array([2, 1]).reshape(-1, 1)
        y = np.array([2, 1]).reshape(-1, 1)
        out = my_smce.do_operation(np.stack([x, y]))
        out = my_smce.get_input_gradient()
        expected_out = np.array([1/(1+1/np.e) - 2, (1/np.e)/(1+1/np.e) - 1, 0., 0.]).reshape(1, -1)
        delta = out - expected_out
        if np.allclose(delta, 0.):
            passed = True
    except:
        pass
    if not passed:
        failed_tests.append('SoftmaxCrossEntropy does not compute gradient correctly')

    #MSE Computes Correctly
    passed = False
    try:
        my_mse = MSE((2, 2, 1))
        x = np.array([1, 1]).reshape(-1, 1)
        y = np.array([2, 0]).reshape(-1, 1)
        out = my_mse.do_operation(np.stack([x, y]))
        expected_out = 1
        delta = out - expected_out
        if np.allclose(delta, 0.):
            passed = True
    except:
        pass
    if not passed:
        failed_tests.append('MSE does not compute correctly')


    #MSE Computes Input Gradient Correctly
    passed = False
    try:
        my_mse = MSE((2, 2, 1))
        x = np.array([1, 1]).reshape(-1, 1)
        y = np.array([2, 0]).reshape(-1, 1)
        _ = my_mse.do_operation(np.stack([x, y]))
        out = my_mse.get_input_gradient().reshape(-1, 1)
        expected_out = np.array([-2., 2., 0., 0.]).reshape(-1, 1)
        delta = out - expected_out
        if np.allclose(delta, 0.):
            passed = True
    except:
        raise
    if not passed:
        failed_tests.append('MSE does not compute gradient correctly')

    return failed_tests

def graph_tests():
    failed_tests = []

    def sgd(gradients, trainable_parameters):
        gradient = np.sum(gradients, axis=0)
        trainable_parameters = trainable_parameters - .5 * gradient
        return trainable_parameters

    #LinearGraph ad_opperation adds operation
    passed = False
    try:
        my_net = LinearGraph()
        my_net.add_operation(FullyConnected1D((2, 1),
                                              (2, 1),
                                              sgd,
                                              initializer_fn=lambda:np.ones((4,))))
        if len(my_net.layers) == 1:
            passed = True
    except:
        raise
    if not passed:
        failed_tests.append('LinearGraph add_operation fails to add operation')

    #LinearGraph correctly backprops
    dummy_loss_op = Operation((2, 2, 1),
                              (1, 1),
                              operation_fn=lambda x, y: np.array(1).reshape(1, 1),
                              input_gradient_fn=lambda x, y: np.array(list(x[1]) + [0, 0]).reshape(1, -1),
                              trainable=False)
    passed = False
    try:
        my_net = LinearGraph()
        my_net.add_operation(FullyConnected1D((2, 1),
                                              (2, 1),
                                              sgd,
                                              initializer_fn=lambda:np.ones((4,))))

        my_net.add_operation(Bias1D((2, 1),
                                    sgd,
                                    initializer_fn=lambda:np.ones((2,))))
        my_net.add_operation(Relu((2, 1)))

        my_net.add_loss(dummy_loss_op)

        my_net.train_on_batch([np.array([1., 1.]).reshape(-1, 1)],
                              [np.array([1., 1.]).reshape(-1, 1)])



        dense_params = my_net.layers[0].trainable_parameters
        delta1 = dense_params - .5*np.ones((4, ))
        bias_params = my_net.layers[1].trainable_parameters
        delta2 = bias_params - .5*np.ones(2, )
        if np.allclose(delta1, 0.) and np.allclose(delta2, 0.):
            passed = True

    except:
        raise
    if not passed:
        failed_tests.append('LinearGraph fails to backprop correctly')




    return failed_tests



if __name__ == '__main__':
    exit_code = 0

    operation_failed_tests = operation_tests()
    if len(operation_failed_tests) > 0:
        exit_code = 1
        print('Operation Failures:')
        for elem in operation_failed_tests:
            print('\t', elem)
        print('\n')

    layer_failed_tests = layer_tests()
    if len(layer_failed_tests) > 0:
        exit_code = 1
        print('Layer Failures:')
        for elem in layer_failed_tests:
            print('\t', elem)
        print('\n')

    activation_failed_tests = activation_tests()
    if len(activation_failed_tests) > 0:
        exit_code = 1
        print('Activation Failures:')
        for elem in activation_failed_tests:
            print('\t', elem)
        print('\n')

    loss_failed_tests = loss_tests()
    if len(loss_failed_tests) > 0:
        exit_code = 1
        print('Loss Failures:')
        for elem in loss_failed_tests:
            print('\t', elem)
        print('\n')


    graph_failed_tests = graph_tests()
    if len(graph_failed_tests) > 0:
        exit_code = 1
        print('Graph Failures:')
        for elem in graph_failed_tests:
            print('\t', elem)
        print('\n')

    print('Testing Complete!')
    sys.exit(exit_code)
