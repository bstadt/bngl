import warnings
import numpy as np
from bngl.operation import Operation
from bngl.layer import FullyConnected1D

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
        raise
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
        raise

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
        raise

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
        failed_tests.append('initializer does not throw error when input_shape is not rank 2')

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
        failed_tests.append('initializer does not throw error when output_shape is not rank 2')


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
        failed_tests.append('initializer does not throw error when input_shape is not (-1, 1)')


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
        failed_tests.append('initializer does not throw error when output_shape is not (-1, 1)')

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
        failed_tests.append('initializer_fn does not correctly initialize weights')


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
        failed_tests.append('initializer does not throw error when initializer_fn returns non-list')

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
        failed_tests.append('no error thrown when initializer_fn returns incorrect number of trainable params')


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
        raise
    if not passed:
        failed_tests.append('FullyConnected1D does not complete 1 iteration of gradient descent')


    return failed_tests


if __name__ == '__main__':
    operation_failed_tests = operation_tests()
    if len(operation_failed_tests) > 0:
        print('Operation Failures:')
        for elem in operation_failed_tests:
            print('\t', elem)
        print('\n')

    layer_failed_tests = layer_tests()
    if len(layer_failed_tests) > 0:
        print('Layer Failures:')
        for elem in layer_failed_tests:
            print('\t', elem)
        print('\n')


    print('Testing Complete!')
