import numpy as np
from bngl.activation import Relu
from bngl.graph import LinearGraph
from bngl.loss import SoftmaxCrossEntropy
from bngl.layer import FullyConnected1D, Bias1D

def get_example(class_label):
    class_features = None
    if class_label == 0:
        class_features = [np.random.normal(-2, 1)
                         for _ in range(2)]
    else:
        class_features = [np.random.normal(2, 1)
                         for _ in range(2)]

    noise_features = [np.random.normal(loc=0, scale=1.0)\
                      for _ in range(3)]

    return np.array(noise_features + class_features).reshape(-1, 1)

def sgd(gradients, trainable_parameters):
    trainable_parameters = trainable_parameters.reshape(-1, 1)
    gradient = np.mean(gradients, axis=0).reshape(-1, 1)
    trainable_parameters = trainable_parameters - .001 * gradient
    return trainable_parameters.flatten()


if __name__ == '__main__':
    #Build Network
    my_nn = LinearGraph()

    my_nn.add_operation(FullyConnected1D((5, 1),
                                          (100, 1),
                                          sgd))
    my_nn.add_operation(Bias1D((100, 1),
                                sgd))

    my_nn.add_operation(Relu((100, 1)))

    my_nn.add_operation(FullyConnected1D((100, 1),
                                          (2, 1),
                                          sgd))
    my_nn.add_operation(Bias1D((2, 1),
                                sgd))


    my_nn.add_loss(SoftmaxCrossEntropy((2, 2, 1)))

    #Train
    batch_size = 16
    num_train_batches = 500
    for batch in range(num_train_batches):
        cur_class_breakdown = [np.random.choice([0, 1]) for _ in range(batch_size)]
        cur_batch = [get_example(elem) for elem in cur_class_breakdown]
        cur_labels = []


        for cur_class in cur_class_breakdown:
            if cur_class:
                cur_labels.append(np.array([0, 1]).reshape(-1, 1))
            else:
                cur_labels.append(np.array([1, 0]).reshape(-1, 1))

        correct = 0
        preds = [my_nn.forward(elem) for elem in cur_batch]
        for idx, pred in enumerate(preds):
            if np.argmax(cur_labels[idx]) == np.argmax(pred):
                correct+=1
        print('Batch Acc: ', correct/len(preds))

        batch_loss = np.squeeze(my_nn.train_on_batch(cur_batch, cur_labels))/batch_size

    #Test
    batch_size=1000
    cur_batch = [get_example(0) for _ in range(batch_size//2)] \
                 + [get_example(1) for _ in range(batch_size//2)]

    cur_labels = [np.array([1, 0]).reshape(-1, 1) for _ in range(batch_size//2)] \
                  + [np.array([0, 1]).reshape(-1, 1) for _ in range(batch_size//2)]

    correct = 0
    preds = [my_nn.forward(elem) for elem in cur_batch]
    for idx, pred in enumerate(preds):
        if np.argmax(cur_labels[idx]) == np.argmax(pred):
            correct+=1
    print('\nTest Set Acc: ', correct/len(preds))
