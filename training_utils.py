import math
import time

import numpy as np
import MNN
from config import cfg

from sklearn import metrics

nn = MNN.nn
F = MNN.expr

F.lazy_eval(True)


def train(model, train_loader, optimizer, local_iters=None, device=None):
    model.train(True)
    train_loader.reset()
    if local_iters is None:
        local_iters = math.ceil(train_loader.size / train_loader.batch_size)

    train_loss = 0.0
    samples_num = 0

    t0 = time.time()
    for i in range(local_iters):
        example = train_loader.next()
        input_data = example[0]
        output_target = example[1]
        data = input_data[0]      # which input, model may have more than one inputs
        label = output_target[0]  # also, model may have more than one outputs
        # data, label = data.to(device), label.to(device)

        predict = model.forward(data)
        target = F.one_hot(F.cast(label, F.int), cfg['classes_size'], 1, 0)
        loss = nn.loss.cross_entropy(predict, target)
        optimizer.step(loss)

        train_loss += loss.read() * data.shape[0]
        samples_num += data.shape[0]

    t1 = time.time()
    if samples_num != 0:
        train_loss /= samples_num
    return train_loss, t1 - t0


def test(model, test_loader, device=None):
    test_loader.reset()

    # test_loss = 0.0
    # correct = 0
    # for i in range(test_loader.iter_number):
    #     example = test_loader.next()
    #     input_data = example[0]
    #     output_target = example[1]
    #     data = input_data[0]      # which input, model may have more than one inputs
    #     label = output_target[0]  # also, model may have more than one outputs
    #     # data, label = data.to(device), label.to(device)

    #     predict = model.forward(data)
    #     target = F.one_hot(F.cast(label, F.int), cfg['classes_size'], 1, 0)
    #     loss = nn.loss.cross_entropy(predict, target)
    #     test_loss += loss.read() * data.shape[0]
    #     predict = F.argmax(predict, 1)
    #     predict = np.array(predict.read())
    #     label = np.array(label.read())
    #     correct += (np.sum(label == predict))

    # test_acc = correct / test_loader.size
    # test_loss = test_loss / test_loader.size

    example = test_loader.next()
    input_data = example[0]
    output_target = example[1]
    data = input_data[0]      # which input, model may have more than one inputs
    label = output_target[0]  # also, model may have more than one outputs

    predict = model.forward(data)
    target = F.one_hot(F.cast(label, F.int), cfg['classes_size'], 1, 0)
    test_loss = nn.loss.cross_entropy(predict, target).read()

    predict1 = F.argmax(predict, 1)
    predict1 = np.array(predict1.read())
    label = np.array(label.read())
    correct = np.sum(label == predict1)
    print(correct)
    test_acc = correct / test_loader.size
    print(test_loader.size)

    predict = np.array(predict.read())
    predict = predict[:, 1]
    # print(predict)

    e, f, g = metrics.roc_curve(label, predict)
    test_auc = metrics.auc(e, f)
    

    return test_loss, test_acc, test_auc

