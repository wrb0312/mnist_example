"""
Edit by Keisuke Oyamada @2018/04/19.
For training bachelor.
"""

import numpy as np
import chainer
import cupy
import argparse
import net
from chainer import optimizers
from chainer import serializers
from tqdm import tqdm
from chainer import functions as F
import math
import os


def main():
    parser = argparse.ArgumentParser(description="MNIST example")
    parser.add_argument("--epochs", "-e", help="number of epochs", default=20, type=int)
    parser.add_argument("--gpu", "-g", help="GPU ID (-1 : CPU)", default=-1, type=int)
    parser.add_argument("--batch_size", "-b", help="batch size", default=100, type=int)
    parser.add_argument("--save_path", "-s", help="data save path", required=True, type=str)

    args = parser.parse_args()

    epochs = args.epochs
    gpu = args.gpu
    batch_size = args.batch_size
    save_path = args.save_path

    print("MNIST example")
    print("epochs: {}".format(epochs))
    print("gpu ID: {}".format(gpu))
    print("batch size: {}".format(batch_size))
    print("save path: {}".format(save_path))

    mnist_train, mnist_test = chainer.datasets.get_mnist()
    x_train, y_train = mnist_train._datasets
    x_test, y_test = mnist_test._datasets

    N = len(x_train)
    N_test = len(x_test)

    print("train data size: {}".format(N))
    print("test data size: {}".format(N_test))

    model = net.sampleNet()

    if gpu >= 0:
        chainer.backends.cuda.get_device_from_id(gpu).use()
        model.to_gpu()

    xp = np if gpu < 0 else cupy

    opt = optimizers.Adam(1e-3)
    opt.setup(model)

    for epoch in range(1, epochs+1):
        perm = np.random.permutation(N)
        sum_loss = 0.
        sum_acc = 0.

        print("epoch: {}".format(epoch))
        bar = tqdm(desc="Training", total=math.ceil(N / batch_size), leave=False)

        for i in range(0, N, batch_size):
            x_v = chainer.Variable(xp.asarray(x_train[perm[i:i+batch_size]]))
            y_v = chainer.Variable(xp.asarray(y_train[perm[i:i+batch_size]]))

            y_h = model.forward(x_v)
            loss = F.softmax_cross_entropy(y_h, y_v)
            acc = F.accuracy(y_h, y_v)

            model.cleargrads()
            loss.backward()
            opt.update()

            loss.unchain_backward()
            model.cleargrads()

            sum_loss += float(loss.data) * len(x_v)
            sum_acc += float(acc.data) * len(x_v)
            bar.update()

        bar.close()


        print("train loss: {}".format(sum_loss / N), "train acc: {}".format(sum_acc / N))

        bar = tqdm(desc="Test", total=math.ceil(N_test / batch_size), leave=False)
        sum_loss_test = 0.
        sum_acc_test = 0.
        perm_test = np.random.permutation(N_test)
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            for i in range(0, N_test, batch_size):
                x_test_v = chainer.Variable(xp.asarray(x_test[perm_test[i:i+batch_size]]))
                y_test_v = chainer.Variable(xp.asarray(y_test[perm_test[i:i+batch_size]]))

                y_test_h = model.forward(x_test_v)
                loss_test = F.softmax_cross_entropy(y_test_h, y_test_v)
                acc_test = F.accuracy(y_test_h, y_test_v)
                sum_loss_test += float(loss_test.data) * len(x_test_v)
                sum_acc_test += float(acc_test.data) * len(x_test_v)
                bar.update()

        bar.close()
        print("test loss: {}".format(sum_loss_test / N_test),
              "test acc: {}".format(sum_acc_test / N_test))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    serializers.save_npz(os.path.join(save_path, "mnist_ex.model"), model)
    serializers.save_npz(os.path.join(save_path, "mnist_ex.optimizer"), opt)


if __name__ == "__main__":
    main()
