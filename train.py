"""
Edit by Keisuke Oyamada @2018/04/19.
For training bachelor.
"""

import chainer
import argparse
import net
from chainer import optimizers
from updater import update


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

    opt = optimizers.Adam(1e-3)
    opt.setup(model)

    updater = update(**{
        "model": model,
        "opt": opt,
        "epochs": epochs,
        "batch_size": batch_size,
        "save_path": save_path,
        "x": (x_train, x_test),
        "y": [y_train, y_test],
        })

    updater()


if __name__ == "__main__":
    main()
