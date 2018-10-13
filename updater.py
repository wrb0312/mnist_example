import chainer
import numpy as np
from tqdm import tqdm
import math
import os
from chainer import serializers
import chainer.functions as F


def up(model, opt, loss):
    model.cleargrads()
    loss.backward()
    opt.update()
    loss.unchain_backward()


class update:
    def __init__(self, **kwargs):
        self.model = kwargs.pop("model")
        self.opt = kwargs.pop("opt")
        self.epochs = kwargs.pop("epochs")
        self.batch_size = kwargs.pop("batch_size")
        self.save_path = kwargs.pop("save_path")
        self.x_train, self.x_test = kwargs.pop("x")
        self.y_train, self.y_test = kwargs.pop("y")
        self.sum_loss = 0.
        self.sum_acc = 0.
        self.sum_loss_test = 0.
        self.sum_acc_test = 0.
        self.xp = self.model.xp

    def __call__(self):
        N = len(self.x_train)
        N_test = len(self.x_test)
        for epoch in range(1, self.epochs+1):
            perm = np.random.permutation(N)

            ### train ###
            print("epoch: {}".format(epoch))
            bar = tqdm(desc="Training", total=math.ceil(N / self.batch_size), leave=False)
            self.forward(self.x_train, self.y_train, N, self.batch_size, perm, bar)
            print("train loss: {}".format(self.sum_loss / N), "train acc: {}".format(self.sum_acc / N))

            ### test ###
            bar_test = tqdm(desc="Test", total=math.ceil(N_test / self.batch_size), leave=False)
            perm_test = np.random.permutation(N_test)
            with chainer.using_config('train', False), chainer.no_backprop_mode():
                self.forward(self.x_test, self.y_test, N_test, self.batch_size, perm_test, bar_test)

            print("test loss: {}".format(self.sum_loss_test / N_test),
                  "test acc: {}".format(self.sum_acc_test / N_test))

            self.sum_loss = 0.
            self.sum_acc = 0.
            self.sum_loss_test = 0.
            self.sum_acc_test = 0.

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        serializers.save_npz(os.path.join(self.save_path, "mnist_ex.model"), self.model)
        serializers.save_npz(os.path.join(self.save_path, "mnist_ex.optimizer"), self.opt)

    def forward(self, x, y, N, batch_size, perm, bar):
        for i in range(0, N, batch_size):
            x_v = chainer.Variable(self.xp.asarray(x[perm[i:i+batch_size]]))
            y_v = chainer.Variable(self.xp.asarray(y[perm[i:i+batch_size]]))
            y_h = self.model.forward(x_v)
            loss = F.softmax_cross_entropy(y_h, y_v)
            acc = F.accuracy(y_h, y_v)
            up(self.model, self.opt, loss)

            if chainer.config.train:
                self.sum_loss += float(loss.data) * len(x_v)
                self.sum_acc += float(acc.data) * len(x_v)
            else:
                self.sum_loss_test += float(loss.data) * len(x_v)
                self.sum_acc_test += float(acc.data) * len(x_v)
            if bar is not None:
                bar.update()

        if bar is not None:
            bar.close()
