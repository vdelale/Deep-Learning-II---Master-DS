from principal_RBM_alpha import *
import matplotlib.pyplot as plt
import numpy as np


class DBN():

    def __init__(self, neurons):
        self.RBMs_list = []
        for i in range(len(neurons) - 1):
            self.RBMs_list.append(RBM(neurons[i], neurons[i+1]))

    def pretrain_DBN(self, data, n_epoch=10000, lr_rate=0.01, batch_size=50):
        losses = []
        for rbm in self.RBMs_list:
            rbm, loss = rbm.train_RBM(data, n_epoch, lr_rate, batch_size)
            _, data = rbm.entree_sortie_RBM(data)
            losses.append(loss)
        return self, losses

    def generer_image_DBN(self, n_imgs, n_iter):
        fig = plt.figure()
        fig.patch.set_facecolor('black')
        for i in range(n_imgs):
            data = 1 * (np.random.rand(self.RBMs_list[0].p) < 0.5)
            data = data.reshape(1, data.shape[0])
            for itr in range(n_iter):
                for rbm in self.RBMs_list:
                    _, data = rbm.entree_sortie_RBM(data)
                for rbm in reversed(self.RBMs_list):
                    _, data = rbm.sortie_entree_RBM(data)
            plt.subplot(n_imgs // 5, 5, i + 1)
            plt.imshow(data.reshape(20, 16), cmap='gray')
            plt.axis('off')
        plt.show()
