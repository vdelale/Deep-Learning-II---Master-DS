from cProfile import label
from principal_RBM_alpha import *
from principal_DBN_alpha import *
from utils import *
import matplotlib.pyplot as plt

PATH = 'data/binaryalphadigs.mat'

# Hyper-parameters settings and training
p = 320  # (equals 20 * 16, the size of the image)
q = 250
rbm = RBM(p, q)

chars = 'defg'
images, indices = lire_alpha_digit(PATH, chars)
epochs = 500
learning_rate = 0.01
batch_size = 10

# RBM_trained, loss = rbm.train_RBM(images, epochs, learning_rate, batch_size)

# plt.plot(loss)
# plt.show()

n_iter = 1000
n_imgs = 10
# RBM_trained.generer_image_RBM(n_imgs, n_iter)

neurons = [images[0, :].shape[0], 200, 150, 100]
dbn = DBN(neurons)
dbn, losses = dbn.pretrain_DBN(images, n_epoch=2000, lr_rate=0.001)
generated = dbn.generer_image_DBN(10, n_iter=2000)

for i, loss in enumerate(losses):
    plt.plot(loss, label=f'Reconstruction error layer {i+1}')
plt.legend()
plt.show()
