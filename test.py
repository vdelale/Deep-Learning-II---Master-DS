from models import *
from utils import *
import matplotlib.pyplot as plt

PATH = 'data/binaryalphadigs.mat'

# Hyper-parameters settings and training
p = 320  #(equals 20 * 16, the size of the image)
q = 250
rbm = RBM(p, q)

chars = 'defg'
images, indices = lire_alpha_digit(PATH, chars)
epochs = 500
learning_rate = 0.01
batch_size = 10

RBM_trained, loss = rbm.train_RBM(images, epochs, learning_rate, batch_size)

plt.plot(loss)
plt.show()

n_iter = 1000
n_imgs = 10
RBM_trained.generer_image_RBM(n_imgs, n_iter)