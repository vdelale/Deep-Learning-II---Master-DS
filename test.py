import matplotlib.pyplot as plt
from mlxtend.data import loadlocal_mnist
from principal_RBM_alpha import *
from principal_DBN_alpha import *
from principal_DNN_MNIST import *
from sklearn.preprocessing import OneHotEncoder
from utils import *


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


"""
neurons = [images[0, :].shape[0], 200, 150, 100]
dbn = DBN(neurons)
dbn, losses = dbn.pretrain_DBN(images, n_epoch=2000, lr_rate=0.001)
generated = dbn.generer_image_DBN(10, n_iter=2000)

for i, loss in enumerate(losses):
    plt.plot(loss, label=f'Reconstruction error layer {i+1}')
plt.legend()
plt.show()
"""

X_, y_ = loadlocal_mnist(images_path='data/train-images-idx3-ubyte',
                         labels_path='data/train-labels-idx1-ubyte')

X_train = X_[:10000]
y_train = y_[:10000]
X_test = X_[10000:12000]
y_test = y_[10000:12000]
X_train = np.where(X_train > 0, 1, 0)
X_test = np.where(X_test > 0, 1, 0)

oh = OneHotEncoder()
y_train = oh.fit_transform(y_train.reshape(-1,1)).toarray()
y_test = oh.fit_transform(y_test.reshape(-1,1)).toarray()

dnn = DNN([784, 100, 50, 10]) 

dnn, losses = dnn.pretrain_DBN(X_train, batch_size=32, n_epoch=50)
for i, loss in enumerate(losses):
    plt.plot(loss, label=f'Reconstruction error layer {i+1}')
plt.legend()
plt.show()
dnn, loss = dnn.retropropagation(data=X_train, labels=y_train, epochs=100,
                                 batch_size=32, lr_rate=0.01)
plt.plot(loss)
plt.show()

dnn.test_dnn(X_test, y_test)
