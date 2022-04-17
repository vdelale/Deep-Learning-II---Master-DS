import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1).reshape(-1, 1)


class RBM():

    def __init__(self, p, q):
        self.p = p
        self.q = q
        # Initialize weights randomly with normal distribution
        self.W = np.random.normal(loc=0, scale=0.01, size=(p, q))
        self.a = np.zeros(p)
        self.b = np.zeros(q)

    def entree_sortie_RBM(self, input, act='sigmoid'):
        if act == 'sigmoid':
            p_h = sigmoid(input@self.W + self.b)
        else:
            p_h = softmax(input@self.W + self.b)
        h = 1 * (np.random.rand(*p_h.shape) < p_h)
        return p_h, h

    def sortie_entree_RBM(self, output):
        p_v = sigmoid(output@self.W.T + self.a)
        v = 1 * (np.random.rand(*p_v.shape) < p_v)
        return p_v, v

    def train_RBM(self, data, epochs=10000, learning_rate=0.01, batch_size=50):
        n_samples = data.shape[0]
        loss = []

        for epoch in range(1, epochs + 1):
            # Shuffle the data
            np.random.shuffle(data)
            for batch in range(0, n_samples, batch_size):
                batch_indices = np.arange(batch,
                                          min(batch + batch_size, n_samples))
                x = data[batch_indices, :]
                v0 = x
                p_hv0, h0 = self.entree_sortie_RBM(v0)
                p_vh0, v1 = self.sortie_entree_RBM(h0)
                p_hv1, h1 = self.entree_sortie_RBM(v1)

                # Computing the gradient
                grad_a = np.mean(v0 - v1, axis=0)
                grad_b = np.mean(p_hv0 - p_hv1, axis=0)
                grad_W = v0.T@p_hv0 - v1.T@p_hv1

                # Updating the weights

                self.a += learning_rate * grad_a
                self.b += learning_rate * grad_b
                self.W += learning_rate * grad_W

            output, _ = self.entree_sortie_RBM(data)
            reconstructed_input, _ = self.sortie_entree_RBM(output)
            size = n_samples * self.p
            loss.append(np.sum((reconstructed_input - data)**2) / size)
            if not(epoch % 5) or epoch == 1:
                print(f'Epoch {epoch} out of {epochs}, loss: {loss[-1]}')

        return self, loss

    def generer_image_RBM(self, n_imgs, n_iter):
        fig = plt.figure()
        fig.patch.set_facecolor('black')
        for i in range(n_imgs):
            v = 1 * (np.random.rand(self.p) < 0.5)
            v = v.reshape(1, v.shape[0])
            for j in range(n_iter):
                _, h = self.entree_sortie_RBM(v)
                _, v = self.sortie_entree_RBM(h)
            plt.subplot(n_imgs // 5, 5, i + 1)
            plt.imshow(v.reshape(20, 16), cmap='gray')
            plt.axis('off')
        plt.show()
