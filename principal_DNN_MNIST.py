from principal_DBN_alpha import *

def cross_entropy(y_hat, y):
    loss = []
    for k in range(y.shape[0]):
        loss.append(np.sum([-y[k,j]*np.log(y_hat[k,j]) for j in range(y.shape[1])]))
    return loss

class DNN(DBN):
    
    def __init__(self, neurons):
        super().__init__(neurons)
        
    def calcul_softmax(self, rbm, data):
        p_h, _ = rbm.entree_sortie_RBM(data, act='softmax')
        return p_h
        
    def entree_sortie_reseau(self, data):
        outputs = [data]
        for rbm in self.RBMs_list[:-1]:
            _ , data = rbm.entree_sortie_RBM(data)
            outputs.append(data)
        outputs.append(self.calcul_softmax(self.RBMs_list[-1], data))
        return outputs

    def retropropagation(self, data, labels, batch_size=50, epochs=10000, lr_rate=0.01, verbose = True):
        n_samples = data.shape[0]
        loss = []
        
        for i in range(1, epochs + 1):
            loss_batches = []
            for z in range(0, n_samples, batch_size):
                batch = data[z:min(n_samples, z + batch_size), :]
                batch_labels = labels[z:min(n_samples, z + batch_size), :]
                outputs = self.entree_sortie_reseau(batch)
                last_layer = self.RBMs_list[-1]
                c = (outputs[-1] - batch_labels)
                last_layer.W -= lr_rate * outputs[-2].T @ c / batch_size
                last_layer.b -= lr_rate * np.mean(c, axis = 0) / batch_size
                for idx, rbm in reversed(list(enumerate(self.RBMs_list[:-1]))):
                    c = c@self.RBMs_list[idx+1].W.T * outputs[idx+1] * (1-outputs[idx+1])
                    rbm.W -= lr_rate / batch_size * outputs[idx].T @ c 
                    rbm.b -= lr_rate * np.mean(c, axis=0) 
                loss_batches += cross_entropy(outputs[-1], batch_labels)
            loss.append(np.mean(loss_batches))
            if verbose:
                if not(i % 25) or i == 1:
                    print(f"Epoch {i} out of {epochs}. CELoss value is {loss[-1]}")
        return self, loss

    def test_dnn(self, data, labels, verbose = True):
        for rbm in self.RBMs_list[:-1]:
            _, data = rbm.entree_sortie_RBM(data)
        preds = np.argmax(self.calcul_softmax(self.RBMs_list[-1], data),
                          axis=1)
        good_labels = 0
        #print(preds)
        for idx, pred in enumerate(preds):
            if pred == np.argmax(labels[idx]):
                good_labels += 1
        print(good_labels,labels.shape[0])
        if verbose:
            print("The percentage of false labeled data is ",
              100*(labels.shape[0] - good_labels) / labels.shape[0])
        return 100*(labels.shape[0] - good_labels) / labels.shape[0]
    
    def get_pred(self, data):
        for rbm in self.RBMs_list[:-1]:
            _, data = rbm.entree_sortie_RBM(data)
        probs = self.calcul_softmax(self.RBMs_list[-1], data)
        return(probs)