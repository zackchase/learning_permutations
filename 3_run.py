import numpy as np
from _3_mnist_cnn_FC import DigitCNN
import ml_datasets
from tqdm import tqdm

trX, trY, teX, teY = ml_datasets.mnist(onehot=True)

Q = np.random.permutation(np.eye(784))

##########################################
#   Permute the columns of the train and test matrices
##########################################

trX = np.dot(trX, Q)
teX = np.dot(teX, Q)

#trX = trX.reshape((-1, 1, 28, 28))
#teX = teX.reshape((-1, 1, 28, 28))


cnn = DigitCNN()

def train(iters=1, eta=.1):

    for i in xrange(iters):
        for start, end in tqdm(zip(range(0, len(trX), 128), range(128, len(trX), 128))):
            cost = cnn.train(trX[start:end], trY[start:end], eta=eta)
            #print cost 

        test_error = 1 - np.mean(np.argmax(teY, axis=1) == cnn.predict(teX))
        train_error = 1- np.mean(np.argmax(trY[:10000], axis=1) == cnn.predict(trX[:10000]))
        print ("test error: %s, train error %s" % (test_error, train_error))
