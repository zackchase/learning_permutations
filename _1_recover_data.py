
import numpy
import theano
from theano import tensor as T
from lib import *
import ml_datasets
from tqdm import tqdm

################################
#  input is a batch of examples (for MNIST has shape(num_examples, 784))
################################
X               = T.matrix()
Y               = T.matrix()

row_penalty     = T.scalar()
col_penalty     = T.scalar()

learning_rate   = T.scalar()
num_features    = 784

########################################
#  consider smarter initialization with nonegative weights and withing the feasible space
########################################
W = positive_weights((num_features, num_features))

ones = theano.shared(np.ones(num_features))

row_diff    = T.dot(W,ones) - ones 
col_diff    = T.dot(W.transpose(), ones) - ones
row_norm    = T.mean((row_diff ** 2))
col_norm    = T.mean((col_diff ** 2))

Yhat        = T.dot(X, W)

penalty     =  row_penalty * row_norm + col_penalty * col_norm
recon_error = T.mean((Yhat - Y) ** 2) 

cost        = recon_error + penalty 

params      = [W]
updates     = SGD(cost, params, learning_rate)

train_func  = theano.function(inputs=[X,Y, row_penalty, col_penalty, learning_rate], outputs=[recon_error, penalty, cost], updates=updates)




########################################
#  Code to run experiment
########################################
Xtrain, Ytrain, Xtest, Ytest = ml_datasets.mnist()
def train(iters=1, row_penalty=0.0, col_penalty=0.0, eta=.1):
    for i in xrange(iters):
        for start, end in tqdm(zip(range(0, len(Xtrain), 128), range(128, len(Xtrain), 128))):
            recon_error, penalty, cost = train_func(Xtrain[start:end], Xtrain[start:end], row_penalty, col_penalty, eta)
            print ("Recon: %s, Penalty: %s, Cost: %s" % (recon_error, penalty, cost))
               
              
               














