import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
from lib import *

class DigitCNN():

    def __init__(self):


        X = T.ftensor4()
        Y = T.fmatrix()
        eta = T.scalar()
        drop_conv = T.scalar()
        drop_full = T.scalar()

        w = random_weights((32, 1, 3, 3))
        w2 = random_weights((64, 32, 3, 3))
        w3 = random_weights((128, 64, 3, 3))
        w4 = random_weights((128*3*3, 625))
        w_o = random_weights((625, 10))


        b_c1 = zeros(32)
        b_c2 = zeros(64)
        b_c3 = zeros(128)

        b_h1 = zeros(625)
        b_o  = zeros(10)

        l1_lin = conv2d(X, w, border_mode='full')+b_c1.dimshuffle('x', 0, 'x', 'x')
        l1a = rectify(l1_lin)
        l1 = max_pool_2d(l1a, (2, 2))
        l1 = dropout(l1, drop_conv)

        l2_lin = conv2d(l1, w2) + b_c2.dimshuffle('x', 0, 'x', 'x')
        l2a =  rectify(l2_lin)
        l2 = max_pool_2d(l2a, (2, 2))
        l2 = dropout(l2, drop_conv)

        l3_lin = conv2d(l2, w3) + b_c3.dimshuffle('x', 0, 'x', 'x')
        l3a = rectify(l3_lin)
        l3b = max_pool_2d(l3a, (2, 2))
        l3 = T.flatten(l3b, outdim=2)
        l3 = dropout(l3, drop_conv)

        l4_lin = T.dot(l3, w4) + b_h1
        l4 = rectify(l4_lin)
        l4 = dropout(l4, drop_full)

        yhat = softmax(T.dot(l4, w_o) + b_o )


        y_x = T.argmax(yhat, axis=1)


        cost = T.mean(T.nnet.categorical_crossentropy(yhat, Y))
        params = [w, w2, w3, w4, w_o,
                  b_c1, b_c2, b_c3, b_h1, b_o,
                ]
        caches = make_caches(params)

        updates = momentum(cost, params, caches, eta)

        self.shapes = theano.function(inputs=[X, drop_conv],outputs=[l1.shape, l2.shape, l3.shape], allow_input_downcast=True)

        self.train_func = theano.function(inputs=[X, Y, drop_conv, drop_full, eta], outputs=cost, updates=updates, allow_input_downcast=True)
        self.predict_func = theano.function(inputs=[X, drop_conv, drop_full], outputs=y_x, allow_input_downcast=True)

    def train(self, X, Y, drop_conv=.3, drop_full=.5, eta=.01):
        return self.train_func(X, Y, drop_conv, drop_full, eta)

    def predict(self, X):
        return self.predict_func(X, 0., 0.)



