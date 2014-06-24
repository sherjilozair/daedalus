import theano
import numpy
from theano import tensor as T

from PIL import Image

import os
import sys
import gzip
import cPickle
import operator as op

sigm = T.nnet.sigmoid
tanh = T.tanh

def id(x):
	return x

theano.config.compute_test_value = 'raise'

from theano.sandbox.linalg.ops import det

class Layer():
    def __init__(self, n_in, n_out, act):
        self.act = act
        self.W = self.init_weight(n_in, n_out, act)
        self.b = self.init_bias(n_out)
        self.params = [self.W]

    def init_weight(self, n_in, n_out, act):
        a = numpy.sqrt(6. / (n_in + n_out))
        if act == sigm:
            a *= 4.
        W = numpy.random.uniform(size=(n_out, n_in), low=-a, high=a)
        Q, R = numpy.linalg.qr(W)
        eye = numpy.eye(n_out)
        return theano.shared(eye)

    def init_bias(self, n_out):
        return theano.shared(numpy.zeros(n_out,))

    def __call__(self, inp):
        #import ipdb; ipdb.set_trace()
        return self.act(T.dot(self.W, inp) + self.b)

class MLP():
    def __init__(self, n_in, n_out, hls, acts):
        self.layers = [Layer(*args) for args in zip([n_in]+hls, hls+[n_out], acts)]
        self.params = reduce(op.add, map(lambda l: l.params, self.layers))

    def __call__(self, inp):
        return reduce(lambda x, fn: fn(x), self.layers, inp)

class JacobianNets():
	def __init__(self, n_in, n_out, hls, acts):
		self.net = MLP(n_in, n_out, hls, acts)
		self.params = self.net.params
		self.X = T.vector('X')
		self.X.tag.test_value = numpy.random.uniform(size=(784, ), high=1, low=0.0).astype('float32')
		self.lr = T.scalar('lr')
		self.lr.tag.test_value = 0.25
		self.Z = self.net(self.X)
		self.W = self.net.layers[0].W
		#self.dtanh = 1 - self.Z**2
		self.J = self.W
		self.logpx = T.log(T.abs_(det(self.J)))
		self.grads = T.grad(self.logpx, self.net.params)
		self.updates = map(lambda (param, grad): (param, param - self.lr * grad), zip(self.params, self.grads))
		self.train_fn = theano.function([self.X, self.lr], self.logpx, updates=self.updates)

	def train(self, dataset, epochs, lr_init, lr_decay):
		lr = lr_init
		for epoch in xrange(epochs):
			logpxs = 0.0
			for i in xrange(dataset.shape[0]):
				logpx = self.train_fn(dataset[i], lr)
				#print epoch, i, logpx
				logpxs += logpx
			print epoch, logpxs / dataset.shape[0]
			lr *= lr_decay

if __name__ == '__main__':
	with gzip.open(os.environ['MNIST']) as f:
		dataset = cPickle.load(f)[0][0][:5]

	jnet = JacobianNets(784, 784, [], [id])
	jnet.train(dataset, 500, 10**2, 0.99)

# Z = tanh(x * W + b)					# b * dimZ
# Z + e2 = tanh(x * W + b + e1 * W)		# b * dimZ
# e2 = (1 - tanh2(W * x + b)) e1 * W	# e1: b * dimX
										# W: dimX * dimZ
										# dtanh: b * dimZ

