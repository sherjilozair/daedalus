import theano
import numpy
from theano import tensor as T
theano.config.compute_test_value = 'raise'

from PIL import Image

import os
import sys
import gzip
import cPickle
import operator as op


sigm = T.nnet.sigmoid
tanh = T.tanh

class Layer():
    def __init__(self, n_in, n_out, act):
        self.act = act
        self.W = self.init_weight(n_in, n_out, act)
        self.b = self.init_bias(n_out)
        self.params = [self.W, self.b]

    def init_weight(self, n_in, n_out, act):
        a = numpy.sqrt(6. / (n_in + n_out))
        if act == sigm:
            a *= 4.
        return theano.shared(numpy.random.uniform(size=(n_in, n_out), low=-a, high=a))

    def init_bias(self, n_out):
        return theano.shared(numpy.zeros(n_out,))

    def __call__(self, inp):
        #import ipdb; ipdb.set_trace()
        return self.act(T.dot(inp, self.W) + self.b)

class MLP():
    def __init__(self, n_in, n_out, hls, acts):
        self.layers = [Layer(*args) for args in zip([n_in]+hls, hls+[n_out], acts)]
        self.params = reduce(op.add, map(lambda l: l.params, self.layers))

    def __call__(self, inp):
        return reduce(lambda x, fn: fn(x), self.layers, inp)


class SSM():
    def __init__(self, dimX, dimZ, dimS, ghls, gacts, shls, sacts):
        self.dimZ = dimZ

        self.inputX = T.matrix()
        self.inputX.tag.test_value = numpy.random.randn(100, dimX).astype('float32')
        self.inputZ = T.matrix()
        self.inputZ.tag.test_value = numpy.random.randn(100, dimZ).astype('float32')
        self.lr = T.scalar()
        self.lr.tag.test_value = 0.25
        self.generator = MLP(dimZ, dimX, ghls, gacts)
        self.statistifier = MLP(dimX, dimS, shls, sacts)
        self.minparams = self.generator.params
        self.maxparams = self.statistifier.params
        self.generatedX = self.generator(self.inputZ)
        self.statsG = self.statistifier(self.generatedX)
        self.statsD = self.statistifier(self.inputX)
        self.EstatsG = self.statsG.mean(axis=0)
        self.EstatsD = self.statsD.mean(axis=0)
        self.diff = mse(self.EstatsG, self.EstatsD)
        self.cost = self.diff.mean(axis=0)
        self.mingrads = T.grad(self.cost, self.minparams)
        self.maxgrads = T.grad(self.cost, self.maxparams)
        self.minupdates = map(lambda (param, grad): (param, param - self.lr * grad), zip(self.minparams, self.mingrads))
        self.maxupdates = map(lambda (param, grad): (param, param + self.lr * grad), zip(self.maxparams, self.maxgrads))
        self.train_fn = theano.function([self.inputZ, self.inputX, self.lr], self.cost, updates = self.minupdates + self.maxupdates)
        self.sample_fn = theano.function([self.inputZ], self.generatedX)

    def sample(self, n):
        z = numpy.random.uniform(size=(n, self.dimZ), low=-numpy.sqrt(3), high=numpy.sqrt(3)).astype('float32')
        return self.sample_fn(z)

    def train(self, dataset, epochs, batch_size, lr_init, lr_decay):
        num_batches = dataset.shape[0] / batch_size
        lr = lr_init
        for epoch in xrange(epochs):
            sum_cost = 0.0
            numpy.random.shuffle(dataset)
            samples = self.sample(100)
            draw_mnist(samples, 'samples/', 10, 10, epoch)
            for i in xrange(num_batches):
                x = dataset[batch_size * i: batch_size * (i+1)]
                z = numpy.random.uniform(size=(batch_size, self.dimZ), low=-numpy.sqrt(3), high=numpy.sqrt(3)).astype('float32')
                cost = self.train_fn(z, x, lr)
                sum_cost += cost
                print epoch, cost
            lr *= lr_decay
            mean_cost = sum_cost / num_batches
            print epoch, mean_cost, lr

def mse(a, b):
    return (a-b)**2

def draw_mnist(samples, output_dir, x, y, name):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    all = Image.new("RGB", (28*x, 28*y))
    for i in xrange(x):
        for j in xrange(y):
            pic = (samples[i*x + j].reshape(28, 28)) * 255
            im = Image.fromarray(pic)
            all.paste(im, (28 * i, 28 * j))
    all.save(os.path.join(output_dir, 'samples_%d.png' % name))



if __name__ == '__main__':
    ssm = SSM(784, 100, 3000, [1200], [tanh, sigm], [2000], [tanh, tanh])
    with gzip.open('/data/lisa/data/mnist/mnist.pkl.gz') as f:
    #with gzip.open('/Users/sherjilozair/Desktop/bengio/DeepLearningTutorials/data/mnist.pkl.gz') as f:
        dataset = cPickle.load(f)[0][0]
    ssm.train(dataset, 500, 100, 0.25, 0.99)


