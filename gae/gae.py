import theano
import numpy
from theano import tensor as T
import time
from PIL import Image

import os
import sys
import gzip
import cPickle
import operator as op

import random
from theano.scalar.basic import UnaryScalarOp, same_out_nocomplex
from theano.tensor.shared_randomstreams import RandomStreams
rng_theano = RandomStreams()
rng_numpy = numpy.random

sigm = T.nnet.sigmoid
tanh = T.tanh
relu = lambda x: T.maximum(x, 0)
ce = lambda x, y: T.nnet.binary_crossentropy(x, y)

NOISE = 0.0
DIMZ = 200

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
        return self.act(T.dot(inp, self.W) + self.b)

class MLP():
    def __init__(self, n_in, n_out, hls, acts):
        self.layers = [Layer(*args) for args in zip([n_in]+hls, hls+[n_out], acts)]
        self.params = reduce(op.add, map(lambda l: l.params, self.layers))

    def __call__(self, inp):
        return reduce(lambda x, fn: fn(x), self.layers, inp)


class Multinomial():
    def __init__(self, dim):
        self.dim = dim
        self.W = theano.shared(numpy.random.normal(size=(dim, ), scale=0.1))
        self.params = [self.W]

    def get_cost(self, x):
        return ce(sigm(self.W), x)

    def sample(self, n):
        w = sigm(self.W).eval()
        return rng_numpy.binomial(n=1, p=w, size=(n, self.dim)).astype('float32')

    def loglikelihood(self, x):
        return self.get_cost(x).sum(axis=1).eval()


def corrupt_SnP(x, corruption_level):
    if corruption_level == 0.0:
        return x
    a = rng_theano.binomial(size=x.shape, n=1, p=1-corruption_level)
    b = rng_theano.binomial(size=x.shape, n=1, p=0.5)
    #b = rng_theano.uniform(size=x.shape, low=0.0, high=1.0)
    return x * a + T.eq(a, 0) * b

class GAE():
    def __init__(self, dimX, dimH, sizes, acts, name, old_models):
        self.name = name
        self.alpha = T.scalar('alpha')
        self.beta = T.scalar('beta')
        self.old_models = old_models
        self.dimX = dimX
        self.encoder = MLP(dimX, dimH, sizes, acts)
        self.decoder = MLP(dimH, dimX, sizes, acts)
        self.toplayer = Multinomial(dimH)
        self.X = T.fmatrix('X')
        self.lr = T.scalar('lr')
        self.H = self.encoder(self.X)
        self.H2 = (T.ident(2.*self.H-1.)+1.)/2.

        self.RX = self.decoder(corrupt_SnP(self.H2, 0.01))
        cost_recons = ce(self.RX, self.X).mean(axis=1).mean(axis=0)
        cost_prior = self.toplayer.get_cost(self.H2).mean(axis=1).mean(axis=0)

        #cost_derivative = ((self.H - self.H**2).sum(axis=1)/784).mean(axis=0)
        cost = self.alpha * cost_recons + self.beta * cost_prior # + cost_derivative
        params = self.encoder.params + self.decoder.params + self.toplayer.params
        grads = T.grad(cost, params)
        updates = map(lambda (param, grad): (param, param - self.lr * grad), zip(params, grads))
        self.train_fn = theano.function([self.X, self.lr, self.alpha, self.beta], [cost_recons, cost_prior], updates=updates, allow_input_downcast=True)
        self.encode_fn = theano.function([self.X], self.H, allow_input_downcast=True)
        hup = T.fmatrix('hup')
        hdown1 = self.decoder(hup)
        hdown2 = (T.ident(2.*hdown1-1.)+1.)/2.
        self.decode_fn = theano.function([hup], hdown1, allow_input_downcast=True)
        self.sign_fn = theano.function([hdown1], hdown2, allow_input_downcast=True)

    def train(self, dataset, epochs, mbsz, lr, lr_scale):
        f = open('alpha')
        [alpha, beta] = map(float, f.read().split(','))
        f.close()
        for e in xrange(epochs):
            rng_numpy.shuffle(dataset)
            t1 = time.clock()
            costs = 0. 
            for i in xrange(0, len(dataset), mbsz):                
                cost = self.train_fn(dataset[i:i+mbsz], lr, alpha, beta)
                cost = numpy.array(cost)
                #print cost
                costs += cost
            t2 = time.clock()
            self.dump_recons(dataset, 15, 15, e)
            self.dump_samples(15, 15, e)
            costs = costs / float(len(range(0, len(dataset), mbsz)))
            print e, (t2 - t1), costs, costs.sum(), lr, alpha, beta
            f = open('alpha')
            [alpha, beta] = map(float, f.read().split(','))
            f.close()
            #lr *= lr_scale

    def loglikelihood(self, X):
        H = X
        for i in xrange(len(self.old_models)):
            mod = self.old_models[i]
            H = mod.encode_fn(H)
            H = mod.sign_fn(H)
        H = self.encode_fn(H)
        H2 = self.sign_fn(H)
        p1 = self.toplayer.loglikelihood(H2)
        M = self.decode_fn(H2)
        for i in reversed(xrange(len(self.old_models))):
            mod = self.old_models[i]
            M = mod.sign_fn(M)
            M = mod.decode_fn(M)
        p2 = ce(M, X).sum(axis=1).eval()
        return p1, p2
        
    def dump_recons(self, D, x, y, name):
        h = self.encode_fn(D[:x*y])
        h = self.sign_fn(h)
        m = self.decode_fn(h)
        for i in reversed(xrange(len(self.old_models))):
            mod = self.old_models[i]
            m = self.sign_fn(m)
            m = mod.decode_fn(m)
        draw_mnist(m, 'recons_%d/' % self.name, x, y, name)

    def dump_samples(self, x, y, name):
        h = self.toplayer.sample(x*y)
        m = self.decode_fn(h)
        for i in reversed(xrange(len(self.old_models))):
            mod = self.old_models[i]
            m = self.sign_fn(m)
            m = mod.decode_fn(m)

        draw_mnist(m, 'samples_%d/' % self.name, x, y, name)

def draw_mnist(samples, output_dir, num_samples, num_chains, name):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    all = Image.new("RGB", (28*num_samples, 28*num_chains))
    for i in xrange(num_samples):
        for j in xrange(num_chains):
            pic = samples[i*num_chains+j].reshape(28, 28) * 255
            im = Image.fromarray(pic)
            all.paste(im, (28*i, 28*j))
    all.save(os.path.join(output_dir, 'samples_%d.png' % name))


if __name__ == '__main__':
    D = numpy.load('../mnist.npy')
    model1 = GAE(784, 300, [1200, 1200, 300, 1200, 1200, 300, 1200, 1200], [relu, relu, relu, relu, relu, relu, relu, relu, sigm], 3, [])
    model1.train(D, 2000, 100, 1., .99)
    #H2 = model2.encode_fn(H1)
    #H2 = model2.sign_fn(H2)
    #model2 = GAE(200, 200, [500], [tanh, sigm], 2, [model1])
    #model2.train(H1, 100, 100, 1., .99)
    #H2 = model2.encode_fn(H1)
    #H2 = model2.sign_fn(H2)
    #model_fine_tune = GAEFineTuner(model1, model2)
    #model3 = GAE(200, 200, [500, 500], [tanh, tanh, sigm], 3, [model1, model2])
    #model3.train(H2, 500, 100, 1., .995)
    #H3 = model3.encode_fn(H2).astype('float32')
    





