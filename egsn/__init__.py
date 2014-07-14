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
ce = T.nnet.binary_crossentropy

from theano.tensor.shared_randomstreams import RandomStreams
rng_theano = RandomStreams()


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

class EGSN():
    def __init__(self, dimX, dimZ):
        self.dimX = dimX
        self.dimZ = dimZ
        self.encoder = MLP(dimX, dimZ, [1200], [tanh, tanh])
        self.encoder = MLP(dimZ, dimX, [1200], [tanh, tanh])
        x = T.fmatrix('x')
        lr = T.scalar('lr')
        rx = self.decoder(self.encoder(x))
        cost1 = ce(rx, x).mean(axis=1).mean(axis=0)
        z = self.encoder(x)
        nz = self.add_noise(z, 0.2)
        self.denoiser = MLP(dimZ, dimZ, [1200], [tanh, tanh])
        rz = self.denoiser(nz)
        cost2 = ce(rz, z).mean(axis=1).mean(axis=0)
        cost = cost1 + cost2
        self.params = self.encoder.params + self.decoder.params + self.denoiser.params
        grads = T.grad(cost, self.params)
        updates = map(lambda (param, grad): (param, param - lr * grad), zip(self.params, grads))
        self.train_fn = theano.function([x, lr], [cost1, cost2], updates=updates)

    def train(self, D, epochs, mbsz, lr_init, lr_scale):
        ind = range(D.shape[0])
        lr = lr_init
        num_batches = D.shape[0] / mbsz
        for e in xrange(epochs):
            cPickle.dump(self, open("models/model_%s.pkl" % e, 'w'))
            random.shuffle(ind)
            cost1 = 0.0
            cost2 = 0.0
            #self.dump_samples(D, e)
            #self.dump_recons(D, e)
            for b in xrange(num_batches):
                bs = D[ind[mbsz * b: mbsz * (b+1)]]
                [cs1, cs2] = self.train_fn(bs, lr)
                cost1 += cs1
                cost2 += cs2
                #print "f g minibatch", e, b, cs1, cs2, lr
            lr *= lr_scale
            print e, cost1 / num_batches, cost2 / num_batches, lr
    
    def add_noise(self, z, corruption_level):
        eturn x + rng_theano.normal(size=z.shape, avg = 0.0, std = corruption_level)

    def dump_samples(self, D, name):
        C = 10
        T = 10
        samples = numpy.zeros((C * T, D.shape[1]))
        #z = numpy.random.uniform(size=(C, self.dimZ))
        #px = self.sample_fn(z)
        #x = sample_bernoulli_fn(px)
        px = x = D[:C]
        for t in xrange(T):
            samples[C*t:C*(t+1)] = px
            z = self.infer_fn(x)
            zz = salt_and_pepper_fn(z, self.lvl)
            px = self.sample_fn(zz)
            x = sample_bernoulli_fn(px)
        draw_mnist(samples, 'samples/', T, C, name)

    def dump_recons(self, D, name):
        C = 10
        T = 10
        samples = numpy.zeros((C * T, D.shape[1]))
        for t in xrange(T):
            z = self.infer_fn(D[C*t:C*(t+1)])
            px = self.sample_fn(z)
            samples[C*t:C*(t+1)] = px
        draw_mnist(samples, 'recons/', T, C, name)

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

    model = EGSN(784, 100)
    D = numpy.load("../mnist.npy")
    D = (D > 0.5).astype('float32')
    #D = D - numpy.mean(D, axis=0)
    #D = D / numpy.max(abs(D), axis=0)
    #D = (D + 1.) / 2.
    #D = D.astype('float32')
    model.train(D, 100, 100, .5, 0.99)








