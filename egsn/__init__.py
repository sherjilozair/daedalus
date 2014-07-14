import theano
import numpy
from theano import tensor as T
import random

from PIL import Image

import os
import sys
import gzip
import cPickle
import operator as op
import time

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
        self.lvl = 0.5
        self.dimX = dimX
        self.dimZ = dimZ
        self.encoder = MLP(dimX, dimZ, [1200], [tanh, sigm])
        self.decoder = MLP(dimZ, dimX, [1200], [tanh, sigm])
        x = T.fmatrix('x')
        lr = T.scalar('lr')
        alpha = T.scalar('alpha')
        rx = self.decoder(self.encoder(x))
        cost1 = ce(rx, x).mean(axis=1).mean(axis=0)
        z = self.encoder(x)
        nz = self.add_noise(z, self.lvl)
        self.denoiser = MLP(dimZ, dimZ, [1200], [tanh, sigm])
        rz = self.denoiser(nz)
        cost2 = ce(rz, z).mean(axis=1).mean(axis=0)
        #cost = cost1 + alpha * cost2
        self.ae_params = self.encoder.params + self.decoder.params
        self.gsn_params = self.denoiser.params
        grads_ae = T.grad(cost1, self.ae_params)
        grads_gsn = T.grad(cost2, self.gsn_params)
        updates_ae = map(lambda (param, grad): (param, param - lr * grad), zip(self.ae_params, grads_ae))
        updates_gsn = map(lambda (param, grad): (param, param - lr * grad), zip(self.gsn_params, grads_gsn))
        updates = updates_ae + updates_gsn
        self.train_fn = theano.function([x, lr], [cost1, cost2], updates=updates, allow_input_downcast=True)
        self.encoder_fn = theano.function([x], self.encoder(x), allow_input_downcast=True)
        z = T.fmatrix('z')
        self.decoder_fn = theano.function([z], self.decoder(z), allow_input_downcast=True)
        nz = T.fmatrix('nz')
        self.denoiser_fn = theano.function([nz], self.denoiser(nz), allow_input_downcast=True)

    def train(self, D, epochs, mbsz, lr_init, lr_scale):
        ind = range(D.shape[0])
        lr = lr_init
        num_batches = D.shape[0] / mbsz
        t1 = time.clock()
        for e in xrange(epochs):
            cPickle.dump(self, open("models/model_%s.pkl" % e, 'w'))
            random.shuffle(ind)
            cost1 = 0.0
            cost2 = 0.0
            self.dump_samples(D, e)
            self.dump_recons(D, e)
            for b in xrange(num_batches):
                bs = D[ind[mbsz * b: mbsz * (b+1)]]
                [cs1, cs2] = self.train_fn(bs, lr)
                cost1 += cs1
                cost2 += cs2
                #print "minibatch", e, b, cs1, cs2, lr
            t2 = time.clock()
            duration = t2 - t1
            t1 = t2
            lr *= lr_scale
            print e, cost1 / num_batches, cost2 / num_batches, lr, duration
    
    def add_noise(self, z, corruption_level):
        return z + rng_theano.normal(size=z.shape, avg = 0.0, std = corruption_level)

    def add_noise_np(self, z, corruption_level):
        return z + numpy.random.normal(0, corruption_level, z.shape)

    def dump_samples(self, D, name):
        C = 10
        T = 20
        samples = numpy.zeros((C * T, D.shape[1]))
        px = x = D[:C]
        z = self.encoder_fn(x)
        for t in xrange(0, T, 2):
            samples[C*t:C*(t+1)] = px
            zz = self.add_noise_np(z, self.lvl)
            pxx = self.decoder_fn(zz)
            samples[C*(t+1):C*(t+2)] = pxx
            z = self.denoiser_fn(zz)
            px = self.decoder_fn(z)
        draw_mnist(samples, 'samples/', T, C, name)

    def dump_recons(self, D, name):
        C = 10
        T = 10
        samples = numpy.zeros((C * T, D.shape[1]))
        for t in xrange(T):
            z = self.encoder_fn(D[C*t:C*(t+1)])
            px = self.decoder_fn(z)
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

    model = EGSN(784, 784/2)
    D = numpy.load("../mnist.npy")
    D = (D > 0.5).astype('float32')
    #D = D - numpy.mean(D, axis=0)
    #D = D / numpy.max(abs(D), axis=0)
    #D = (D + 1.) / 2.
    #D = D.astype('float32')
    model.train(D, 100, 100, 0.25, 0.99)








