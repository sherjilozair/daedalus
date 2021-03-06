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

class AE():
    def __init__(self, dim):
        [dimX, dimH, dimZ] = dim
        self.lvl = 0.5
        self.dimX = dimX
        self.dimH = dimH
        self.dimZ = dimZ
        self.encoder1 = MLP(dimX, dimH, [1200], [tanh, sigm])
        self.encoder2 = MLP(dimH, dimZ, [500], [tanh, sigm])
        self.decoder2 = MLP(dimZ, dimH, [500], [tanh, sigm])
        self.decoder1 = MLP(dimH, dimX, [1200], [tanh, sigm])

        #self.denoiserZ = MLP(dimZ, dimZ, [700], [tanh, sigm])

        self.params = self.encoder1.params + self.decoder1.params
        self.params += self.encoder2.params + self.decoder2.params
        self.x = T.fmatrix('x')
        self.z = T.fmatrix('z')
        self.lr = T.scalar('lr')
        h = self.encoder1(self.x)
        hn = self.add_noise(h, 0.97)
        self.rx = self.decoder1(hn)
        z = self.encoder2(h)

        #nz = self.add_noise(z, 0.1)
        #rz = self.denoiserZ(nz)
        #self.params += self.denoiserZ.params
        #self.cost3 = ce(rz, z).mean(axis=1).mean(axis=0)
        #gradsH = T.grad(self.cost3, self.denoiserZ.params)
        #updatesH = map(lambda (param, grad): (param, param - self.lr * grad), zip(self.denoiserZ.params, gradsH))

        zn = self.add_noise(z, 0.897)
        self.rh = self.decoder2(zn)
        #self.rz = self.encoder(self.decoder(self.z))
        self.cost1 = ce(self.rx, self.x).mean(axis=1).mean(axis=0)
        self.cost2 = ce(self.rh, h).mean(axis=1).mean(axis=0)
        self.cost = self.cost1 + 0.01 * self.cost2 #+ 0.005 * self.cost3
        self.grads = T.grad(self.cost, self.params)
        #self.grads2 = T.grad(self.cost1, self.decoder.params)
        #self.grads3 = T.grad(self.cost2, self.decoder.params)
        #self.grads4 = map(lambda (i, j): i + j, zip(self.grads2, self.grads3))
        #self.grads = self.grads1 + self.grads4
        self.updates = map(lambda (param, grad): (param, param - self.lr * grad), zip(self.params, self.grads))
        #self.updates += updatesH
        self.train_fn = theano.function([self.x, self.lr], [self.cost1, self.cost2], updates=self.updates, allow_input_downcast=True)
        z = T.fmatrix('z')
        self.decoder_fn = theano.function([z], self.decoder1(self.decoder2(z)), allow_input_downcast=True)
        self.encoder_fn = theano.function([self.x], self.encoder2(self.encoder1(self.x)), allow_input_downcast=True)
        #self.denoiser_fn = theano.function([z], self.denoiserZ(z), allow_input_downcast=True)

    def neglogprob(self, z):
        t2 = z.norm(2, axis=1)**2 / (2 * z.shape[1])
        return t2

    def train(self, D, epochs, mbsz, lr_init, lr_scale):
        ind = range(D.shape[0])
        lr = lr_init
        num_batches = D.shape[0] / mbsz
        t1 = time.clock()
        for e in xrange(epochs):
            #cPickle.dump(self, open("models/model_%s.pkl" % e, 'w'))
            random.shuffle(ind)
            cost = 0.0
            self.dump_samples(D, e)
            self.dump_recons(D, e)
            for b in xrange(num_batches):
                bs = D[ind[mbsz * b: mbsz * (b+1)]]
                z = numpy.random.normal(0, 1, (mbsz, self.dimZ)).astype('float32')
                cs = self.train_fn(bs, lr)
                cs = numpy.array(cs)
                cost += cs
                #print "minibatch", e, b, cs1, cs2, lr
            t2 = time.clock()
            duration = t2 - t1
            t1 = t2
            lr *= lr_scale
            print e, cost / num_batches, lr, duration

    def add_noise(self, z, corruption_level):
        return z + rng_theano.normal(size=z.shape, avg = 0.0, std = corruption_level)

    def add_noise_np(self, z, corruption_level):
        return z + numpy.random.normal(0, corruption_level, z.shape)

    noise = 0.1

    def dump_samples(self, D, name):
        C = 100
        T = 100
        samples = numpy.zeros((C * T, D.shape[1]))
        px = D[:C]
        for t in xrange(T):
            samples[C*t:C*(t+1)] = px
            z = self.encoder_fn(px)
            z = self.add_noise_np(z, 0.897)
            #z = self.denoiser_fn(z)
            px = self.decoder_fn(z)

        draw_mnist(samples, 'samples/', T, C, "%s_%s" % (name, self.noise))
        #self.noise += 0.1
        #if self.noise > 0.95:
        #    self.noise = 0.1

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
    all.save(os.path.join(output_dir, 'samples_%s.png' % name))


if __name__ == '__main__':
    model = AE([784, 400, 100])
    D = numpy.load("../mnist.npy")
    D = (D > 0.5).astype('float32')
    model.train(D, 200, 100, 1., 0.99)




# (2 * pi)^(-k/2) exp (-1/2 * x.T x)
# neg log of this
# k/2 * log (2 * pi) + 1/2 * z.T * z



