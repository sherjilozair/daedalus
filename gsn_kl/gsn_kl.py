import theano
import numpy
import time
import cPickle
import os
import gzip
from PIL import Image
from theano import tensor as T
rng_numpy = numpy.random
from theano.tensor.shared_randomstreams import RandomStreams
rng_theano = RandomStreams()

dumpfile = 'model.pkl'

with gzip.open(os.environ['MNIST']) as f:
    dataset = cPickle.load(f)
train = dataset[0]
trainX = train[0]
trainY = train[1]
trainX = (trainX > 0.5).astype('float32')
numpy.random.shuffle(trainX)
trainX = list(trainX)
trainY = list(trainY)
trainXY = zip(trainX, trainY)
trainXY.sort(key=lambda i: i[1])
#trainX = trainX[:1000]


def corrupt_SnP(x, corruption_level):
    if corruption_level == 0.0:
        return x
    a = rng_theano.binomial(size=x.shape, n=1, p=1-corruption_level)
    b = rng_theano.binomial(size=x.shape, n=1, p=0.5)
    return x * a + T.eq(a, 0) * b

def corrupt_gaussian(x, corruption_level):
    if corruption_level == 0.0:
        return x
    return x + rng_theano.normal(size=x.shape, avg = 0.0, std = corruption_level)

class HiddenLayer():
    def __init__(self, n_in, n_out, noise, inputs, act):
        self.n_in = n_in
        self.n_out = n_out
        self.inputs = inputs
        a = numpy.sqrt(6. / (n_in + n_out))
        if act == 'sigmoid':
            a *= 4.
        self.W = theano.shared(rng_numpy.uniform(size=(self.n_in, self.n_out), low=-a, high=a))
        self.b = theano.shared(numpy.zeros(self.n_out, ))
        activation = T.dot(self.inputs, self.W) + self.b
        activation_c = corrupt_gaussian(activation, noise[0])
        if act == 'sigmoid':
            postactivation = T.nnet.sigmoid(activation_c)
            #self.outputs = postactivation
        elif act == 'tanh':
            postactivation = T.tanh(activation_c)
        self.outputs = corrupt_gaussian(postactivation, noise[1])
        self.params = [self.W, self.b]

class HiddenLayers():
    def __init__(self, num_units, hidden_noises, inputs, acts):
        self.num_units = num_units
        self.inputs = inputs
        self.layers = []
        self.params = []
        inp = self.inputs
        for i in xrange(len(self.num_units)-1):
            self.layers.append(HiddenLayer(self.num_units[i], self.num_units[i+1], hidden_noises[i], inp, acts[i]))
            inp = self.layers[-1].outputs
            self.params += self.layers[-1].params
        self.outputs = self.layers[-1].outputs

class GSN():
    def __init__(self, num_units, input_noise, hidden_noises, acts):
        self.inputs_end = T.matrix()
        self.inputs_start = T.matrix()
        self.corrupted_input = corrupt_SnP(self.inputs_start, input_noise)
        self.network = HiddenLayers(num_units, hidden_noises, self.corrupted_input, acts)
        self.outputs = self.network.outputs
        self.params = self.network.params

        ce = - T.mean(self.inputs_end * T.log(self.outputs) + (1-self.inputs_end) * T.log(1-self.outputs), axis=1)
        #p = p.mean(axis=0)
        self.cost = ce.mean()
        self.grads = T.grad(self.cost, self.params)
        self.updates = []
        self.learning_rate = T.scalar()

        for i in xrange(len(self.params)):
            self.updates.append((self.params[i], self.params[i] - self.learning_rate * self.grads[i]))

        self.train_fn = theano.function(inputs = [self.inputs_start, self.inputs_end, self.learning_rate], outputs=self.cost, updates=self.updates)
        self.sample_fn = theano.function(inputs = [self.corrupted_input], outputs=self.outputs)
        self.samples = (self.outputs > 0.5).astype('float32')
        self.step_fn = theano.function(inputs = [self.inputs_start], outputs=[self.corrupted_input, self.samples])

    def train(self, trainXY, mbsz, epochs, lr):
        for e in xrange(epochs):
            cum_cost = 0.0
            count = 0
            rng_numpy.shuffle(trainXY)
            trainXY.sort(key=lambda i: i[1])
            t1 = time.clock()
            for i in xrange(0, len(trainXY), 2 * mbsz):
                d1 = numpy.array(map(lambda i: i[0], trainXY[i+0*mbsz:i+1*mbsz]))
                d2 = numpy.array(map(lambda i: i[0], trainXY[i+1*mbsz:i+2*mbsz]))
                #import ipdb; ipdb.set_trace()
                cost1 = self.train_fn(d1, d1, lr)
                cost2 = self.train_fn(d2, d2, lr)
                cum_cost += cost1 + cost2
                count+=2
                print e, i, cum_cost/float(count)
            t2 = time.clock()
            print e, (t2 - t1), cum_cost/float(count), lr
            samples = self.sample(10, 10, 0, 1)
            draw_mnist(samples, 'samples/', 10, 10, e)
            lr *= 0.99

    def dumpto(self, fname):
        with open(fname, 'w') as f:
            cPickle.dump(self, f)

    def sample(self, num_samples, num_chains, burnin, interval):
        samples = numpy.zeros((num_samples * 2+2, num_chains, 784))
        x = trainX[:num_chains]
        for i in xrange(burnin):
            [cx, x] = map(lambda i: i.astype('float32'), self.step_fn(x))
        samples[0] = x
        for i in xrange(num_samples):
            for j in xrange(interval):
                [cx, x] = map(lambda i: i.astype('float32'), self.step_fn(x))
            samples[2*i+1] = cx
            samples[2*i+2] = x

        return samples

def draw_mnist(samples, output_dir, num_samples, num_chains, name):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    all = Image.new("RGB", (28*num_samples, 28*num_chains))
    for i in xrange(num_samples):
        for j in xrange(num_chains):
            pic = (samples[i, j].reshape(28, 28)) * 255
            im = Image.fromarray(pic)
            all.paste(im, (28 * i, 28 * j))
    all.save(os.path.join(output_dir, 'samples_%d.png' % name))

def main():
    num_units = [784, 1200, 1200, 784]
    input_noise = 0.6
    hidden_noises = [(0., 0.), (0., 0.), (0., 0.)]
    acts = ['tanh', 'tanh', 'sigmoid']
    learning_rate = 0.25
    gsn = GSN(num_units, input_noise, hidden_noises, acts)
    gsn.train(trainXY, 50, 1000, learning_rate)
    gsn.dumpto(dumpfile)

if __name__ == '__main__':
    main()


