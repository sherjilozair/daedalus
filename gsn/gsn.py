import theano
import numpy
import time
import cPickle
import os
from PIL import Image
from theano import tensor as T
rng_numpy = numpy.random
from theano.tensor.shared_randomstreams import RandomStreams
rng_theano = RandomStreams()

dumpfile = 'models/model7.pkl'

def corrupt_SnP(x, corruption_level):
    #return x
    if corruption_level == 0.0:
        return x
    a = rng_theano.binomial(size=x.shape, n=1, p=1-corruption_level)
    b = rng_theano.binomial(size=x.shape, n=1, p=0.5)
    return x * a + T.eq(a, 0) * b

def corrupt_gaussian(x, corruption_level):
    #return x
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
        self.inputs = T.matrix()
        self.corrupted_input = corrupt_SnP(self.inputs, input_noise)
        self.network = HiddenLayers(num_units, hidden_noises, self.corrupted_input, acts)
        self.outputs = self.network.outputs
        self.params = self.network.params
        ce = -self.inputs*T.log(self.outputs)-(1-self.inputs)*T.log(1-self.outputs)
        diff = (self.inputs - self.outputs)**2
        self.cost = ce.mean()
        self.grads = T.grad(self.cost, self.params)
        self.updates = []
        self.learning_rate = T.scalar()

        for i in xrange(len(self.params)):
            self.updates.append((self.params[i], self.params[i] - self.learning_rate * self.grads[i]))

        #self.train_fn = theano.function(inputs = [self.inputs], updates=self.updates)

        self.train_fn = theano.function(inputs = [self.inputs, self.learning_rate], outputs=self.cost, updates=self.updates)

        self.sample_fn = theano.function(inputs = [self.corrupted_input], outputs=self.outputs)

        self.samples = (self.outputs > 0.5).astype('float32')
        self.step_fn = theano.function(inputs = [self.inputs], outputs=[self.corrupted_input, self.samples])

    def train(self, dataset, mbsz, epochs, lr):
        for e in xrange(epochs):
            rng_numpy.shuffle(dataset)
            t1 = time.clock()
            costs = map(lambda i: self.train_fn(dataset[i:i+mbsz], lr), xrange(0, len(dataset), mbsz))
            t2 = time.clock()
            print e, (t2 - t1), sum(costs) / float(len(costs)), lr
            if e % 10 == 0:
                self.dumpto(dumpfile)
            samples = self.sample(10, 10, 0, 1)
            draw_mnist(samples, 'samples/', 10, 10, e)
            lr *= 0.99

    def dumpto(self, fname):
        with open(fname, 'w') as f:
            cPickle.dump(self, f)

    def sample(self, num_samples, num_chains, burnin, interval):
        dataset = numpy.load('/data/lisa/data/mnist/mnist-python/train_data.npy')
        samples = numpy.zeros((num_samples * 2+2, num_chains, 784))
        #x = rng_numpy.binomial(size=(num_chains, 784), n=1, p=0.5).astype('float32')
        x = (dataset[:num_chains] > 0.5).astype('float32')
        for i in xrange(burnin):
            [cx, x] = map(lambda i: i.astype('float32'), self.step_fn(x))
        samples[0] = x
        for i in xrange(num_samples):
            for j in xrange(interval):
                [cx, x] = map(lambda i: i.astype('float32'), self.step_fn(x))
            samples[2*i+1] = cx
            samples[2*i+2] = x

        return samples


#def draw_mnist(samples, output_dir, (x, y)):
#    if not os.path.exists(output_dir):
#        os.makedirs(output_dir)
#    all = Image.new("RGB", (28*x, 28*y))
#    for i in xrange(len(samples)):
#        pic = samples[i].reshape(28, 28) * 255
#        im = Image.fromarray(pic.astype('uint8'))
#        all.paste(im, (28*(i//y), 28*(i%y)))
#    all.save(os.path.join(output_dir, "all.png"))


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
    input_noise = 0.3
    hidden_noises = [(0., 0.), (0., 0.), (0., 0.)]
    acts = ['tanh', 'tanh', 'sigmoid']
    learning_rate = 0.25
    gsn = GSN(num_units, input_noise, hidden_noises, acts)
    dataset = numpy.load('/data/lisa/data/mnist/mnist-python/train_data.npy')
    dataset = (dataset > 0.5).astype('float32')
    dataset = dataset[:50000]
    gsn.train(dataset, 100, 1000, learning_rate)
    gsn.dumpto(dumpfile)



if __name__ == '__main__':
    main()


