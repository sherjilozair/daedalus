import theano as th
import numpy as np
from theano import tensor as T

class Elman:
    def __init__(self, lenW, dimW, dimS):
        self.W = th.shared(np.random.randn(lenW, dimW))
        self.Uw = th.shared(np.random.randn(dimW, dimS))
        self.Us = th.shared(np.random.randn(dimS, dimS))
        self.V = th.shared(np.random.randn(dimS, lenW))
        self.S0 = th.shared(np.random.randn(dimS,))
        self.idx = T.icol()
        self.w = self.W[self.idx].reshape((self.idx.shape[0], self.W.shape[1]))
        def recurrence(w, s):
            # import ipdb; ipdb.set_trace()
            s1 = T.nnet.sigmoid(T.dot(w, self.Uw))
            s2 = T.nnet.sigmoid(T.dot(s, self.Us))
            ss = s1 + s2
            pp = T.dot(s, self.V)
            return [ss, pp]
        [self.S, self.PP], _ = th.scan(fn=recurrence, sequences=self.w, outputs_info=[self.S0, None], n_steps=self.w.shape[0])
        self.P = T.nnet.softmax(self.PP)
        self.RP = self.P[T.arange(self.w.shape[0]), self.idx[:,0]]
        self.cost = -T.sum(T.log(self.RP))
        self.params = [self.W, self.Uw, self.Us, self.V, self.S0]
        self.grads = T.grad(self.cost, self.params)
        self.lr = T.scalar()
        self.updates = map(lambda (param, grad): (param, param - self.lr * grad), zip(self.params, self.grads))
        self.train_fn = th.function([self.idx, self.lr], [self.cost], updates=self.updates, allow_input_downcast=True)
        self.fprop = th.function([self.idx], [self.S, self.P, self.cost], allow_input_downcast=True)

nn = Elman(10, 15, 20)
idx = np.array([0, 1, 2, 0, 3, 2, 0, 1]).reshape(-1, 1)

for i in xrange(1000):
    nn.train_fn(idx, 0.01)
    S, P, C = nn.fprop(idx)
    print C, 

