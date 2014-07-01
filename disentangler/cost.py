import numpy
import theano
from theano import tensor as T
# z is shape (n, d)
def uniformity(z):
	n = z.shape[0]
	d = z.shape[1]
	t1 = (4./3.)**d
	t2 = -2. * (1. + 2. * z - 2. * z**2).prod(axis=1).sum(axis=0) / n
	t4 = (1 - abs(z[:, numpy.newaxis] - z)).prod(axis=2).sum()
	t3 = (2.**d) * (t4) / (n**2)
	t5 = t1 - 2. + (2.**d)
	return (t1 + t2 + t3)/t5

sigmoid = lambda x: 1. / (1. + numpy.exp(-x))
sigmoid = lambda x: numpy.exp(x) / (1. + numpy.exp(x))

z1 = numpy.random.uniform(size=(1000, 50))
z2 = z1 + numpy.random.normal(0, 0.005, size=z1.shape)


#print uniformity(z1)
#print uniformity(z2)



## distance to boundary: http://www.uam.es/personal_pdi/ciencias/joser/articulos/2006-cjs-pre.pdf

def dtb(z):
	g = T.switch(z < 0.5, z, 1. - z).min(axis=1)
	r = 0.5
	return g/r

z = T.fmatrix('z')
dtb_fn = theano.function([z], dtb(z), allow_input_downcast=True)
print uniformity(dtb_fn(z1)[:, numpy.newaxis])
print uniformity(dtb_fn(z2)[:, numpy.newaxis])
z3 = 0.5 * numpy.ones_like(z2)
print uniformity(dtb_fn(z3)[:, numpy.newaxis]/2.)
