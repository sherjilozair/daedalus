import numpy
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

z1 = numpy.random.uniform(size=(1000, 5))
z2 = z1 + numpy.random.normal(0, 0.1, size=(1000, 5))


print uniformity(z1)
print uniformity(z2)
