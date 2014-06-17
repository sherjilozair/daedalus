# times is shape (n+1, n+1).
# times[i][j] gives time from city i to city j
# n end cities, 1 start city.
# return K lists for K paths to take to optimize total time

import random, numpy

def solveMTSP(times, N, K, max_iter=200, pop_size=300):
	population = initialize_population(pop_size, N, K)		# initialize randomly
	for iter in xrange(max_iter):
		parents = selection(population, times)
		population = mutate(parents)

def initialize_population(pop_size, N, K):
	return [initialize_chromosome(N, K) for i in xrange(pop_size)]

def initialize_chromosome(N, K):
	respos = [(i+1, random.randrange(K)) for i in xrange(N)]
	random.shuffle(respos)										# is this best way?
	return map(lambda k: map(lambda i: i[0], filter(lambda i: i[1] == k, respos)), xrange(K))

def compute_fitness_all(population, times):
	return map(lambda chromo: comptue_fitness(chromo, times), population)

def comptue_fitness(chromo, times):
	return sum(compute_time(r, times) for r in chromo)

def compute_time(route, times):
	return sum(times[i][j] for i, j in zip([0]+route, route+[0]))

def selection(population, times):
	fitnesses = compute_fitness_all(population, times)
	population_fitness = zip(population, fitnesses)
	tourney = [population_fitness[i:i+6] for i in xrange(0, len(population), 6)]
	return map(lambda t: min(t, key=lambda i: i[1])[0], tourney)

def inversion(chromo):
	chromo = map(list, chromo)
	route = random.choice(chromo)
	s1 = random.randrange(len(route))
	s2 = random.randrange(len(route))
	s1, s2 = min(s1, s2), max(s1, s2)
	route[s1:s2] = reversed(route[s1:s2])
	return chromo

def transposition(chromo):
	chromo = map(list, chromo)
	route = random.choice(chromo)
	s1 = random.randrange(len(route))
	s2 = random.randrange(len(route))
	route[s1], route[s2] = route[s2], route[s1]
	return chromo

def insertion(chromo):
	chromo = map(list, chromo)
	route = random.choice(chromo)
	s1 = random.randrange(len(route))
	s2 = random.randrange(len(route))
	route[s1], route[s2] = route[s2], route[s1] # change this for insertion.
	return chromo


def applyallmutations(chromo):
	return map(lambda f: f(chromo), [identity, inversion, transposition, insertion, cross_transposition, contraction])

def mutate(parents):
	return map(applyallmutations, parents)


def test():
	a = numpy.random.randint(size=(31, 31), low=2, high=100)
	b = abs((a - a.T)/2)
	times = b.tolist()
	population = initialize_population(300, 30, 5)
	return selection(population, times)

