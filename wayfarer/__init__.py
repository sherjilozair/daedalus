# times is shape (n+1, n+1).
# times[i][j] gives time from city i to city j
# n end cities, 1 start city.
# return K lists for K paths to take to optimize total time

import random, numpy
import urllib, json, csv, os
import cPickle

def solveMTSP(times, K, max_iter=10000, pop_size=500):
	N = len(times) - 1
	population = initialize_population(pop_size, N, K)		# initialize randomly
	prev_cost = -1
	for iter in xrange(max_iter):
		parents = selection(population, times)
		population = mutate(parents)
		fitnesses = map(lambda c: comptue_fitness1(c, times), population)
		cost = min(fitnesses)
		if cost != prev_cost:
			prev_cost = cost
			print iter, cost/60.0
	return min(population, key=lambda c: comptue_fitness2(c, times))

def initialize_population(pop_size, N, K):
	return [initialize_chromosome(N, K) for i in xrange(pop_size)]

def initialize_chromosome(N, K):
	respos = [(i+1, random.randrange(K)) for i in xrange(N)]
	random.shuffle(respos)										# is this best way?
	return map(lambda k: map(lambda i: i[0], filter(lambda i: i[1] == k, respos)), xrange(K))

def compute_fitness_all(population, times):
	return map(lambda chromo: comptue_fitness1(chromo, times), population)

def comptue_fitness1(chromo, times):
	return sum(compute_time(r, times) for r in chromo)

def comptue_fitness2(chromo, times):
	return max(compute_time(r, times) for r in chromo)

def compute_time(route, times):
	#print route
	return sum(times[i][j] for i, j in zip([0]+route, route+[0]))

def selection(population, times):
	fitnesses = compute_fitness_all(population, times)
	population_fitness = zip(population, fitnesses)
	tourney = [population_fitness[i:i+5] for i in xrange(0, len(population), 5)]
	return map(lambda t: min(t, key=lambda i: i[1])[0], tourney)

def inversion(chromo):
	chromo = map(list, chromo)
	route = random.choice(chromo)
	if not route:
		return chromo
	s1 = random.randrange(len(route))
	s2 = random.randrange(len(route))
	s1, s2 = min(s1, s2), max(s1, s2)
	route[s1:s2] = reversed(route[s1:s2])
	return chromo

def transposition(chromo):
	chromo = map(list, chromo)
	route = random.choice(chromo)
	if not route:
		return chromo
	s1 = random.randrange(len(route))
	s2 = random.randrange(len(route))
	route[s1], route[s2] = route[s2], route[s1]
	return chromo

def insertion(chromo):
	chromo = map(list, chromo)
	route = random.choice(chromo)
	if not route:
		return chromo
	s1 = random.randrange(len(route))
	s2 = random.randrange(len(route))
	s1, s2 = min(s1, s2), max(s1, s2)
	if s1 != s2:
		route[s1: s2] = route[s1+1: s2]+[route[s1]]
	return chromo

def cross_transposition(chromo):
	chromo = map(list, chromo)
	r1 = random.choice(chromo)
	r2 = random.choice(chromo)
	if not r1 or not r2 or r1 == r2:
		return chromo
	r1s1 = random.randrange(len(r1))
	r1s2 = random.randrange(len(r1))
	r2s1 = random.randrange(len(r2))
	r2s2 = random.randrange(len(r2))
	r1s1, r1s2 = min(r1s1, r1s2), max(r1s1, r1s2)
	r2s1, r2s2 = min(r2s1, r2s2), max(r2s1, r2s2)
	#print r1, r2, r1s1, r1s2, r2s1, r2s2
	r1c = r1[:]
	r1[:] = r1[:r1s1] + r2[r2s1:r2s2] + r1[r1s2:]
	r2[:] = r2[:r2s1] + r1c[r1s1:r1s2] + r2[r2s2:]
	return chromo

def identity(chromo):
	return map(list, chromo)

def applyallmutations(chromo):
	return map(lambda f: f(chromo), [identity, inversion, transposition, insertion, cross_transposition])

def mutate(parents):
	return reduce(lambda i, j: i + j, map(applyallmutations, parents))

def valid_sol(chromo):
	cities = reduce(lambda i, j: i + j, chromo)
	return sorted(cities) == range(1, len(cities)+1)

#######################################################
# Geo
#######################################################

def geocode(addr):
    url = "http://maps.googleapis.com/maps/api/geocode/json?address=%s&sensor=false" %   (urllib.quote(addr.replace(' ', '+')))
    data = urllib.urlopen(url).read()
    info = json.loads(data).get("results")[0].get("geometry").get("location")  
    return info


KEY = "AIzaSyAvdYcjcHlczmnZqPevfocmixv8N2GZCzA"
def distance_matrix(row):
	f = "%s.pkl" % abs(hash(row))
	if os.path.exists(f):
		with open(f) as fp:
			times = cPickle.load(fp)
		return times
	addrs = row.split(';')[1::2]
	addrs = map(lambda addr: addr.replace(' ', '+'), addrs)
	times = [[0.0]*len(addrs) for i in xrange(len(addrs))]
	for i in xrange(len(addrs)):
		for j in xrange(len(addrs)):
			url = "https://maps.googleapis.com/maps/api/distancematrix/json?origins=%s&destinations=%s&key=%s" % (addrs[i], addrs[j], KEY)
			data = json.loads(urllib.urlopen(url).read())
			times[i][j] = data["rows"][0]["elements"][0]["duration"]["value"]
			print times[i][j],
		print
	with open(f, 'w') as fp:
		cPickle.dump(times, fp)
	return times


#row = "BNJBEG201	25.4227356,86.1244413	120691	25.5313963,85.996848	BNJ_AD_00213	25.4815948,86.124288	BNJ_AD_00034	25.6833818,86.154827	BN0020	25.4595335,86.185347	BNJ_AD_00225	24.5535899,86.404422	120665	25.4446776,85.99728	122037	25.549306,85.96467	BN0010	25.597813,86.258876	120687	25.4206894,86.122435	122038	25.41301,86.124077	BNJ00002	25.4141668,86.121717	120876	25.626882,85.922788	120855	25.6869924,86.068253	120873	25.4595335,86.185347	BNJ_AD_00088	25.4409867,86.388351	BNJ_AD_00113	25.5584829,86.014708	BNJ_AD_00042	25.30703,86.236174	BNJ_AD_00212	25.5283593,86.103919	BNJ_AD_00226	25.30703,86.236174	120876	25.626882,85.922788	120855	25.6869924,86.068253	122047	25.5909247,86.236174	BNJ_AD_00201	25.2202021,86.560261	120885	25.5503738,86.042763"
#row = "BNJRAN232	22.54924,85.808342	120785	22.7023346,85.9278993	120791	22.67181,85.626221	BNJ_AD_00229	22.093214,85.883229"
#row = "BNJBHA207	25.2560821,86.9849308	121009	25.4408654,87.253826	BNJ_AD_00224	25.3674628,87.0021512	120290	25.24421,86.74468	121040	25.1683013,86.8922723	121061	25.1466223,86.9813617	BNJ00010	25.2177422,86.9924358	120292	25.3974889,86.859321	120213	25.249746,86.9655599	121085	25.9111958,86.8149079	120203	25.2177422,86.9924358	121109	25.4689681,87.1223362	BNJ00015	25.150192,87.1722037	BN0038	25.2177422,86.9924358"
row = "BNJRAN235	22.8026354,86.2043594	120786	22.95544349,86.05359277	126013	22.77282357,86.18860739	BNJ_AD_00247	22.65392388,86.35157869	120789	22.83703665,86.22863212	BN0024	22.81173565,86.17007587	126015	22.78672393,86.16093198	120405	22.80810555,86.21242414	120787	22.76741822,86.2204385	126008	22.81447755,86.10412226	BNJ_AD_00130	22.51070845,86.45754708	BNJ_AD_00186	22.27904345,86.7278621"
row = row.replace("\t", ";")
times = distance_matrix(row)
ret = solveMTSP(times, 8)
