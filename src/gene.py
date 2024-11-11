import random

# genetic algorithm stuff.
# have methods to put in initial members, with initial fitness scores,
# while throwing out less-fit initial overpopulation

class gmemb:
	id = None
	fit = 9999999
	genome = None
	spect = None
	def __init__(self,id,fit):
		self.id = id
		self.fit = fit
		self.genome = None
	def __str__(self):
		return f'{self.id=} {self.fit=} {self.genome=}'

class gene:
	ideal = None
	p = []
	p_ct = 0
	p_max = 500
	fbest = 999999999
	fworst = -999999999
	fitfunc = None
	def __init__(self, p_max=500, ideal=None):
		self.p_max = p_max	
		self.ideal = ideal

	def add(self, id, fit):  # fitness: lower is better
		if self.p_ct < self.p_max:
			gm = gmemb(id,fit)
			self.p.append(gm)
			if fit<self.fbest:
				self.fbest=fit
			if fit>self.fworst:
				self.fworst=fit
			self.p_ct+=1
		elif fit < self.fworst:
			gm = gmemb(id,fit)
			self.p.append(gm)
			self.p.sort(key=lambda m: m.fit)
			self.p = self.p[0:-1]
			self.p_ct = self.p_max
			self.fbest = self.p[0].fit 
			self.fworst = self.p[-1].fit 

	def setInitialFitness(self, fitfunc):
		print('Doing inital fitness assessment...')
		self.fitfunc = fitfunc
		for gm in self.p:
			gm.fit, gm.spect = self.fitfunc(self.ideal, gm.genome)
		self.p.sort(key=lambda m: m.fit)
		print('done')

	# replace the worst half (or more) with new offspring
	# though crossover and ocassionally, mutation.
	# calc fitness for all ofspring, re-sort population
	def generate(self, mutatefcn):
		splitpoint = len(self.p)//2
		while self.p[splitpoint].spect is None:
			splitpoint-=1
			if splitpoint<0:
				print('WTF EXTINCTION!')
				exit()
		self.p = self.p[0:splitpoint]
		mutants = 0
		for i in range(splitpoint, self.p_max):
			a = random.randint(0,splitpoint-1)  # parents
			ga = self.p[a].genome
			b = random.randint(0,splitpoint-1)
			gb = self.p[b].genome
			gm = gmemb(0,0)			            # child
			
			cp = random.randint(0,511)          # single-point crossover
			gm.genome = ga[0:cp] + gb[cp:]

			if random.randint(0,11) == 0:
				mutants+=1
				gm.genome = mutatefcn(gm.genome)
			gm.fit, gm.spect = self.fitfunc(self.ideal, gm.genome)
			self.p.append(gm)
		self.p.sort(key=lambda m: m.fit)
		print(f'split:{splitpoint:3d}, mutants:{mutants}')

