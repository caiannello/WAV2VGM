import random
import datetime

random.seed(datetime.datetime.now().timestamp())

# genetic algorithm stuff.
# have methods to put in initial members, with initial fitness scores,
# while throwing out less-fit initial overpopulation

class gmemb:
  id = None
  fit = 9999999
  genome = None
  spect = None
  def __init__(self,id,fit,spect=None,genome=None):
    self.id = id
    self.fit = fit
    self.spect = spect
    self.genome = genome
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

  # the chromosomes of length 25 might actually be 
  # two chromosomes of lengths 13 and 12, if the
  # first element of the sequence is zero.

  chromosome_lens = [25,25,25,12,12,12,25,25,25,12,12,12]  

  def __init__(self, p_max=500, ideal=None, fitfunc=None):
    self.p_max = p_max  
    self.ideal = ideal
    self.fitfunc = fitfunc

  def add(self, id, fit, spect=None, genome=None):  # fitness: lower is better
    if self.p_ct < self.p_max:
      gm = gmemb(id,fit,spect,genome)
      self.p.append(gm)
      if fit<self.fbest:
        self.fbest=fit
      if fit>self.fworst:
        self.fworst=fit
      self.p_ct+=1
    elif fit < self.fworst:
      gm = gmemb(id,fit,spect,genome)
      self.p.append(gm)
      self.p.sort(key=lambda m: m.fit)
      self.p = self.p[0:-1]
      self.p_ct = self.p_max
      self.fbest = self.p[0].fit 
      self.fworst = self.p[-1].fit 

  # replace the worst half (or more) with new offspring
  # though crossover and ocassionally, mutation.
  # calc fitness for all ofspring, re-sort population
  def generate(self, mutatefcn, desperation):
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
      if random.random()<=0.9:                # 90% of offspring from single-point crossover
        cp = random.randint(0,len(self.chromosome_lens)-1)  # split at chromosomal boundary
        # figure out vector elem where that happens.
        vidx = 0
        j = 0
        while j<cp:
          cl = self.chromosome_lens[j]
          if (j==(cp-1)) and (cl==25) and ga[vidx] < 0.5:
            # if the last chrom of parent A is really a 
            # 13-long + 12-long rather than a 25-long, 
            # inherrit just the 13-long part from parent A 
            vidx+=13
            break
          vidx += cl
          j+=1
        genome = ga[0:vidx] + gb[vidx:]
      else:                                   # 10% chance of chromosomal shuffle
        genome = []  
        j = 0
        for cl in self.chromosome_lens:
          parent = random.randint(0,1)
          # if cl is 25: it could really be two chromosomes, 
          # a 13-long and a 12-long, if the first value of the 
          # segment is 0.00
          if cl==25:
            if parent:
              cfg = gb[j]
              if cfg>=0.5:
                genome += gb[j:j+cl]
              else:
                genome += gb[j:j+13]
                genome += ga[j+13:j+13+12]
            else:
              cfg = ga[j]
              if cfg>=0.5:
                genome += ga[j:j+cl]
              else:
                genome += ga[j:j+13]
                genome += gb[j+13:j+13+12]
          else:
            if parent:
              genome += gb[j:j+cl]
            else:
              genome += ga[j:j+cl]
          j+=cl
      mutagen = desperation
      if mutagen>4:
        mutagen = 4
      if random.randint(0,11) <= mutagen:
        mutants+=1
        genome = mutatefcn(genome, desperation)
      fit, spect = self.fitfunc(self.ideal, genome)
      self.p.append(gmemb(i,fit,spect,genome))  # child
    self.p.sort(key=lambda m: m.fit)
    print(f'die-off:{splitpoint:3d}, mutants:{mutants:3d}',end='')

