###############################################################################
#
# genetic algorithm stuff.
#
###############################################################################
import random
import datetime
from   copy import deepcopy
import math
import bisect
try:
  from .OPL3 import OPL3
except:
  from OPL3 import OPL3

random.seed(datetime.datetime.now().timestamp())

opl3 = OPL3()
class gene:
  chromosome_lens = [25,25,25,12,12,12,25,25,25,12,12,12]  
  def __init__(self, p_max, ideal, permutables, mutatefcn):
    self.p = []
    self.p_max = p_max
    self.ideal = deepcopy(ideal)
    self.permutables = deepcopy(permutables)
    self.mutatefcn = mutatefcn
  def add(self, genome):
    global opl3
    gn = deepcopy(genome)
    fit, spect = opl3.fitness(self.ideal, gn)
    memb = {'genome':gn, 'spect':deepcopy(spect), 'fit':fit}
    bisect.insort_left(self.p, memb, key=lambda m: m['fit'])
    ct = len(self.p)
    if ct>self.p_max:
      self.p = self.p[0:self.p_max]
  def verifyAll(self):
    self.p.sort(key=lambda m: m['fit'])
    global opl3
    for i,o in enumerate(self.p[0:10]):
      f = deepcopy(o['fit'])
      s = deepcopy(o['spect'])
      v = deepcopy(o['genome'])
      vfit, vspect = opl3.fitness(self.ideal, v)
      dif = abs(vfit-f)
      print(f'p[{i:3d}]: p.fit={f:10.5f}, vfit={vfit:10.5f}')

  def generate(self, desperation, quiet):
    splitpoint = len(self.p)//2
    while self.p[splitpoint]['spect'] is None:      
      splitpoint-=1
      if splitpoint<0:
        print('WTF EXTINCTION!')
        exit()
    self.p = deepcopy(self.p[0:splitpoint])
    mutants = 0
    for i in range(splitpoint, self.p_max):
      a = random.randint(0,splitpoint-1)  # parents
      ga = deepcopy(self.p[a]['genome'])
      b = random.randint(0,splitpoint-1)
      gb = deepcopy(self.p[b]['genome'])
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
        gchild = ga[0:vidx] + gb[vidx:]
      else:                                   # 10% chance of chromosomal shuffle
        gchild = []  
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
                gchild += gb[j:j+cl]
              else:
                gchild += gb[j:j+13]
                gchild += ga[j+13:j+13+12]
            else:
              cfg = ga[j]
              if cfg>=0.5:
                gchild += ga[j:j+cl]
              else:
                gchild += ga[j:j+13]
                gchild += gb[j+13:j+13+12]
          else:
            if parent:
              gchild += gb[j:j+cl]
            else:
              gchild += ga[j:j+cl]
          j+=cl

      # possibly impose some mutations
      mutagen = desperation
      if mutagen>4:
        mutagen = 4
      if random.randint(0,11) <= mutagen:
        mutants+=1
        gchild = self.mutatefcn(gchild, desperation, self.permutables)      

      # add child to population
      self.add(gchild)
    if not quiet:
      print(f'mutants:{mutants:3d}',end='')      
###############################################################################
# EOF
###############################################################################
