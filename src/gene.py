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

opl3 = OPL3()

random.seed(datetime.datetime.now().timestamp())

# the gene pool
class gene:
  # spectrum we hope to achieve
  ideal = None
  # the population
  p = []
  # upper limit of pop size
  p_max = 500

  # for debug: trying to diagnose a problem using these
  best_fit = None
  best_genome = None
  best_spect = None

  # The chromosomes below of length 25 might actually be 
  # two chromosomes of lengths 13 and 12, if the first 
  # element of that chromosome is zero.
  chromosome_lens = [25,25,25,12,12,12,25,25,25,12,12,12]  
  # indexes in the above array of the long ones
  long_chromos = []
  # and the short ones
  short_chromos = []

  def __init__(self, p_max, ideal):
    self.p_max = p_max  
    self.ideal = deepcopy(ideal)
    ofs = 0
    for i,l in enumerate(self.chromosome_lens):
      if l==25:
        self.long_chromos.append(ofs)
      else:
        self.short_chromos.append(ofs)
      ofs+=l

  def add(self, genome):  # fitness: lower is better
    gn = deepcopy(genome)
    fit, spect = opl3.fitness(self.ideal, gn)
    memb = (genome, deepcopy(spect), fit)
    bisect.insort_left(self.p, memb, key=lambda r: r[2])
    ct = len(self.p)
    if ct>self.p_max:
      self.p = self.p[0:self.p_max]
    self.best_fit = deepcopy(self.p[0][2])
    self.best_genome = deepcopy(self.p[0][0])
    self.best_spect = deepcopy(self.p[0][1])

  # replace the worst half (or more) with new offspring
  # though crossover, etc, with occasional mutaations.
  # Then, re-sort population by fitness.

  def generate(self, mutatefcn, desperation):
    splitpoint = len(self.p)//2
    while self.p[splitpoint][1] is None:
      splitpoint-=1
      if splitpoint<0:
        print('WTF EXTINCTION!')
        exit()
    self.p = self.p[0:splitpoint]
    mutants = 0
    for i in range(splitpoint, self.p_max):
      a = random.randint(0,splitpoint-1)  # parents
      ga = self.p[a][0]
      b = random.randint(0,splitpoint-1)
      gb = self.p[b][0]
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
        genome = deepcopy(ga[0:vidx]) + deepcopy(gb[vidx:])
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
                genome += deepcopy(gb[j:j+cl])
              else:
                genome += deepcopy(gb[j:j+13])
                genome += deepcopy(ga[j+13:j+13+12])
            else:
              cfg = ga[j]
              if cfg>=0.5:
                genome += deepcopy(ga[j:j+cl])
              else:
                genome += deepcopy(ga[j:j+13])
                genome += deepcopy(gb[j+13:j+13+12])
          else:
            if parent:
              genome += deepcopy(gb[j:j+cl])
            else:
              genome += deepcopy(ga[j:j+cl])
          j+=cl

      # Lets also sometimes swap around any chromosomes 
      # that are interchangable to hopefully snap out of
      # some local minima
      '''
      swaps = 0
      if random.random()<0.05:
        swaps = random.randint(1,4)
        for s in range(swaps):
          if random.random()>=0.5:
            ca,cb = random.sample(self.long_chromos,2)
            a = genome[ca:ca+25]
            b = genome[cb:cb+25]
            genome[cb:cb+25] = a
            genome[ca:ca+25] = b
          else:
            ca,cb = random.sample(self.short_chromos,2)
            a = genome[ca:ca+12]
            b = genome[cb:cb+12]
            genome[cb:cb+12] = a
            genome[ca:ca+12] = b
      '''
      # And possibly, impose some mutations
      mutagen = desperation
      if mutagen>4:
        mutagen = 4
      if random.randint(0,11) <= mutagen:
        mutants+=1
        genome = mutatefcn(genome, desperation)      
      
      self.add(genome)

    # if desperate, do per-chromosome piecewise fitness 
    # and impose upon the worst ones a higher chance of
    # mutation.
    nukes=0
    jostles = 0
    if desperation>=5:
      cgene = deepcopy(self.p[0][0])
      j = 0
      k = 0
      pfits = []
      fsum = 0
      for cl in self.chromosome_lens:
        chromo = cgene[j:j+cl]
        pgenome = [0.00]*j + chromo + [0.00]*(len(cgene) - (j+cl))
        pfit, _ = opl3.fitness(self.ideal, pgenome)
        fsum+=pfit
        pfits.append((k,j,cl,pfit))
        j+=cl
        k+=1
      pfits.sort(key=lambda m: m[3])
      fsum/=k
      for f in pfits:
        idx,pos,cl,fit = f
        if math.isinf(fit): # nuke chromosomes that arent helping at all
          cgene[pos:pos+cl] = [ random.random() for q in range(cl) ]
          nukes+=1
        elif fit>fsum: # if worse than average, jostle vals
          jostles += 1
          for j,velem in enumerate(cgene[pos:pos+cl]):
            f = random.random()+0.5
            velem *= f
            if velem<0.0:
              velem = 0
            elif velem>1.0:
              velem = 1.0
            cgene[pos+j] = velem
      self.add(cgene)
    print(f'mutants:{mutants:3d}, nukes:{nukes:2d}, jostles:{jostles:2d}',end='')
###############################################################################
# EOF
###############################################################################
