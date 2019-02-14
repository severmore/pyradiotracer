import numpy
from numpy.linalg import norm

TOLERANCE = 1e-6

class Singleton(type):
  _instances = {}
  def __call__(cls, *args, **kwargs):
    if cls not in cls._instances:
      cls._instances[cls] = super().__call__(*args, **kwargs)
    return cls._instances[cls]

class ProgressBar:

  def __init__(self, total, prefix='', suffix='', 
                    decimals=0, length=100, filling='█', empty='_'):
    """
    Creates terminal progress bar.

    Args:
      total   (int)           : a total number of iterations
      prefix  (str, optional) : prefix to progress bar
      suffix  (str, optional) : suffix of progress bar
      decimal (int, optional) : a number of decimals in percent complete
      length  (int, optional) : a bar length in characters
      fill    (str, optional) : the fillig of a progress bar
      empty   (str, optional) : an empty symbol in a progress bar
    """ 
    self.total = total - 1 # consider starting with 0
    self.prefix = prefix
    self.suffix = suffix
    self.decimals = decimals
    self.length = length
    self.filling = filling
    self.empty = empty
  
  def show(self, iteration):
    """Print and update a progress bar status; `iteration` is a current 
    iterations """
    percent = 100 * iteration / float(self.total)
    percent_str = f"{percent:0.{self.decimals}f}"
    len_filled = int(self.length * iteration // self.total)
    bar = self.filling * len_filled + self.empty * (self.length - len_filled)

    print(f'{self.prefix} |{bar}| {percent_str}% {self.suffix}', end='\r')
    # print a new line on complete
    if iteration == self.total:
      print()


def reversed_enumerate(sequence): 
  return zip(range(len(sequence) - 1, -1, -1), reversed(sequence))


####################
# Geometry utulities
####################

vec3d = lambda x, y, z: numpy.array([float(x), float(y), float(z)])

def normalize(x, tolerance=TOLERANCE):
  n = length(x)
  return x / n if n > tolerance else zero

def length(x):
  return norm(x)

zero = vec3d(0.,0.,0.)
inf  = vec3d(numpy.inf, numpy.inf, numpy.inf)

  