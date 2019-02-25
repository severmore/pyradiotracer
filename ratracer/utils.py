import numpy
import itertools
from inspect import signature
from functools import wraps
from numpy.linalg import norm

from ratracer import settings

TOLERANCE = 1e-6
COLORS = {
  'red': 31, 
  'green': 32,
  'yellow': 33,
  'blue': 34,
  'turq': 36,
}

class Singleton(type):
  _instances = {}
  def __call__(cls, *args, **kwargs):
    if cls not in cls._instances:
      cls._instances[cls] = super().__call__(*args, **kwargs)
    return cls._instances[cls]


class ProgressBar():

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


def product_no_consecutives(iterables, repeat=2):
    """ Returns Cartesian product of input iterable excluding those contaning
    equal consecutives elements.

    See: `itertools.product`
    """
    def has_no_equal_consecutives(sequence):
      it, it_shifted = itertools.tee(sequence)
      next(it_shifted, None)
      return not any(id1 == id2 for id1, id2 in zip(it, it_shifted))

    return filter(has_no_equal_consecutives, 
                  itertools.product(iterables, repeat=repeat))


####################
# Geometry utulities
####################
def vec3d(x, y, z):
  return numpy.array([float(x), float(y), float(z)])

def vec3d_sequnced(sequence):
  return vec3d(*sequence)

def normalize(x, tolerance=TOLERANCE):
  n = length(x)
  return x / n if n > tolerance else zero

def length(x):
  return norm(x)

zero = vec3d(0.,0.,0.)
inf  = vec3d(numpy.inf, numpy.inf, numpy.inf)


########################
# PRINTING ROUTINS
########################

def _apply_color(color_str):
  """ Produce color function based on its name; if None is given the output 
  will be an identical function, i.e. lambda s: s """
  color_no = COLORS.get(color_str)
  if color_no is None:
    return lambda s: s
  return lambda s: f'\x1b[{color_no}m{s}\x1b[0m'

def verbose(color):
  def decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
      if not settings['verbosity']:
        return
      color_ = color if settings['vcolored'] else None
      kwargs['color'] = _apply_color(color_)
      return func(*args, **kwargs)
    return wrapper
  return decorator

