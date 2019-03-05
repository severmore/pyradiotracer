import numpy
from numpy import linalg as lin
from itertools import count

from ratracer.utils import normalize, Singleton, TOLERANCE, zero, inf, vec3d

_SHADOWING_INDENT = .99999
_NO_INTERSECTION = numpy.nan, numpy.nan

# def abstract(method):
#   class AbstractMethodExcception(Exception): pass
#   def wrapper(*args, **kwargs):
#     raise AbstractMethodExcception(f'Method {method.__qualname__} is abstract')
#   return wrapper

class _Identifiable:
  _id_gen = count()
  def __init__(self):
    self.id = next(_Identifiable._id_gen)

# from inspect import getmembers

# def abstract_class(cls):
#   def f(o):
#     print('!', str(o).find(cls.__name__), str(o))
    
#     return False
#   for name, method in getmembers(cls, f):
#     print('[A]', name)
#     setattr(cls, name, abstract(method))
#   return cls


# @abstract_class
class Shape(_Identifiable):

  # @abstract
  def normal(self, point=None):
    """ Returns normal to the shape at a `point` """
    pass
  
  # @abstract
  def is_intersected(self, start, direction):
    """ Check whether a ray specified by its `start` and `direction` crosses
    the plane """
    pass
  
  # @abstract
  def intersection(self, start, direction, end=None):
    """ Returns an intersection point of a ray specified by its `start` 
    and `direction` with the plane """
    pass

  # @abstract
  def is_shadowing(self, start, direction, end=None):
    """ Check whether 'self' shadows the ray at a segement start - end. """
    pass
  
  # @abstract
  def grazing_angle(self, direction):
    """ Returns a cosine of grazing angle for ray hitting towards `direction`"""
    pass


class Empty(Shape, metaclass=Singleton):
  """ An empty shape """
  def __str__(self):
    self.id = numpy.inf
    return repr(self)
  
  def __repr__(self):
    return f'{self.__class__.__name__}'

  def normal(self, point=None): return zero
  def is_intersect(self, start, direction): return False
  def is_shadowing(self, start, end): return False
  def intersect(self, start, direction): 
    return


class Plane(Shape):
    
  def __init__(self, init_point, normal):
    self._point = init_point
    self._normal = normalize(normal)
    super().__init__()

  def __str__(self):
    return repr(self)
  
  def __repr__(self):
    classname = self.__class__.__name__.lower()
    return f'{classname}({self._point} {self._normal})'

  def _distance_to(self, point):
    """ A signed distance between a `point` and the plane; a positive value 
    means the normal directed to the same semi-plane as a point to be, negative 
    - an oposite semi-plane 
    """
    return numpy.dot(self._point - point, self._normal)

  def distance_to(self, point):
    """ Return the distance from the plane to a `point` """
    return numpy.abs(self._distance_to(point))
  
  def _aoa_cosine(self, direction):
    """ Returns a signed cosine of grazing angle (angle of arrival) for ray 
    hitting towards `direction`. 
    """
    return numpy.dot(direction, self._normal)

  def intersect(self, start, direction):
    """ Return the coeffient for an intersection point computation; an infinity
    value means orthogonality of `direction` and a plane normal, a negative 
    value mean a ray specified by `start` and `direction is directed in 
    opposition to the plane, thus in both cases there is no intersection
 
    Parameters
    __________
    start : 3-`numpy.ndarray`
        Starting point of ray to intersect
    direction : 3-`numpy.ndarray`
        A normalized direction of ray to intersect. 
    
    Note
    ____
    The unit length of direction vector is crucial.
    """
    aoa_cosine = self._aoa_cosine(direction)
    if numpy.abs(aoa_cosine) < TOLERANCE: 
      return _NO_INTERSECTION
    length = self._distance_to(start) / aoa_cosine
    return length, aoa_cosine if length > 0 else _NO_INTERSECTION

  # def is_intersected(self, start, direction):
    # """ Check whether a ray specified by its `start` and `direction` crosses
    # the plane """
    # return self._intersection_fraction(start, direction) != numpy.infty

  def is_shadowing(self, start, delta):
    """ Check whether 'self' shadows the ray at a segement start - end. """
    # delta is not normalized, thus one should check if length is in [0,1)
    length, aoa = self.intersect(start, delta * _SHADOWING_INDENT)
    return length > 0 and length < 1

  # def intersect(self, start, end):
    # """ Returns an intersection point of a ray specified by its `start` 
    # and `direction` with the plane """
    # direction = end - start
    # tau = self._intersection_fraction(start, direction)
    # return inf if tau == numpy.inf else start + tau * direction
  
  def grazing_angle(self, direction):
    """ Returns a cosine of grazing angle for ray hitting towards `direction`"""
    return numpy.abs(numpy.dot(direction, self._normal))


  def project(self, point):
    """ Project a point on the plane """
    return point + self._distance_to(point) * self._normal

  def reflect(self, point):
    """ Reflect a point from the plane """
    return point + 2 * self._distance_to(point) * self._normal
  
  def normal(self, point=None):
    """ Returns normal to the shape at a `point` """
    return self._normal


def build(specs):
  return [Plane(vec3d(*i), vec3d(*n)) for i, n in specs.items()]



if __name__ == '__main__':
  e1 = Empty()
  e2 = Empty()

  print(e1 is e2)