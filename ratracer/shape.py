import numpy
from numpy import linalg as lin
from itertools import count

from ratracer.utils import normalize, Singleton, TOLERANCE, zero, inf, vec3d

class Shape:

  _id_gen = count()

  def __init__(self):
    self.id = next(Shape._id_gen)
  
  def normal(self, point=None):
    """ Returns normal to the shape at a `point` """
    raise NotImplementedError('Normal is not defined for an abstarct shape')
  
  def is_intersected(self, start, direction):
    """ Check whether a ray specified by its `start` and `direction` crosses
    the plane """
    raise NotImplementedError(
      'Intersection is not defined for an abstract shape')
  
  def intersection(self, start, direction, end=None):
    """ Returns an intersection point of a ray specified by its `start` 
    and `direction` with the plane """
    raise NotImplementedError(
      'Intersection is not defined for an abstract shape')
  
  def grazing_angle(self, direction):
    """ Returns a cosine of grazing angle for ray hitting towards `direction`"""
    raise NotImplementedError(
      'Grazing angle is not defined for an abstract shape')


class Empty(Shape, metaclass=Singleton):
  """ An empty shape """
  def __str__(self):
    self.id = numpy.inf
    return repr(self)
  
  def __repr__(self):
    return f'{self.__class__.__name__}'

  def normal(self, point=None): return zero
  def is_intersect(self, start, direction): return False
  def is_shadowed(self, start, end): return False
  def intersect(self, start, end): return end


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

  def normal(self, point=None):
    """ Returns normal to the shape at a `point` """
    return self._normal


  def _distance_to(self, point):
    """ A signed distance between a `point` and the plane; a positive value 
    means the normal directed to the same semi-plane as a point to be, negative 
    - an oposite semi-plane 
    """
    return numpy.dot(point - self._point, self._normal)

  def _intersection_fraction(self, start, direction):
    """ Return the coeffient for an intersection point computation; an infinity
    value means orthogonality of `direction` and a plane normal, a negative 
    value mean a ray specified by `start` and `direction is directed in 
    opposition to the plane, thus in both cases there is no intersection 
    """
    denom = numpy.dot(direction, self._normal)
    if numpy.abs(denom) < TOLERANCE:
        return numpy.inf

    tau = -self._distance_to(start) / denom
    return tau if tau > 0 else numpy.inf


  def distance_to(self, point):
    """ Return the distance from the plane to a `point` """
    return numpy.abs(self._distance_to(point))


  def is_intersected(self, start, direction):
    """ Check whether a ray specified by its `start` and `direction` crosses
    the plane """
    return self._intersection_fraction(start, direction) != numpy.infty

  def is_shadowed(self, start, end):
    """ Check whether 'self' shadows the ray at a segement start - end. """
    tau = self._intersection_fraction(start, end - start)
    # print('[si_shadowed]', self, start, end, tau, tau >= 0 and tau <= 1, flush=True)
    return tau >= 0 and tau <= 1

  def intersect(self, start, end):
    """ Returns an intersection point of a ray specified by its `start` 
    and `direction` with the plane """
    direction = end - start
    tau = self._intersection_fraction(start, direction)
    # print('[tau]:', tau, direction, start)
    return inf if tau == numpy.inf else start + tau * direction


  def project(self, point):
    """ Project a point on the plane """
    return point - self._distance_to(point) * self._normal

  def reflect(self, point):
    """ Reflect a point from the plane """
    return point - 2 * self._distance_to(point) * self._normal
  
  def reflect_ray(self, start, direction):
    """ Returns a reflected ray """
    intersection = self.intersection(start, direction)
    r_dir = direction - 2 * numpy.dot(self._normal, direction) * self._normal
    return intersection, r_dir

  def reflected_mirrow_path(self, start, end):
    """ Build a path of the ray that crosses `start` and `end` points via 
    reflecting the plane using the method of mirrow reflection """
    dir_grazing = normalize(self.reflect(end) - start)
    intersection = self.intersection(start, dir_grazing)
    dir_reflected = normalize(end - intersection)

    return intersection, dir_grazing, dir_reflected
  
  def grazing_angle(self, direction):
    """ Returns a cosine of grazing angle for ray hitting towards `direction`"""
    direction = normalize(direction)
    return numpy.abs(numpy.dot(direction, self._normal))


def build(specs):
  return [Plane(vec3d(*i), vec3d(*n)) for i, n in specs.items()]


class Ray:

  id_ = count()

  def __init__(self, start, direction=None, end=None, length=0,
                force_geometry=False, rtype='primary'):
    """ Base class for rays object, containing all necessary geometrical 
    paramters required by a tracer. A ray is represented as a 3-vector 
    impelemented by numpy.ndarray object with starting, or initial, point 
    specified (also numpy.ndarray).

    Args:
      start     (numpy.ndarray) - the starting point of a ray.
      direction (numpy.ndarray, optional) - the direction of a ray.
      end       (numpy.ndarray, optional) - the ending point of a ray.
      length    (float, optional) - the lenght of a ray, i.e. the lenght of the 
              vector equaling `end` - `start`.
  
      force_geometry (bool, optional) - enforce ray geometry computation when
              not all geometrical parameters are specified.
      
      rtype (obj:`str`, optional) - a type of a ray
    
    Raises:
      obj:`ValueError` - in case of contrudiction between the parameters given 
              when `force_geometry` is set to `True`.
    """
    self.start = start
    self.direction = direction
    self.end = end
    self.length = length

    if force_geometry:
      self._compute_ray_geometry()
    
    self.id = next(Ray.id_)
    self.rtype = rtype


  def _compute_ray_geometry(self):
    """ Compute missed parameters and raises `ValueError` if they are not 
    specified properly or they contrudict to each other """

    if self.end is None and (self.direction is None or self.length == 0):
      raise ValueError(
        '`end` or a pair `direction` should `length` should be specified'
      )
    
    if self.end is None:
      self.direction = normalize(self.direction)
      self.end = self.start + self.direction * self.length
    
    if self.direction is None:
      self.direction = normalize(self.end - self.start)
      
      if numpy.abs(self._compute_length() - self.length) > TOLERANCE:
        raise ValueError('`length` does not match `start` and `end`')

  def _compute_length(self):
    return lin.norm(self.end - self.start)


if __name__ == '__main__':
  e1 = Empty()
  e2 = Empty()

  print(e1 is e2)