import numpy
from numpy import linalg as lin


TOLERANCE = 1e-6
INFTY = vec3d(numpy.inf, numpy.inf, numpy.inf)

# Geometry utilities
def vec3d(x, y, z):
  return numpy.array([x, y, z])

def normalize(x):
  norm = lin.norm(x)
  return x / norm if norm > TOLERANCE else vec3d(0., 0., 0.)


class Shape:
  
  def normal(self, point=None):
    """ Returns normal to the shape at a `point` """
    pass
  
  
class Plane(Shape):
    
  def __init__(self, init_point, normal, **kwargs):
    self._point = init_point
    self._normal = normalize(normal)

  def normal(self, point=None):
    """ Returns normal to the shape at a `point` """
    return self.normal


  def _distance_to(self, point):
    """ A signed distance between a `point` and the plane; a positive value 
    means the normal directed to the same semi-plane as a point to be, negative 
    - an oposite semi-plane """
    return numpy.dot(point - self._point, self._normal)

  def _intersection_fraction(self, start, direction):
    """ Return the coeffient for an intersection point computation; an infinity
    value means orthogonality of `direction` and a plane normal, a negative 
    value mean a ray specified by `start` and `direction is directed in 
    opposition to the plane, thus in both cases there is no intersection """
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

  def intersection(self, start, direction):
    """ Returns an intersection point of a ray specified by its `start` 
    and `direction` with the plane """
    tau = self._intersection_fraction(start, direction)
    return start + tau * direction


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