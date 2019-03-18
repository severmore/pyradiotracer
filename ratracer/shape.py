"""
'ratracer.shape' define all types of shapes for ray tracing to be used.
Contains classes:

  :class:`.Identifiable` -
      class defining local and brief ids for user's shapes.
  :class:`.Plane` -
      a plane specified by its normal and some point on its surface,
      that can be treated as zero-point in 2D-coordinates on the plane.
"""
import numpy
from itertools import count

from ratracer.utils import normalize, TOLERANCE, vec3d

_SHADOWING_INDENT = TOLERANCE
_SHADOWING_INDENT_INV = 1 - _SHADOWING_INDENT
_NO_INTERSECTION = numpy.nan, numpy.nan

class Identifiable:
  """ A class that intended to be subclassed to provide a brief id. """
  _id_gen = count()
  def __init__(self):
    self.id = next(Identifiable._id_gen)

class Plane(Identifiable):

  def __init__(self, init_point, normal):
    self._point = init_point
    self._normal = normalize(normal)
    super().__init__()

  def __str__(self):
    return repr(self)

  def __repr__(self):
    classname = self.__class__.__name__.lower()
    return f'{classname}({self._point} {self._normal})'

  def _aoa_cosine(self, direction):
    """ Returns a signed cosine of grazing angle (angle of arrival) for ray
    hitting towards `direction`.
    """
    return numpy.dot(direction, self._normal)

  def aoa_cosine(self, direction):
    """ Returns a cosine of grazing angle for ray hitting towards `direction`"""
    return numpy.abs(self._aoa_cosine(direction))

  def _distance_to(self, point):
    """ A signed distance between a `point` and the plane; a positive value
    means the normal directed to the same semi-plane as a point to be, negative
    - an oposite semi-plane
    """
    return numpy.dot(self._point - point, self._normal)

  def distance_to(self, point):
    """ Return the distance from the plane to a `point` """
    return numpy.abs(self._distance_to(point))

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
    return (length, aoa_cosine) if length > 0 else _NO_INTERSECTION

  def is_intersected(self, start, direction):
    """ Check whether a ray specified by its :arg:`start` and :arg:`direction`
    crosses the plane. """
    return self.intersect(start, direction) != _NO_INTERSECTION

  def is_shadowing(self, start, delta):
    """ Check whether 'self' shadows the ray at a segement start - end. Note,
    that :arg:delta should not be normalized. """
    length, _ = self.intersect(start, delta)
    return length > _SHADOWING_INDENT and length < _SHADOWING_INDENT_INV

  def normal(self, point=None):
    """ Returns normal to the shape at a :arg:`point` (:arg:`point` is
    insufficient here). """
    return self._normal

  def project(self, point):
    """ Project a point on the plane. """
    return point + self._distance_to(point) * self._normal

  def reflect(self, point):
    """ Reflect a point from the plane. """
    return point + 2 * self._distance_to(point) * self._normal


def build(specs):
  return [Plane(vec3d(*i), vec3d(*n)) for i, n in specs.items()]
