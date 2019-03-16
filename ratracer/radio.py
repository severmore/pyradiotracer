import numpy
import math
from functools import reduce
from operator import mul

from ratracer import shape
from ratracer.utils import zero, vec3d
from ratracer.tracer import Tracer, view

LIGHT_SPEED = 299792458. # mps
TOLERANCE = 1e-9


# Radio signal utilities
def to_log(value, tol=TOLERANCE):
  return 10 * numpy.log10(value) if value > tol else -numpy.inf
def to_linear(value): return numpy.exp(value / 10)
def power(signal): return amplitude(signal) ** 2
def phase(signal): return numpy.angle(signal)
def amplitude(signal): return numpy.abs(signal)

def sine(cosine): return (1 - cosine ** 2) ** 0.5

class Reflectivity:

  class InvalidKind(Exception): pass

  def __init__(self, *, kind='fresnel', const_freq=True, frequency=1e9,
      permittivity=1, conductivity=.01, rvalue=-1.):

    if kind == 'fresnel':
      self.frequency = frequency
      self.permittivity = permittivity
      self.conductivity = conductivity
      self.const_freq = const_freq
      self.eta = permittivity - 60j * LIGHT_SPEED / frequency * conductivity
      self._model = self._fresnel

    elif kind == 'constant':
      self.rvalue = rvalue
      self._model = self._constant

    else:
      raise Reflectivity.InvalidKind('should be "fresnel" or "constant"')

  def __call__(self, aoa_cosine, *args):
    return self._model(aoa_cosine, *args)

  def _constant(self, aoa_cosine, *args):
    return self.rvalue

  def _fresnel(self, aoa_cosine, polarization=1., frequency=1e9, *args):
    aoa_sine = sine(aoa_cosine)
    eta = self.eta if self.const_freq else \
        self.permittivity - 60j * LIGHT_SPEED / frequency * self.conductivity

    c_prl = (eta - aoa_cosine ** 2) ** 0.5
    c_prp   = c_prl / eta

    r_prl = (aoa_sine - c_prl) / (aoa_sine + c_prl) if polarization !=0 else 0.j
    r_prp = (aoa_sine - c_prp) / (aoa_sine + c_prp) if polarization !=1 else 0.j

    return polarization * r_prl + (1 - polarization) * r_prp


class AntennaPattern:

  class InvalidKind(Exception): pass
  class InvalidSector(Exception): pass
  MAX_SECTOR_WIDTH = numpy.pi

  def __init__(self, *, kind='isotropic', wavelen=1e9, width=0., height=0.,
                      sector_width=numpy.pi/3):
    if kind == 'isotropic':
      self._model = self._isotropic
    elif kind == 'dipole':
      self._model = self._dipole
    elif kind == 'sector':
      if sector_width > AntennaPattern.MAX_SECTOR_WIDTH:
        raise AntennaPattern.InvalidSector('Sector width should be less pi')
      self._sector_width = numpy.cos(sector_width / 2)
      self._model = self._sector
    elif kind == 'patch':
      # Complete this later
      self._model = self._patch
      self.wavelen = wavelen
      self.width = width
      self.height = height
      raise NotImplementedError('Patch antenna is not supported yet')
    else:
      raise AntennaPattern.InvalidKind('kind should be "isotropic" or "dipole"')

  def __call__(self, ra_cos, rt_cos=None):
    return self._model(ra_cos, rt_cos)

  def _isotropic(self, *args):
    return 1.

  def _sector(self, ra_cos, *args):
    return 1. if ra_cos > self._sector_width else 0.

  def _dipole(self, ra_cos, *args):
    """ Radiation pattern of dipole. :param:`ra_cos` (float) - cosine of an
    angle between radiating direction and antenna axis.
    """
    if ra_cos < TOLERANCE:
      return 0.
    ra_sin = sine(ra_cos)
    return numpy.abs(numpy.cos(numpy.pi / 2 * ra_sin) / ra_cos)

  def _patch_factor(self, ra_cos, rt_cos):
    ra_sin = sine(ra_cos)
    rt_sin = sine(rt_cos)
    kw = numpy.pi / self.wavelen * self.width
    kh = numpy.pi / self.wavelen * self.height

    if ra_cos < TOLERANCE:
      return 0
    if numpy.abs(ra_sin) < TOLERANCE:
      return 1.
    elif numpy.abs(rt_sin) < TOLERANCE:
      return numpy.cos(kh * ra_sin)

    return numpy.sin(kw * ra_sin * rt_sin) *  \
           numpy.cos(kh * ra_sin * rt_cos) /  \
                    (kw * ra_sin * rt_sin)

  def _patch(self, ra_cos, rt_cos):
    return numpy.abs(self._patch_factor(ra_cos, rt_cos)) * \
          (rt_cos ** 2 + ra_cos ** 2 * sine(rt_cos) ** 2) ** 0.5


class RFAttenuator:

  def __init__(self, reflectivity):
    self._reflectivity = reflectivity if reflectivity else Reflectivity()

  def att(self, aoa, *args):
    return self._reflectivity(aoa, *args)

class Movable:
  def __init__(self, velocity=None):
    self.velocity = velocity if velocity is not None else zero

class RFPlane(shape.Plane, Movable, RFAttenuator):
  def __init__(self, init_point, normal, velocity=None, reflectivity=None):
    shape.Plane.__init__(self, init_point, normal)
    Movable.__init__(self, velocity)
    RFAttenuator.__init__(self, reflectivity)


class RFDevice:
  def __init__(self, position, freq=1e9, *, pattern=None, ant_normal=None):
    self.position = position
    self.frequency = freq
    self._pattern = pattern if pattern else AntennaPattern()
    self._default_an = vec3d(1,0,0)
    self.ant_normal = ant_normal if ant_normal is not None else self._default_an

  def att(self, direction):
    return self._pattern(numpy.dot(self.ant_normal, direction))


def build(specs, freq):
  reflector = Reflectivity(kind='constant', frequency=freq)
  return (RFPlane(vec3d(*i), vec3d(*n), zero, reflector)
          for i, n in specs.items())

class KRayPathloss:

  def __init__(self, scene, frequency=1e9):
    """ Produce k-ray pathloss model based on the 'scene'. """
    self._tracer = Tracer(build(scene, frequency))

    self._frequency = frequency
    self._wavelen = LIGHT_SPEED / frequency
    self._k = 2 * numpy.pi / self._wavelen

    self._pathloss = 0

  def __call__(self, tx, rx, max_reflections=2):
    """ Compute pathloss on propagation from `tx` to `rx`. """
    self.clear()
    result = self._tracer(tx.position, rx.position, max_reflections)

    for dirs, lens, sids, aoas in result.trace:
      length = sum(lens)
      reflectance = reduce(mul, self._r_att(sids, aoas), 1)
      pattern_att = tx.att(dirs[0]) * rx.att(dirs[-1])
      print('k-ray-pathloss', length, reflectance, pattern_att)
      self._pathloss += self._1ray_pathloss(length, reflectance, pattern_att)

    return self._pathloss


  def _r_att(self, sids, aoas):
    """ Get attenuation due to reflection from the shapes. """
    for sid, aoa in zip(sids, aoas):
      yield self._tracer.scene[sid].att(aoa)

  def _1ray_pathloss(self, length, reflectance, pattern_att):
    """ Compute pathloss along a path. """
    kl = self._k * length
    return .5 / kl * numpy.exp(-1j * kl) * reflectance * pattern_att

  def clear(self):
    """ Clear the results of previous pathloss evaluation. """
    self._pathloss = 0

  def set_frequency(self, f):
    """ Change frequency with concominant `wavelen` and `k` parameters. """
    self._frequency = f
    self._wavelen = LIGHT_SPEED / f
    self._k = 2 * numpy.pi / self.wavelen



if __name__ == '__main__':
  from ratracer.utils import vec3d as vec

  scene = {
    (0,0, 0): (0,0,1),
    # (0,0,10): (0,0,-1),
    # (5,0,0): (-1,0,0),
    # (-5,0,0): (1,0,0),
  }

  transmitter = RFDevice(vec(0,0,5), 860e6, ant_normal=vec(0,1,0))
  receiver = RFDevice(vec(0,10,5), 860e6, ant_normal=vec(0,-1,0))
  pathloss_model = KRayPathloss(scene)
  pathloss = pathloss_model(transmitter, receiver, max_reflections=1)

  print(to_log(power(pathloss)))