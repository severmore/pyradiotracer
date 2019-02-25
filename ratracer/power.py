import numpy

LIGHT_SPEED = 299792458. # mps
TOLERANCE = 1e-9


# Radio signal utilities
def to_log(value, tolerance=TOLERANCE):
  return 10 * numpy.log10(value) if value > tolerance else -numpy.inf

def to_linear(value): return numpy.exp(value / 10)
def power(signal): return amplitude(signal) ** 2
def phase(signal): return numpy.angle(signal)
def amplitude(signal): return numpy.abs(signal)

def sine(cosine): return (1 - cosine ** 2) ** 0.5

class Reflectivity:

  class InvalidKindException(Exception): pass

  def __init__(self, *, kind='fresnel', const_freq=True, frequency=1e9,
      permittivity=1, conductivity=.01, rvalue=-1., **kwargs):
    
    if kind == 'fresnel':
      self.frequency = frequency
      self.permittivity = permittivity
      self.conductivity = conductivity
      self.const_freq = const_freq
      self.eta = permittivity - 60j * LIGHT_SPEED / frequency * conductivity
      self.reflection = self._fresnel
    
    elif kind == 'constant':
      self.rvalue = rvalue
      self.reflection = self._constant
    
    else:
      msg = '`kind` should be "fresnel" or "constant"'
      raise Reflectivity.InvalidKindException(msg)


  def _constant(self, aoa_cosine, **kwargs):
    return self.rvalue
  
  def _fresnel(self, aoa_cosine, polarization=1., frequency=1e9):

    aoa_sine = sine(aoa_cosine)

    eta = self.eta if self.const_freq else \
        self.permittivity - 60j * LIGHT_SPEED / frequency * self.conductivity

    c_prl = (eta - aoa_cosine ** 2) ** 0.5
    c_prp   = c_prl / eta

    r_prl = (aoa_sine - c_prl) / (aoa_sine + c_prl) if polarization !=0 else 0.j
    r_prp = (aoa_sine - c_prp) / (aoa_sine + c_prp) if polarization !=1 else 0.j

    return polarization * r_prl + (1 - polarization) * r_prp


class RadiactionPattern:

  class InvalidKindException(Exception): pass

  def __init__(self, *, kind='isotropic', wavelen=1e9, width=0., height=0.):
    if kind == 'isotropic':
      self.pattern = self._isotropic
    elif kind == 'dipole':
      self.pattern = self._dipole
    elif kind == 'patch':
      # Complete this later
      self.pattern = self._patch
      self.wavelen = wavelen
      self.width = width
      self.height = height
      raise NotImplementedError('Patch antenna is not supported yet')
    else:
      msg = 'kind should be "isotropic" or "dipole"'
      raise RadiactionPattern.InvalidKindException(msg)
    
  def _isotropic(self, *args):
    return 1.0
  
  def _dipole(self, ra_cos):
    """ Radiation pattern of dipole. :param:`a_cos` (float) - cosine of an 
    angle between radiating direction and antenna axis.
    """
    if ra_cos < TOLERANCE:
      return 0.
    ra_sin = get_sine(ra_cos)
    return numpy.abs(numpy.cos(numpy.pi / 2 * ra_sin) / ra_cos)

  def _patch_factor(ra_cos, rt_cos):
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
    
    return numpy.sin(kw * ra_sin * rt_sin) *
            numpy.cos(kh * ra_sin * rt_cos) /  \
                    (kw * ra_sin * rt_sin)
               
  def _patch(self, ra_cos, rt_cos):
    return numpy.abs(_patch_factor(ra_cos, rt_cos)) * \
          (rt_cos ** 2 + ra_cos ** 2 * sine(rt_cos) ** 2) ** 0.5
