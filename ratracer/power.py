import numpy

LIGHT_SPEED = 299792458. # mps
TOLERANCE = 1e-9


# Radio signal utilities
def to_log(value, tolerance=TOLERANCE):
  return 10 * numpy.log10(value) if value > tolerance else -numpy.inf

def to_linear(value):
  return numpy.exp(value / 10)

def power(signal):
  return amplitude(signal) ** 2

def phase(signal):
  return numpy.angle(signal)

def amplitude(signal):
  return numpy.abs(signal)




class ReflectivityMixin:

  def __init__(self, *, reflection='fresnel', const_freq=True, frequency=1e9,
      permittivity=1, conductivity=.01, rvalue=-1., **kwargs):
    
    if reflection == 'fresnel':
      self.frequency = frequency
      self.permittivity = permittivity
      self.conductivity = conductivity
      self.const_freq = const_freq
      self.eta = permittivity - 60j * LIGHT_SPEED / frequency * conductivity
      self.reflection = self._r_fresnel
    
    elif reflection == 'constant':
      self.rvalue = rvalue
      self.reflection = self._r_constant
    
    else:
      raise ValueError('reflection should be `fresnel` or `constant`')


  def r_constant(self, aoa_cosine, **kwargs):
    return self.rvalue
  
  def r_fresnel(self, aoa_cosine, polarization=1., frequency=1e9):

    aoa_sine = (1 - aoa_cosine ** 2) ** 0.5

    eta = self.eta if self.const_freq else \
        self.permittivity - 60j * LIGHT_SPEED / frequency * self.conductivity

    c_prl = (eta - aoa_cosine ** 2) ** 0.5
    c_prp   = c_prl / eta

    r_prl = (aoa_sine - c_prl) / (aoa_sine + c_prl) if polarization !=0 else 0.j
    r_prp = (aoa_sine - c_prp) / (aoa_sine + c_prp) if polarization !=1 else 0.j

    return polarization * r_prl + (1 - polarization) * r_prp