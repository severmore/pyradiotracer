#%%
import numpy as np
from ratracer import radio

isotropic_rp = lambda a: 1.0
dipole_rp = lambda a: np.abs(np.cos(np.pi/2 * np.sin(a)) / np.cos(a))
constant_r = lambda a: -1.0
w2db = lambda w: 10.0 * np.log10(w)

class Node:
  def __init__(self, height, angle=None, rp=None):
    self.height = height
    self.angle = angle if angle is not None else np.pi/2
    self.rp = rp if rp else dipole_rp

  def G(self, theta):
    return self.rp(self.angle - theta)

class Pathloss:
  def __init__(self, freq=860e6):
    self.freq = freq
    self.wavelen = radio.LIGHT_SPEED / freq
    self.K = self.wavelen / (4 * np.pi)

  def los(self, distance, node_a, node_p):
    d0 = np.sqrt((node_a.height - node_p.height) ** 2 + distance ** 2)
    a0 = np.arctan(distance / (node_a.height - node_p.height))
    g0 = node_a.G(a0) * node_p.G(a0)
    return (self.K*g0/d0)**2

  def tworay(self, distance, node_a, node_p, gr=constant_r):
    d0 = np.sqrt((node_a.height - node_p.height) ** 2 + distance ** 2)
    d1 = np.sqrt((node_a.height + node_p.height) ** 2 + distance ** 2)
    a0 = np.arctan(distance / (node_a.height - node_p.height))
    a1 = np.arctan(distance / (node_a.height + node_p.height))
    g0 = node_a.G(a0) * node_p.G(a0)
    g1 = node_a.G(a1) * node_p.G(a1)
    gd = gr(a1)
    # print('tworay', g0, g1, gd)
    return self.K**2 * ((g0/d0)**2 + (g1*gd/d1)**2 +
                  2*(g0*g1*gd) / (d0*d1) * np.cos((d1-d0)/(2*self.K)))

if __name__ == '__main__':

  import sys
  if sys.stdin.isatty():
    import matplotlib
    matplotlib.use('TkAgg')

  from matplotlib import pyplot as plt
  try:
    from jupyterthemes import jtplot
    jtplot.style(theme='monokai', fscale=0.9, figsize=(8,6))
  except:
    pass

  iso_reader = Node(height=5, rp=isotropic_rp)
  iso_tag = Node(height=0.5, rp=isotropic_rp)
  dip_reader = Node(height=5, angle=np.pi/2)
  dip_tag = Node(height=.5, angle=np.pi/2)

  pl = Pathloss()

  ox = np.linspace(0.1, 20, 1000)
  pl_los_iso = w2db(pl.los(ox, iso_reader, iso_tag))
  pl_2ray_iso = w2db(pl.tworay(ox, iso_reader, iso_tag))
  pl_2ray_dip = w2db(pl.tworay(ox, dip_reader, dip_tag))
  # print(w2db(pl.tworay(ox, dip_reader, dip_tag)))

  fig = plt.figure()
  ax = plt.subplot(111)
  plt.plot(ox, pl_los_iso, 'y--', label='FSPL for isotropic antennas')
  plt.plot(ox, pl_2ray_iso, 'r', label='2-Ray PL for isotropic antennas')
  plt.plot(ox, pl_2ray_dip, 'b', label='2-Ray PL for dipole antennas')
  ax.set_ybound(lower=-90, upper=-30)
  plt.legend()
  plt.show()


#%%
