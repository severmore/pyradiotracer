#%%
import numpy as np
from ratracer import radio

import sys
if sys.stdin.isatty():
  import matplotlib
  matplotlib.use('TkAgg')

from matplotlib import pyplot as plt

try:
  from jupyterthemes import jtplot
  jtplot.style(theme='monokai', fscale=0.9)
except:
  pass

isotropic_rp = lambda a: 1.0
dipole_rp = lambda a: np.abs(np.cos(np.pi/2 * np.sin(a)) / np.cos(a))
constant_r = lambda a: -1.0
w2db = lambda w: 10.0 * np.log10(w)

C = 299792458
frequency = 860e6
wavelen = C / frequency
reflection = constant_r

class Node:
  def __init__(self, height, angle, rp=None):
    self.height = height
    self.angle = angle
    self.rp = rp if rp else dipole_rp

  def G(self, theta):
    return self.rp(self.angle - theta)

def PL1(distance, node_a, node_p):
  d0 = np.sqrt((node_a.height - node_p.height) ** 2 + distance ** 2)
  a0 = np.arctan(distance / (node_a.height - node_p.height))
  g0 = node_a.G(a0) * node_p.G(a0)
  K = wavelen / (4*np.pi)
  return (K*g0/d0)**2

def PL2(distance, node_a, node_p):
  d0 = np.sqrt((node_a.height - node_p.height) ** 2 + distance ** 2)
  d1 = np.sqrt((node_a.height + node_p.height) ** 2 + distance ** 2)
  a0 = np.arctan(distance / (node_a.height - node_p.height))
  a1 = np.arctan(distance / (node_a.height + node_p.height))
  g0 = node_a.G(a0) * node_p.G(a0)
  g1 = node_a.G(a1) * node_p.G(a1)
  gd = reflection(a1)
  K = wavelen / (4*np.pi)
  return K**2 * ((g0/d0)**2 + (g1*gd/d1)**2 +
                 2*(g0*g1*gd) / (d0*d1) * np.cos((d1-d0)/(2*K)))

isotropic_reader = Node(height=5, angle=np.pi/3, rp=isotropic_rp)
isotropic_tag = Node(height=0.5, angle=np.pi/2, rp=isotropic_rp)
dipole_reader = Node(height=5, angle=np.pi/2)
dipole_tag = Node(height=.5, angle=np.pi/2)

ox = np.arange(.1, 20, 0.1)
pl_los_iso = w2db(PL1(ox, isotropic_reader, isotropic_tag))
pl_2ray_iso = w2db(PL2(ox, isotropic_reader, isotropic_tag))
pl_2ray_dip = w2db(PL2(ox, dipole_reader, dipole_tag))

# print(pathloss_los)

fig = plt.figure()
ax = plt.subplot(111)
plt.plot(ox, pl_los_iso, 'y--', label='FSPL for isotropic antennas')
plt.plot(ox, pl_2ray_iso, 'r', label='2-Ray PL for isotropic antennas')
plt.plot(ox, pl_2ray_dip, 'b', label='2-Ray PL for dipole antennas')
ax.set_ybound(lower=-100, upper=-20)
plt.legend()


#%%
