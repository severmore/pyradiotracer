#%%
import sys
if sys.stdin.isatty():
  import matplotlib
  matplotlib.use('TkAgg')

from matplotlib import pyplot as plt

try:
  from jupyterthemes import jtplot
  jtplot.style(theme='monokai', fscale=0.8, figsize=(8,6))
except:
  pass

import numpy
from ratracer import radio

if __name__ == '__main__':
  from ratracer.utils import vec3d as vec

  scene = {
    (0,0, 0): (0,0,1),
    # (0,0,10): (0,0,-1),
    # (5,0,0): (-1,0,0),
    # (-5,0,0): (1,0,0),
  }

  tx = radio.RFDevice(vec(0,0,5), 860e6, ant_normal=vec(0,1,0))
  rx = radio.RFDevice(vec(0,10,.5), 860e6, ant_normal=vec(0,-1,0))
  pl_model = radio.KRayPathloss(scene, frequency=860e6)
  def log_att(d, maxr):
    return radio.to_log(radio.power(
        pl_model(tx, rx.update_pos(y=d), max_reflections=maxr)
      ))

  distance = numpy.linspace(.1,20,1000)
  pl_los  = [log_att(d, maxr=0) for d in distance]
  pl_2ray = [log_att(d, maxr=1) for d in distance]
  tx.set_pattern(radio.AntennaPattern(kind='dipole'))
  rx.set_pattern(radio.AntennaPattern(kind='dipole'))
  pl_2ray_dipole = [log_att(d, maxr=1) for d in distance]

  plt.figure()
  ax = plt.subplot(111)
  plt.plot(distance, pl_los, label='only LoS-component with isotropic antenna')
  plt.plot(distance, pl_2ray, label='2-ray with isotropic antenna')
  plt.plot(distance, pl_2ray_dipole, label='2-ray with dipole antenna')
  ax.set_ybound(upper=-30, lower=-90)
  plt.legend()
  plt.show()

#%%