#%%
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

  pl_model = radio.KRayPathloss(scene, frequency=860e6)

  distance = numpy.linspace(1,10,100)
  pathloss = numpy.zeros(100)

  for i, d in enumerate(distance):
    pl = pl_model(vec(0,0,5), vec(0,d,.5), max_reflections=1)
    pathloss[i] = radio.to_log(radio.power(pl))

  plt.figure(figsize=(4,3))
  plt.plot(distance, pathloss)
  plt.show()