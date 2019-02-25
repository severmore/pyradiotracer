from numpy import linalg as lin
import numpy
from shape import Ray

TOLERANCE = 1e-9

class RayTree:

  class _RayNode:

    def __init__(self, ray, parent=None):
      """ A ray tree node, `ray` is obj:`Ray` class instance. """
      self.ray = ray
      self.parent = parent
      self.children = []

    def __str__(self):
      return f'({self.ray})'

  def __init__(self, ray=None):
    """ A tree for storing rays while running tracer and after it terminates.
    The tree is assumed to spread downwards only. """
    self.root = None if ray is None else self._RayNode(ray)
    self.current = self.root
  

  def insert(self, ray):
    
    if self.current is None: # the first ray in a tree
      self.root = self._RayNode(ray)
      self.current = self.root
    
    else:
      node = self._RayNode(ray, self.current)
      self.current.children.append(node)
      self.current = node
    
    return self


class Tracer:

  def __init__(self, scene):
    self.scene = scene
  
  def run(self, tx_position, rx_position):

    forest = []

    # compute line-of-sight components
    ray = Ray(start=tx_position, end=rx_position)
    forest.append(RayTree(ray))

    # compute 1-reflected components
    for shape in self.scene:

      intersection, dir_grazing, dir_reflected = \
            shape.reflected_mirrow_path(tx_position, rx_position)
      
      if intersection == shape.inf: # no intersection
        continue
      
      ray_grazing = Ray(start=tx_position, end=intersection, direction=dir_grazing)
      ray_reflected = Ray(start=intersection, direction=dir_reflected)

      tree = RayTree(ray_grazing).insert(ray_reflected)
      forest.append(tree)

    return forest

