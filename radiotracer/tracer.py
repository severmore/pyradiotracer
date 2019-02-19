import numpy
import itertools

from radiotracer import shape
from radiotracer.utils import reversed_enumerate, product_no_consecutives, inf
from radiotracer.utils import view, p_red, p_green, p_yellow, p_blue, p_turq

_SHADOWING_INDENT = .99999


class Tracer:
  """ Image reflection method """

  def __init__(self, scene):
    """ Performs ray tracing of a given `scene` """
    self.scene = {shape.id: shape for shape in scene}
    self._sids = {id_ for id_ in self.scene}
    self._scene_size = len(scene)
    self._paths = []
    self.scene[numpy.inf] = shape.Empty()
  
  def __call__(self, tx, rx, max_reflections=2):
    """ Build rays pathes from `tx` to `rx` with at most `reflection_num` 
    reflections.
    """
    self._paths = []

    for k in range(max_reflections + 1):
      for sid_sequence in product_no_consecutives(self._sids, repeat=k):
          
        print(f'{p_green("[run]")} tracing {p_green(k)} reflections, shapes {p_green([self.scene[sid] for sid in sid_sequence])}')

        path = self.trace_path(tx, rx, sid_sequence)
        if path:
          self._paths.append(path)
          print(f'{p_green("[run]")} traced path is {p_green(view(path))}')

    return self._paths

  
  def trace_path(self, start, end, sid_sequence):
    """ Build a path from `start` to `end` via reflecting from `shapes` """
    images = self._i_compute_images(end, sid_sequence)
    return self._i_compute_ray_path(start, images, sid_sequence)
  

  def _i_compute_images(self, point, sid_sequence):
    """ Compute a sequence of images of `point` by subsequently reflecting from
    `shapes` in backward direction starting with the endpoint """
    images = [point]
    for sid in sid_sequence:
      images.append(self.scene[sid].reflect(images[-1]))
    return images


  def _i_compute_ray_path(self, start, images, sid_sequence):
    """ Build path component in forward direction """
    path = [start]
    sid_sequence = (numpy.inf,) + sid_sequence

    print(f'images: {p_turq(", ".join([str(i) for i in images]))}')
    print(f'shapes {sid_sequence}')

    for i, sid in reversed_enumerate(sid_sequence):

      i_point = self.scene[sid].intersect(start, images[i])
      print(f'intersect {p_yellow(self.scene[sid])}({sid}) with image {p_blue(images[i])} at {p_blue(i_point)}')
      if numpy.array_equal(i_point, inf) or self.is_shadowed(start, i_point):
        print(f'{p_red("Shadowed or not intersected")}')
        return None
      
      start = i_point
      path.append(i_point)
    
    return path
  

  def is_shadowed(self, start, end):
    """ Check if ray with `start` and `end` shadowed by any shape of a scene """
    # Indent from end to exclude shadowing by the shape that forms this ray.
    end = start + _SHADOWING_INDENT * (end - start)
    return any(shape.is_shadowed(start, end) for shape in self.scene.values())


if __name__ == '__main__':

  from utils import vec3d as vec

  scene = {
    (0,0, 0): (0,0,1),
    (0,0,10): (0,0,-1),
    (5,0,0): (-1,0,0),
    (-5,0,0): (1,0,0),
  }

  tracer_ = Tracer(shape.build(scene))
  paths = tracer_(vec(0,0,5), vec(0,10,5), max_reflections=2)

  for p in paths:
    print(p_green(len(p) - 2), view(p))