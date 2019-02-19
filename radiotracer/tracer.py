import numpy
import itertools

from radiotracer import shape
from radiotracer.utils import reversed_enumerate, product_no_consecutives, inf
from radiotracer.utils import view, verbose_routine, COLORS

_SHADOWING_INDENT = .99999

class Tracer:
  """ Image reflection method """

  def __init__(self, scene):
    """ Performs ray tracing of a given `scene` """
    self.scene = {shape.id: shape for shape in scene}
    self._sids = {id_ for id_ in self.scene}
    self._scene_size = len(scene)
    self._paths = None
    self.scene[numpy.inf] = shape.Empty()
  
  def shapes(self, sid_sequence):
    return [self.scene[sid] for sid in sid_sequence]
  
  def __call__(self, tx, rx, max_reflections=2, verbose=0):
    """ Build rays pathes from `tx` to `rx` with at most `reflection_num` 
    reflections.
    """
    self._paths = []

    for k in range(max_reflections + 1):
      for sid_sequence in product_no_consecutives(self._sids, repeat=k):
        path = self._trace_path(tx, rx, sid_sequence, verbose)
        if path:
          self._paths.append(path)

    return self._paths

  
  def _trace_path(self, start, end, sid_sequence, verbose):
    """ Build a path from `start` to `end` via reflecting from `shapes` """
    _print_shapes(verbose, self.shapes(sid_sequence))
    images = self._i_compute_images(end, sid_sequence)
    _print_images(verbose, images)

    return self._i_compute_ray_path(start, images, sid_sequence, verbose)
  

  def _i_compute_images(self, point, sid_sequence):
    """ Compute a sequence of images of `point` by subsequently reflecting from
    `shapes` in backward direction starting with the endpoint """
    images = [point]
    for sid in sid_sequence:
      images.append(self.scene[sid].reflect(images[-1]))
    return images


  def _i_compute_ray_path(self, start, images, sid_sequence, verbose):
    """ Build path component in forward direction """
    path = [start]
    sid_sequence = (numpy.inf,) + sid_sequence

    for i, sid in reversed_enumerate(sid_sequence):

      ipoint = self.scene[sid].intersect(start, images[i])
      _print_intersection(verbose, self.scene[sid], images[i], ipoint)
      
      if numpy.array_equal(ipoint, inf) or self.is_shadowed(start, ipoint):
        _print_is_shadowed(verbose)
        return None
      
      start = ipoint
      path.append(ipoint)

    _print_traced_path(verbose, path)
    return path
  

  def is_shadowed(self, start, end):
    """ Check if ray with `start` and `end` shadowed by any shape of a scene """
    # Indent from end to exclude shadowing by the shape that forms this ray.
    end = start + _SHADOWING_INDENT * (end - start)
    return any(shape.is_shadowed(start, end) for shape in self.scene.values())


### VERBOSE ROUTINES

@verbose_routine
def _print_shapes(verbose, shapes, color=None):
  green = lambda s: color(s, COLORS['green']) 
  print(f'{green("[tracing] " + str(len(shapes)))} reflections'
        f' from {green(shapes)}')

@verbose_routine
def _print_images(verbose, images, color=None):
  turq = lambda s: color(s, COLORS['turq'])
  print(f'images: {turq(", ".join([str(i) for i in images]))}')

@verbose_routine
def _print_intersection(verbose, shape, image, ipoint, color=None):
  blue = lambda s: color(s, COLORS['blue'])
  print(f'intersect {blue(shape)} ({shape.id}) with '
        f'image {blue(image)} at {blue(ipoint)}')

@verbose_routine
def _print_is_shadowed(verbose, color=None):
  red = lambda s: color(s, COLORS['red'])
  print(f'{red("shadowed or not intersected")}')

@verbose_routine
def _print_traced_path(verbose, path, color=None):
  green = lambda s: color(s, COLORS['green']) 
  print(f'traced path is {green(view(path))}')


if __name__ == '__main__':

  from utils import vec3d as vec

  scene = {
    (0,0, 0): (0,0,1),
    (0,0,10): (0,0,-1),
    (5,0,0): (-1,0,0),
    (-5,0,0): (1,0,0),
  }

  tracer_ = Tracer(shape.build(scene))
  paths = tracer_(vec(0,0,5), vec(0,10,5), max_reflections=1, verbose=2)

  for p in paths: print(len(p) - 2, view(p))