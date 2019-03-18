"""
'ratracer.tracer' contains an engine for ray-tracing based on mirrow reflection
method. An example of usage:

  from ratracer import shape, tracer

  scene = {
    (0,0, 0): (0,0,1),
    (0,0,10): (0,0,-1),
    (5,0,0): (-1,0,0),
    (-5,0,0): (1,0,0),
  }
  tr = tracer.Tracer(shape.build(scene))
  paths, shapes_ids = tr(vec(0,0,5), vec(0,10,5), max_reflections=2)

:class:`ratracer.tracer.Tracer` class implement MR-method. Note, such method is
useful to compute scene with specular reflecting shapes.
"""
import itertools
import numpy

from ratracer import shape, settings
from ratracer.utils import reversed_enumerate, product_no_consecutives, inf
from ratracer.utils import verbose, normalize, norm

DIR_CODE = 0
LEN_CODE = 1
SID_CODE = 2
AOA_CODE = 3

class Tracer:
  """ Implements mirrow reflection method. Contains methods which are as
  follows:

    - :func:`__call__` - run ray-tracing for a scene
    - :func:`clear` - clear results of previous tracer call; when tracer is
        called, cleaning routine run automatically
    - :func:`get_shapes` - return generator that produces shapes stored in
        `self.scene` by the sequence of specified ids
    - :func:`is_shadowed` - check if specified ray intersect any object of
        scene.
  """
  class Result:

    def __init__(self):
      """ The class to store results of ray tracing and to provide access to it.
      """
      self.trace = []
      self._temp = ([], [], [], [])
      self.start = None
      self.end = None

    def __str__(self):
      paths = ('-'.join(str(bp) for bp in path) for path in self.paths())
      return "\n".join(paths)

    def __len__(self):
      return len(self.trace)

    def append(self, *point):
      """ Append point traced right now to the building path. """
      for p, v in zip(self._temp, point):
        p.append(v)

    def append_last(self, *last_point):
      """ Append the last traced point of the ray path. """
      self.append(*last_point)  # zip will skip `sid` and `aoa` parameters
      self.save()
      return self.trace[-1]

    def clear(self):
      """ Clear currently built path. """
      self._temp = ([], [], [], [])

    def path(self, idx):
      breaks = [self.start]
      for i in range(self.path_len(idx)):
        direction = self.trace[idx][DIR_CODE][i]
        length = self.trace[idx][LEN_CODE][i]
        breaks.append(breaks[i] + direction * length)
      return breaks

    def paths(self):
      """ Reconstruct breakpoint of traced paths. """
      return (self.path(idx) for idx in range(len(self.trace)))

    def path_len(self, idx):
      """ Return the length of :arg:`idx`-th path traced. """
      return len(self.trace[idx][0])

    def refresh(self, start, end):
      """ Clear all path being built before. """
      self.trace.clear()
      self.clear()
      self.start = start
      self.end = end

    def save(self):
      """ Save building path. """
      if self._temp[0]:   # any of dir~ and len~ lists are possible
        self.trace.append(self._temp)
        self.clear()


  def __init__(self, scene):
    """ Performs ray tracing of a given :param:`scene`. Scene itself is a
    sequence of shapes located in module :mod:'ratracer.shape'. The tracing
    based on mirrow reflection method and it is subject to specular relfections.

    Parameters
    __________
    scene : list of `shape.Shape`
        A list of shape forming the scene to trace
    etype : subclass of `shape.Empty` (optional)
        A class of empty shape being used inside tracing engine to skip some
        operations with shapes. Default value is `shape.Empty`.

    See Also
    ________
    .shape.build_shapes
    .shape.Plane
    .shape.Empty
    """
    self.scene = {shape.id: shape for shape in scene}
    self.result = Tracer.Result()

  def __call__(self, start, end, max_reflections=2):
    """ Build rays pathes from :arg:`start` to :arg:`end` with at most
    :arg:`max_reflections` reflections.

    Parameters
    __________
    start : 3-`np.ndarray`
        Starting point for rays
    end : 3-`np.ndarray`
        Ending point for rays
    max_reflections : int, optional
        Restict rays with a maximum number of reflection

    Returns
    _______
    result: obj:`.Tracer.Result`
        result object storing traced ray paths parameters:
          - sigments direction
          - length
          - cosine of angle of arriving (aoa)
          - shape ID from which a ray reflects
    """
    self.result.refresh(start, end)

    for k in range(max_reflections + 1):
      for sid_sequence in product_no_consecutives(self.scene, repeat=k):
        self._trace_path(start, end, sid_sequence)

    return self.result


  def _trace_path(self, start, end, sid_sequence):
    """ Build a path from `start` to `end` via reflecting from `shapes` """
    _print__shapes(list(self.get_shapes(sid_sequence))) # TODO: take preparation into _print__
    images = self._i_compute_images(end, sid_sequence)
    _print__images(images)
    path = self._i_compute_ray_path(start, end, images, sid_sequence)
    _print__traced_path(path, start)


  def _i_compute_images(self, point, sid_sequence):
    """ Compute a sequence of images of `point` by subsequently reflecting from
    `shapes` in backward direction starting with the endpoint """
    images = [point]
    for sid in sid_sequence:
      images.append(self.scene[sid].reflect(images[-1]))
    return images[1:]


  def _i_compute_ray_path(self, start, end, images, sid_sequence):
    """ Build path component in forward direction """
    for i, sid in reversed_enumerate(sid_sequence):

      direction = normalize(images[i] - start)
      length, aoa = self.scene[sid].intersect(start, direction)
      delta = direction * length
      if length is numpy.nan or self.is_shadowed(start, delta):
        self.result.clear()
        return None

      start = start + delta
      self.result.append(direction, length, sid, aoa)
      _print__intersection(self.scene[sid], images[i], start)

    delta = end - start
    length = norm(delta)
    direction = delta / length
    if self.is_shadowed(start, delta):
      self.result.clear()
      return None

    return self.result.append_last(direction, length)

  def get_shapes(self, sid_sequence):
    """ Get a generator of shapes stored in :param:`self.scene` by its ids. """
    return (self.scene[sid] for sid in sid_sequence)

  def is_shadowed(self, start, delta):
    """ Check if ray with `start` and `end` shadowed by any shape of a scene """
    return any(shape.is_shadowing(start,delta) for shape in self.scene.values())



def view(path, *, sep='->'):
  """ Forms string representation of path. """
  if path is None:
    return 'None'
  return sep.join([str(point) for point in path])


### VERBOSE ROUTINES

@verbose(color='green')
def _print__shapes(shapes, *, color):
  print(f'{color("[tracing] " + str(len(shapes)))} reflections'
        f' from {color(shapes)}')

@verbose(color='turq')
def _print__images(images, *, color):
  print(f'images: {color(", ".join([str(i) for i in images]))}')

@verbose(color='blue')
def _print__intersection(shape, image, ipoint, *, color):
  print(f'intersect {color(shape)} ({shape.id}) with '
        f'image {color(image)} at {color(ipoint)}')

@verbose(color='green')
def _print__traced_path(path, start, *, color):
  if not path:
    print(f'None is traced')
  else:
    breaks = [start]
    for d, l in zip(path[0], path[1]):
      start = start + d*l
      breaks.append(start)
    print(f'traced path is {color(view(breaks))}')



if __name__ == '__main__':
  from ratracer.utils import vec3d as vec

  scene = {
    (0,0, 0): (0,0,1),
    (0,0,10): (0,0,-1),
    # (5,0,0): (-1,0,0),
    # (-5,0,0): (1,0,0),
  }

  tracer_ = Tracer(shape.build(scene))
  result = tracer_(vec(0,0,5), vec(0,10,5), max_reflections=1)

  print(result)

  # @verbose('red')
  # def _sids(sids, *, color): return f'{color(view(sids, sep=" "))}'
  # def _shapes(sids): return view([tracer_.scene[s] for s in sids], sep=", ")

  # for p, s in zip(paths, shapes):
  #   print(f'{len(s)} {view(p)},\t{_sids(s)},\t{_shapes(s)}', sep='\n')
