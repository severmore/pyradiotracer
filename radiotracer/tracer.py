import numpy
import itertools
from radiotracer import utils, shape

inf = utils.inf
reversed_enumerate = utils.reversed_enumerate
_SHADOWING_INDENT = .99999

# coloring routines
p_red    = utils.p_red    
p_green  = utils.p_green  
p_yellow = utils.p_yellow 
p_blue   = utils.p_blue   
p_turq   = utils.p_turq

view = utils.view

 

class Tracer:
  """ Image reflection method """

  def __init__(self, scene):
    """ Performs ray tracing of a given `scene` """
    self.scene = {i: scene[i] for i in range(len(scene))}
    self._scene_ids = {id_ for id_ in self.scene}
    self._scene_size = len(scene)
    self._paths = []
    self.scene[numpy.inf] = shape.Empty()
    # print(scene, self.scene)
  
  def __call__(self, tx, rx, max_reflections=2):
    """ Build rays pathes from `tx` to `rx` with at most `reflection_num` 
    reflections.
    """
    self._paths = []

    for k in range(max_reflections + 1):
      for sid_sequence in self._all_shape_ids_sequences(shapes_num=k):
          
        print(f'{p_green("[run]")} tracing {p_green(k)} reflections, shapes {p_green([self.scene[sid] for sid in sid_sequence])}')

        path = self.trace_path(tx, rx, sid_sequence)
        if path:
          self._paths.append(path)
          print(f'{p_green("[run]")} traced path for {p_green(k)} reflections is {p_green(view(path))}')

    return self._paths


  def _all_shape_ids_sequences(self, shapes_num):
    """ Generate all possible valid sequences of shape IDs to perform tracing.
    All sequences having two or equal consecutive shapes (IDs of these shapes) 
    are illigal, and thus should be missed.
    """
    def has_no_equal_consecutives(sid_sequence):
      it, it_shifted = itertools.tee(sid_sequence)
      next(it_shifted, None)
      return not any(id1 == id2 for id1, id2 in zip(it, it_shifted))

    return filter(has_no_equal_consecutives, 
                  itertools.product(self._scene_ids, repeat=shapes_num))

  
  def trace_path(self, start, end, sid_sequence):
    """ Build a path from `start` to `end` via reflecting from `shapes` """
    images = self._i_compute_images(end, sid_sequence)
    print(f'{p_turq("[trace_path]")} images: {p_turq(", ".join([str(i) for i in images]))}')
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

    print(f'[_i_compute_ray_path] shapes {sid_sequence}')

    for i, sid in reversed_enumerate(sid_sequence):

      print(f'[_i_compute_ray_path] shape no {i}, intersect shape {p_yellow(self.scene[sid])} with image {p_red(images[i])}')
      i_point = self.scene[sid].intersect(start, images[i])
      print(f'[_i_compute_ray_path] intersection point is {p_blue(i_point)}', flush=True)
      if numpy.array_equal(i_point, inf) or self.is_shadowed(start, i_point):
        print('Is it really true?', p_red(True))
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
  
  tracer_ = Tracer(
    shape.build({
      (0,0, 0): (0,0,1),
      (0,0,-10): (0,0,1),
    })
  )

  print(f'starting tracing from {utils.vec3d(0,0,5)} to {utils.vec3d(0,10,5)}')
  
  paths = tracer_(
    utils.vec3d(0,0,5), 
    utils.vec3d(0,10,5), 
    max_reflections=3
  )


  for p in paths:
    print(p_green(len(p) - 2), view(p))