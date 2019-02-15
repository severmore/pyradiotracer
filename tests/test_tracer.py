import unittest
import numpy
from radiotracer import shape
from radiotracer import tracer
from radiotracer import utils

vec = utils.vec3d
vec_s = utils.vec3d_sequnced

red    = utils.p_red    
green  = utils.p_green  
yellow = utils.p_yellow 
blue   = utils.p_blue   
turq   = utils.p_turq

view = utils.view

class BaseTracerTestCase(unittest.TestCase):

  def assertNdarrayListEqual(self, list1, list2, sep=', '):
    """ Replace `assertListEqual` function to compare list containing
    `ndarray` objects.
    """
    len1, len2 = len(list1), len(list2)

    self.assertEqual(
        len1,
        len2, 
        msg=f'\nThe lengths {red(len1)} and {red(len2)} mismatches for lists:\n'
            f'[{turq(view(list1, sep=sep))}]\n'
            f'[{turq(view(list2, sep=sep))}]'
    )

    for a1, a2 in zip(list1, list2):
      self.assertTrue(
        numpy.allclose(a1, a2),
        msg=f'\nThe elements {red(a1)} and {red(a2)} are not equal for lists:\n'
            f'[{turq(view(list1, sep=sep))}]\n'
            f'[{turq(view(list2, sep=sep))}]'
      )
  
  def assertPathsEqual(self, paths1, paths2):
    """ Call 'assertNdarrayListEqual' for each pair of paths from corresponding
    possitions in paths1 and paths2.
    """
    for p1, p2 in zip(paths1, paths2):
      self.assertNdarrayListEqual(p1, p2, sep='->')


### TESTS ITSELF ###

class TracerTestCase(BaseTracerTestCase):

  def _to_ndarray(self, paths):
    return [list(map(vec_s, p)) for p in paths]

  def test_empty_scene(self):
    paths = tracer.Tracer([])(vec(0,0,5), vec(0,10,5), max_reflections=3)
    self.assertEqual(len(paths), 1)
    self.assertPathsEqual(paths[0], [vec(0,0,5), vec(0,10,5)])

  def test_one_plane_three_reflections(self):
    reference = self._to_ndarray([
      [(0,0,5), (0,10,5)],
      [(0,0,5), (0,5,0), (0,10,5)],
    ])
    tracer_ = tracer.Tracer(
      shape.build({(0,0,0): (0,0,1)})
    )
    paths = tracer_(vec(0,0,5), vec(0,10,5), max_reflections=3)
    self.assertEqual(len(paths), 2)
    self.assertPathsEqual(paths, reference)

  def test_two_parallel_planes_three_reflections(self):
    reference = self._to_ndarray([
      [(0,0,5), (0,10,5)],
      [(0,0,5), (0,5,0), (0,10,5)],
      [(0,0,5), (0,5,10), (0,10,5)],
      [(0,0,5), (0,2.5,10), (0,7.5,0), (0,10,5)],
      [(0,0,5), (0,2.5,0), (0,7.5,10), (0,10,5)],
      [(0,0,5), (0,1.66666667,0), (0,5,10), (0,8.33333333,0), (0,10,5)],
      [(0,0,5), (0,1.66666667,10), (0,5,0), (0,8.33333333,10), (0,10,5)],
    ])
    tracer_ = tracer.Tracer(
      shape.build({
        (0,0, 0): (0,0,1),
        (0,0,10): (0,0,-1),
      })
    )
    paths = tracer_(vec(0,0,5), vec(0,10,5), max_reflections=3)
    self.assertEqual(len(paths), len(reference))
    self.assertPathsEqual(paths, reference)

  def test_unused_plane(self):
    reference = self._to_ndarray([
      [(0,0,5), (0,10,5)],
      [(0,0,5), (0,5,0), (0,10,5)],
    ])
    tracer_ = tracer.Tracer(
      shape.build({
        (0,0,0)  : (0,0,1),
        (0,0,-10): (0,0,1), # should be unused
      })
    )
    paths = tracer_(vec(0,0,5), vec(0,10,5), max_reflections=3)
    self.assertEqual(len(paths), 2)
    self.assertPathsEqual(paths, reference)
    
    

if __name__ == '__main__':
  unittest.main()