import unittest
import numpy
from ratracer import shape
from ratracer import tracer
from ratracer import utils
from ratracer import radio
from ratracer import simple

vec = utils.vec3d
vec_s = utils.vec3d_sequnced

red = utils._apply_color('red')
turq = utils._apply_color('turq')

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
            f'[{turq(tracer.view(list1, sep=sep))}]\n'
            f'[{turq(tracer.view(list2, sep=sep))}]'
    )

    for a1, a2 in zip(list1, list2):
      self.assertTrue(
        numpy.allclose(a1, a2),
        msg=f'\nThe elements {red(a1)} and {red(a2)} are not equal for lists:\n'
            f'[{turq(tracer.view(list1, sep=sep))}]\n'
            f'[{turq(tracer.view(list2, sep=sep))}]'
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
    trace = tracer.Tracer([])(vec(0,0,5), vec(0,10,5), max_reflections=3)
    self.assertEqual(len(trace), 1)
    self.assertPathsEqual(next(trace.paths()), [vec(0,0,5), vec(0,10,5)])

  # @unittest.skip
  def test_one_plane_three_reflections(self):
    reference = self._to_ndarray([
      [(0,0,5), (0,10,5)],
      [(0,0,5), (0,5,0), (0,10,5)],
    ])
    tracer_ = tracer.Tracer(
      shape.build({(0,0,0): (0,0,1)})
    )
    trace = tracer_(vec(0,0,5), vec(0,10,5), max_reflections=3)
    self.assertEqual(len(trace), 2)
    self.assertPathsEqual(list(trace.paths()), reference)

  # @unittest.skip
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
    trace = tracer_(vec(0,0,5), vec(0,10,5), max_reflections=3)
    self.assertEqual(len(trace), len(reference))
    self.assertPathsEqual(list(trace.paths()), reference)

  # @unittest.skip
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
    trace = tracer_(vec(0,0,5), vec(0,10,5), max_reflections=3)
    self.assertEqual(len(trace), 2)
    # self.assertPathsEqual(list(trace.paths()), reference)


class AttenuationEvaluationReceiverApproachingTestCase(unittest.TestCase):

  class AttFormatter:
    def __init__(self, scene, txrx_specs):

      self.tx = radio.RFDevice(
        numpy.array(txrx_specs['tx'][0]),
        txrx_specs['freq'],
        ant_normal=numpy.array(txrx_specs['tx'][1])
      )
      self.rx = radio.RFDevice(
        numpy.array(txrx_specs['rx'][0]),
        txrx_specs['freq'],
        ant_normal=numpy.array(txrx_specs['rx'][1])
      )
      self.model = radio.KRayPathloss(scene, frequency=txrx_specs['freq'])

    def __call__(self, d, maxr):
      att = self.model(self.tx, self.rx.update_pos(y=d), max_reflections=maxr)
      return radio.to_log(radio.power(att))

    def change_antenna(self, pattern):
      self.tx.set_pattern(pattern)
      self.rx.set_pattern(pattern)

  GRID_RESOLUTION = 4

  def setUp(self):
    # Ray-tracing-enabled part
    scene = {
      (0,0, 0): (0,0,1),
    }
    txrx_specs = {
      'tx': [(0,0,5), (0,1,0)],
      'rx': [(0,10,.5), (0,-1,0)],
      'freq': 860e6
    }
    self.freq = txrx_specs['freq']
    self.formatter = self.AttFormatter(scene, txrx_specs)

    # Predefined-model-based computation part
    self.model = simple.Pathloss(self.freq)
    self.reader = simple.Node(height=5)
    self.tag = simple.Node(height=0.5)

    self.distance = numpy.linspace(.1, 20, self.GRID_RESOLUTION)

  def test_only_los_communications(self):
    # Computing 1-ray pathloss by ray-tracing model
    pathloss = [self.formatter(d, maxr=0) for d in self.distance]

    # Computing 1-ray pathloss by predefined model (no ray-tracing)
    self.reader.rp = simple.isotropic_rp
    self.tag.rp = simple.isotropic_rp
    pathloss2 = simple.w2db(self.model.los(self.distance, self.reader, self.tag))

    self.assertTrue(numpy.allclose(pathloss, pathloss2))

  def test_2ray_with_isotropic_antenna(self):
    # Computing 1-ray pathloss by ray-tracing model
    pathloss = [self.formatter(d, maxr=1) for d in self.distance]

    # Computing 1-ray pathloss by predefined model (no ray-tracing)
    self.reader.rp = simple.isotropic_rp
    self.tag.rp = simple.isotropic_rp
    pathloss2 = simple.w2db(self.model.tworay(self.distance, self.reader, self.tag))

    self.assertTrue(numpy.allclose(pathloss, pathloss2))

  def test_2ray_with_dipole_antenna(self):
    # Computing 1-ray pathloss by ray-tracing model
    self.formatter.change_antenna(radio.AntennaPattern(kind='dipole'))
    pathloss = [self.formatter(d, maxr=1) for d in self.distance]

    # Computing 1-ray pathloss by predefined model (no ray-tracing)
    self.reader.rp = simple.dipole_rp
    self.tag.rp = simple.dipole_rp
    pathloss2 = simple.w2db(self.model.tworay(self.distance, self.reader, self.tag))

    print('traced', pathloss)
    print('predefined', pathloss2)
    self.assertTrue(numpy.allclose(pathloss, pathloss2))


if __name__ == '__main__':
  unittest.main()