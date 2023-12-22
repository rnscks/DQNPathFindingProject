import unittest
import torch

from rl_env.env_util.grids_converter import TensorConvertor
from rl_env.env_util.grids_generator import RandomBoxGenerator3D
from rl_env.grids_map.grids3d import Grids3D


class TestTensorConvertor(unittest.TestCase):
    def setUp(self):
        self.grids3d: Grids3D = RandomBoxGenerator3D(10, 10).generate((1, 1, 1), (9, 9, 9))
        self.convertor = TensorConvertor(self.grids3d)
        

    def test_convert(self):
        tensor = self.convertor.convert()
        self.assertIsInstance(tensor, torch.FloatTensor)
        
    def test_tensor_dimension(self):
        tensor = self.convertor.convert()
        self.assertEqual(tensor.dim(), 5) # (batch, channel, depth, height, width)
        
    def test_tensor_shape(self):    
        tensor = self.convertor.convert()
        max_x, max_y, max_z = self.grids3d.map_size, self.grids3d.map_size, self.grids3d.map_size   
        self.assertEqual(tensor.shape, (1, 1, max_x, max_y, max_z))
        
    

if __name__ == "__main__":
    unittest.main()