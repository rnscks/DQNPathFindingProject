import unittest
import torch
from dqn_network.dqn_cnn3d import CNN3D

class TestCNN3D(unittest.TestCase):
    def setUp(self):
        self.map_size = 10
        self.action_size = 26   
        self.model = CNN3D(self.map_size)

    def test_forward(self):
        input_tensor = torch.randn(1, 1, self.map_size, self.map_size, self.map_size)
        output = self.model.forward(input_tensor)
        self.assertEqual(output.shape, (1, self.action_size))

    def test_conv1_output_shape(self):
        input_tensor = torch.randn(1, 1, self.map_size, self.map_size, self.map_size)
        output = self.model.conv1(input_tensor)
        self.assertEqual(output.shape, (1, 32, self.map_size, self.map_size, self.map_size))

    def test_conv2_output_shape(self):
        input_tensor = torch.randn(1, 1, self.map_size, self.map_size, self.map_size)
        output = self.model.conv2(input_tensor)
        self.assertEqual(output.shape, (1, 32, self.map_size, self.map_size, self.map_size))

    def test_fc_output_shape(self):
        input_tensor = torch.randn(1, 1, self.map_size, self.map_size, self.map_size)
        output = self.model.fc(torch.flatten(self.model.conv2(self.model.conv1(input_tensor))))
        self.assertEqual(output.shape, (1, self.action_size))

if __name__ == "__main__":
    unittest.main()