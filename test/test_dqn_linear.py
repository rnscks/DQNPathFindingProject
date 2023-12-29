import unittest
import torch
from dqn_network.dqn_linear import LinearNetwork

class TestLinearNetwork(unittest.TestCase):
    def setUp(self):
        self.input_size = 10
        self.output_size = 5
        self.network = LinearNetwork(self.input_size, self.output_size)

    def test_forward(self):
        x = torch.randn(1, self.input_size)
        output = self.network.forward(x)
        self.assertEqual(output.shape, torch.Size([1, self.output_size]))

    def test_relu1(self):
        x = torch.randn(1, self.input_size)
        output = self.network.relu1(x)
        self.assertEqual(output.shape, torch.Size([1, self.input_size]))

    def test_fc1(self):
        x = torch.randn(1, self.input_size)
        output = self.network.fc1(x)
        self.assertEqual(output.shape, torch.Size([1, self.output_size]))

    def test_fc2(self):
        x = torch.randn(1, self.output_size)
        output = self.network.fc2(x)
        self.assertEqual(output.shape, torch.Size([1, self.output_size]))

if __name__ == "__main__":
    unittest.main()