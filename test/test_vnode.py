import unittest
from rl_env.grids_map.grids_node.vnode import VNode 


class TestVNode(unittest.TestCase):
    def test_equal_nodes(self):
        node1 = VNode(1, 1, 1)
        node2 = VNode(1, 1, 1)
        self.assertEqual(node1, node2)

    def test_not_equal_nodes(self):
        node1 = VNode(1, 1, 1)
        node2 = VNode(2, 2, 2)
        self.assertNotEqual(node1, node2)

    def test_hash(self):
        node1 = VNode(1, 1, 1)
        node2 = VNode(1, 1, 1)
        self.assertEqual(hash(node1), hash(node2))

    def test_less_than(self):
        node1 = VNode(1, 1, 1)
        node1.f = 1
        node2 = VNode(2, 2, 2)
        node2.f = 2
        self.assertLess(node1, node2)

    def test_to_string(self):
        node = VNode(1, 2, 3)
        self.assertEqual(str(node), "voxel node (1,2,3)")


if __name__ == "__main__":
    unittest.main()