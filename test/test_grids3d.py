import unittest
from rl_env.grids_map.grids3d import Grids3D
from rl_env.grids_map.grids_node.vnode import VNode


class TestGrids3D(unittest.TestCase):
    def setUp(self):
        self.grid = Grids3D(3)
        self.node = VNode(1, 1, 1)
        self.grid.set_goal_node(self.grid[2, 2, 2])
        self.grid[1, 1, 1].is_obstacle = True

    def test_get_neighbors(self):
        neighbors = self.grid.get_neighbors(self.node)
    
        expected_neighbors = [
            self.grid.nodes_map3d[0][0][0],
            self.grid.nodes_map3d[0][0][1],
            self.grid.nodes_map3d[0][0][2],
            self.grid.nodes_map3d[0][1][0],
            self.grid.nodes_map3d[0][1][1],
            self.grid.nodes_map3d[0][1][2],
            self.grid.nodes_map3d[0][2][0],
            self.grid.nodes_map3d[0][2][1],
            self.grid.nodes_map3d[0][2][2],
            self.grid.nodes_map3d[1][0][0],
            self.grid.nodes_map3d[1][0][1],
            self.grid.nodes_map3d[1][0][2],
            self.grid.nodes_map3d[1][1][0],
            self.grid.nodes_map3d[1][1][2],
            self.grid.nodes_map3d[1][2][0],
            self.grid.nodes_map3d[1][2][1],
            self.grid.nodes_map3d[1][2][2],
            self.grid.nodes_map3d[2][0][0],
            self.grid.nodes_map3d[2][0][1],
            self.grid.nodes_map3d[2][0][2],
            self.grid.nodes_map3d[2][1][0],
            self.grid.nodes_map3d[2][1][1],
            self.grid.nodes_map3d[2][1][2],
            self.grid.nodes_map3d[2][2][0],
            self.grid.nodes_map3d[2][2][1],
            self.grid.nodes_map3d[2][2][2]
        ]
        self.assertEqual(neighbors, expected_neighbors)

    def test_is_goal(self):
        self.assertFalse(self.grid.is_goal(self.node))
        self.assertTrue(self.grid.is_goal(self.grid.goal_node))

    def test__is_valid_node(self):
        self.assertFalse(self.grid._is_valid_node(-1, 0, 0))
        self.assertFalse(self.grid._is_valid_node(0, -1, 0))
        self.assertFalse(self.grid._is_valid_node(0, 0, -1))
        self.assertFalse(self.grid._is_valid_node(3, 0, 0))
        self.assertFalse(self.grid._is_valid_node(0, 3, 0))
        self.assertFalse(self.grid._is_valid_node(0, 0, 3))
        self.assertFalse(self.grid._is_valid_node(1, 1, 1))
        self.assertTrue(self.grid._is_valid_node(0, 0, 0))

    def test___get_initialized_node_list(self):
        self.assertEqual(len(self.grid), 3)
        self.assertIsInstance(self.grid[0, 0, 0], VNode)
        self.assertEqual(self.grid[0, 1, 2].i, 0)
        self.assertEqual(self.grid[0, 1, 2].j, 1)
        self.assertEqual(self.grid[0, 1, 2].k, 2)


    def test___iter__(self):
        for node in self.grid:
            self.assertIsInstance(node, VNode)
            self.assertTrue(node.i >= 0 and node.i < 3)
            self.assertTrue(node.j >= 0 and node.j < 3)
            self.assertTrue(node.k >= 0 and node.k < 3) 



if __name__ == "__main__":
    unittest.main()