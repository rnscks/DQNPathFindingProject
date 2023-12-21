import unittest
from rl_env.env_util.grids_generator import RandomBoxGenerator3D
from rl_env.grids_map.grids3d import Grids3D


class TestRandomBoxGenerator3D(unittest.TestCase):
    def test_generate(self):
        map_size, box_number = 10, 5    
        generator = RandomBoxGenerator3D(map_size, box_number)
        start_index, goal_index = (1, 1, 1), (8, 8, 8)
        grids3d = generator.generate(start_index, goal_index)
        
        self.assertIsInstance(grids3d, Grids3D)
        self.assertEqual(grids3d.start_node, grids3d[start_index[0], start_index[1], start_index[2]])
        self.assertEqual(grids3d.goal_node, grids3d[goal_index[0], goal_index[1], goal_index[2]])
        
    def test_is_box_in_grids(self):
        map_size, box_number = 10, 5    
        generator = RandomBoxGenerator3D(map_size, box_number)
        start_index, goal_index = (1, 1, 1), (8, 8, 8)
        grids3d = generator.generate(start_index, goal_index)
        
        is_box_in_grids = False
        for node in grids3d:
            if (node.is_obstacle == True):
                is_box_in_grids = True
                break
        
        self.assertTrue(is_box_in_grids)    
        
            

if __name__ == "__main__":
    unittest.main()