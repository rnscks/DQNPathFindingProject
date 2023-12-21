from random import randint

from rl_env.env_util.grids_util import GridsUtil
from rl_env.grids_map.grids3d import Grids3D    


class RandomBoxGenerator3D(GridsUtil):
    def __init__(self, map_size: int, box_number: int) -> None:
        """
        This Object using for generating a 3D grid map with random boxes.

        Args:
            map_size (int): The size of the map.
            box_number (int): The number of boxes to generate.
        """
        super().__init__()
        self.map_size: int = map_size   
        self.box_number: int = box_number   
        
        
    def generate(self, start_index: tuple[int, int, int], end_index: tuple[int, int, int]) -> Grids3D:
        """
        Generate a 3D grid map with random boxes.

        Args:
            start_index (tuple[int, int, int]): The starting index of the map.
            end_index (tuple[int, int, int]): The ending index of the map.

        Returns:
            Grids3D: The generated 3D grid map.
        """
        grids3d = Grids3D(self.map_size)
        grids3d.set_start_node(grids3d[start_index[0], start_index[1], start_index[2]])
        grids3d.set_goal_node(grids3d[end_index[0], end_index[1], end_index[2]])
        
        box_index_list = self.__box_index_list_generate(start_index, end_index) 
        self.__initialize_obstacle_by_box(grids3d, box_index_list)
        
        return grids3d
        
    def __box_index_list_generate(self, start_index: tuple[int, int, int], end_index: tuple[int, int, int]) -> list[tuple[int, int, int]]:
        """
        Generate a list of box indices.

        Args:
            start_index (tuple[int, int, int]): The starting index of the map.
            end_index (tuple[int, int, int]): The ending index of the map.

        Returns:
            list[tuple[int, int, int]]: The list of box indices.
        """
        box_index_list = []
        # box의 시작점과 끝점을 생성
        for _ in range(self.box_number * 2):
            # 최외곽 지역을 제외한 지역에서 장애물을 생성하도록 함
            i: int = randint(start_index[0] + 1, end_index[0] - 2)
            j: int = randint(start_index[1] + 1, end_index[1] - 2)
            k: int = randint(start_index[2] + 1, end_index[2] - 2)
            box_index_list.append((i, j, k))
        
        return box_index_list
    
    def __initialize_obstacle_by_box(self, grids3d: Grids3D, box_index_list: list[tuple[int, int, int]]) -> None:
        """
        Initialize the obstacles in the grid map based on the box indices.

        Args:
            grids3d (Grids3D): The 3D grid map.
            box_index_list (list[tuple[int, int, int]]): The list of box indices.
        """
        # box의 시작점과 끝점을 추출
        for i in range(0, len(box_index_list) - 1, 2):
            min_x, min_y, min_z = self.__get_box_min_coord(box_index_list[i], box_index_list[i + 1])
            max_x, max_y, max_z = self.__get_box_max_coord(box_index_list[i], box_index_list[i + 1])
            
            for x in range(min_x, max_x):
                for y in range(min_y, max_y):
                    for z in range(min_z, max_z):
                        grids3d[x, y ,z].is_obstacle = True
                        
        grids3d.start_node.is_obstacle = False
        grids3d.goal_node.is_obstacle = False                        
        return 
    
    def __get_box_min_coord(self, left_box_index, right_box_index):
        """
        Get the minimum coordinates of a box.

        Args:
            left_box_index (tuple[int, int, int]): The left box index.
            right_box_index (tuple[int, int, int]): The right box index.

        Returns:
            tuple[int, int, int]: The minimum coordinates of the box.
        """
        min_x = min(left_box_index[0], right_box_index[0])   
        min_y = min(left_box_index[1], right_box_index[1])   
        min_z = min(left_box_index[2], right_box_index[2])   

        return min_x, min_y, min_z
    
    def __get_box_max_coord(self, left_box_index, right_box_index):
        """
        Get the maximum coordinates of a box.

        Args:
            left_box_index (tuple[int, int, int]): The left box index.
            right_box_index (tuple[int, int, int]): The right box index.

        Returns:
            tuple[int, int, int]: The maximum coordinates of the box.
        """
        max_x = max(left_box_index[0], right_box_index[0])   
        max_y = max(left_box_index[1], right_box_index[1])   
        max_z = max(left_box_index[2], right_box_index[2])   

        return max_x, max_y, max_z