from typing import Optional

from rl_env.grids_map.grids import Grids    
from rl_env.grids_map.grids_node.vnode import VNode    


class Grids3D(Grids):
    """
    This Object represents a 3D grid map for pathfinding algorithm.
    
    Attributes:
        map_size (int): The size of the map.
        nodes_map3d (list[list[list[VNode]]]): A 3D list representing the nodes in the map.
    """
    def __init__(self, map_size: int) -> None:
        super().__init__()  
        self.map_size: int = map_size   
        self.nodes_map3d: list[list[list[VNode]]] = self.__get_initialized_node_list()    
        
    
    def get_neighbors(self, node: VNode) -> list:
        """
        Returns a list of neighboring nodes for a given node.
        
        Args:
            node (VNode): The node to get the neighbors for.
        
        Returns:
            list: A list of neighboring nodes.
        """
        ret_neighbors: list[VNode] = []
        
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    if i == 0 and j == 0 and k == 0:
                        continue
                    
                    if self._is_valid_node(node.i + i, node.j + j, node.k + k):
                        ret_neighbors.append(self.nodes_map3d[node.i + i][node.j + j][node.k + k])
        
        return ret_neighbors 
    
    def set_start_node(self, node: VNode) -> None:
        """
        Sets the start node of the map.
        
        Args:
            node (VNode): The node to set as the start node.
        """
        self.start_node = node
        node.is_start = True
        return
    
    def set_goal_node(self, node: VNode) -> None:   
        """
        Sets the goal node of the map.
        
        Args:
            node (VNode): The node to set as the goal node.
        """
        self.goal_node = node
        node.is_goal = True
        return  
    
    def is_goal(self, node: VNode) -> bool:
        """
        Checks if a given node is the goal node.
        
        Args:
            node (VNode): The node to check.
        
        Returns:
            bool: True if the node is the goal node, False otherwise.
        """
        return node == self.goal_node
        
    def _is_valid_node(self, i: int, j: int, k: int):   
        """
        Checks if a given node is a valid node in the map.
        
        Args:
            i (int): The x-coordinate of the node.
            j (int): The y-coordinate of the node.
            k (int): The z-coordinate of the node.
        
        Returns:
            bool: True if the node is valid, False otherwise.
        """
        if(i < 0 or j < 0 or k < 0):
            return False    
        elif (i >= self.map_size or j >= self.map_size or k >= self.map_size):
            return False    
        elif (self.nodes_map3d[i][j][k].is_obstacle):
            return False    
        
        return True
    
    def __get_initialized_node_list(self) -> list[list[list[VNode]]]:
        """
        Initializes the 3D list of nodes in the map.
        
        Returns:
            list[list[list[VNode]]]: The initialized 3D list of nodes.
        """
        ret_nodes_map3d: list[list[list[VNode]]] = []
        
        for i in range(self.map_size):
            ret_nodes_map3d.append([])
            for j in range(self.map_size):
                ret_nodes_map3d[i].append([])
                for k in range(self.map_size):
                    ret_nodes_map3d[i][j].append(VNode(i, j, k))   
                    
        return ret_nodes_map3d
    
    def __is_out_of_range(self, i: int, j: int, k: int) -> bool:    
        """
        Checks if a given coordinate is out of range of the map.
        
        Args:
            i (int): The x-coordinate.
            j (int): The y-coordinate.
            k (int): The z-coordinate.
        
        Returns:
            bool: True if the coordinate is out of range, False otherwise.
        """
        if (i < 0 or j < 0 or k < 0 or i >= self.map_size or j >= self.map_size or k >= self.map_size):
            return True
        return False
    
    def __iter__(self):
        """
        Iterates over all the nodes in the map.
        """
        for i in range(self.map_size):
            for j in range(self.map_size):
                for k in range(self.map_size):
                    yield self.nodes_map3d[i][j][k]    
                    
    def __getitem__(self, index: tuple[int, int, int]) -> VNode:
        """
        Gets the node at the specified index.
        
        Args:
            index (tuple[int, int, int]): The index of the node.
        
        Returns:
            VNode: The node at the specified index.
        
        Raises:
            IndexError: If the index is out of range.
        """
        if (not self.__is_out_of_range(index[0], index[1], index[2])):
            return self.nodes_map3d[index[0]][index[1]][index[2]]
        else:
            raise IndexError("Index out of range")  
        
    def __len__(self):
        """
        Returns the length of the map.
        
        Returns:
            int: The length of the map.
        """
        return len(self.nodes_map3d)