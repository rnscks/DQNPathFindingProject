from typing import Optional

from rl_env.grids_map.grids_node.node import Node


class VNode(Node):
    def __init__(self, i: int, j: int, k: int) -> None:
        """
        This object represents a voxel node in a 3D grid map using pathfinding algorithm.   

        Args:
            i (int): The i-coordinate of the voxel node.
            j (int): The j-coordinate of the voxel node.
            k (int): The k-coordinate of the voxel node.
        """
        super().__init__()
        self.i: int = i
        self.j: int = j
        self.k: int = k
        
        self.f: float = 0.0
        self.g: float = 0.0
        
        self.parent: Optional[VNode] = None 
        
        
    def __eq__(self, other: object) -> bool:
        """
        Compare if two VNode objects are equal.

        Args:
            other (object): The object to compare with.

        Returns:
            bool: True if the two VNode objects are equal, False otherwise.
        """
        if isinstance(other, VNode):
            return self.i == other.i and self.j == other.j and self.k == other.k
        else:
            return False        
        
    def __hash__(self) -> int:  
        """
        Generate a hash value for the VNode object.

        Returns:
            int: The hash value of the VNode object.
        """
        return hash((self.i, self.j, self.k))
    
    def __str__(self) -> str:
        """
        Get a string representation of the VNode object.

        Returns:
            str: The string representation of the VNode object.
        """
        return f"voxel node ({self.i},{self.j},{self.k})"
    
    def __lt__(self, other: "VNode") -> bool:
        """
        Compare if the current VNode object is less than another VNode object.

        Args:
            other (VNode): The other VNode object to compare with.

        Returns:
            bool: True if the current VNode object is less than the other VNode object, False otherwise.
        """
        return self.f < other.f
    
    