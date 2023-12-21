from typing import Optional 
from abc import ABC, abstractmethod 

from rl_env.grids_map.grids_node.node import Node


class Grids(ABC):
    def __init__(self) -> None:
        super().__init__()  
        self.start_node: Optional[Node] = None
        self.goal_node: Optional[Node] = None
    
        
        
    @abstractmethod
    def get_neighbors(self, node: "Node") -> list:
        pass    
    
    @abstractmethod
    def is_goal(self, node: "Node") -> bool:
        pass    
    
    @abstractmethod
    def _is_valid_node(self) -> bool:
        pass    
    
    @abstractmethod
    def __iter__(self):
        pass