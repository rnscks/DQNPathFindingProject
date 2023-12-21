from abc import ABC, abstractmethod

class Node(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.is_obstacle: bool = False    
        self.is_start: bool = False   
        self.is_goal: bool = False
        
        
    @abstractmethod
    def __eq__(self, other: "Node") -> bool:
        pass    
    
    @abstractmethod 
    def __hash__(self) -> int:
        pass
    
    @abstractmethod
    def __str__(self) -> str:
        pass
    
    @abstractmethod
    def __lt__(self) -> bool:
        pass
    
    
    