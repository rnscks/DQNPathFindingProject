from OCC.Core.gp import gp_Pnt
from OCC.Core.TopoDS import TopoDS_Shape    
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse

from rl_env.env_util.grids_util import GridsUtil
from rl_env.grids_map.grids3d import Grids3D


class TopoDSShapeConvertor(GridsUtil):
    def __init__(self, grids3d: Grids3D) -> None:
        """
        This object for converting grids3d map to TopoDS(OCCT).

        Args:
            grids3d (Grids3D): The 3D grid map.
        """
        super().__init__()
        self.grids3d: Grids3D = grids3d 
        
        
    def convert(self, corner_min: gp_Pnt, corner_max: gp_Pnt) -> TopoDS_Shape:
        """
        Converts the grid map into a fused TopoDS_Shape.

        Args:
            start_pnt (gp_Pnt): The grids box's starting point.
            end_pnt (gp_Pnt): The grids box's ending point.
        Returns:
            TopoDS_Shape: The fused TopoDS_Shape for final return.
        """
        gap: float = corner_min.Distance(corner_max)    
        gap /= self.grids3d.map_size    
        
        fused_shape: TopoDS_Shape = TopoDS_Shape()
        
        for node in self.grids3d:
            if node.is_obstacle:                        
                x, y, z = node.i, node.j, node.k    
                voxel_shape: TopoDS_Shape = self.__get_voxel_shape(gap, corner_min, x, y, z) 
                fused_shape = self.__fuse_voxel_shape(fused_shape, voxel_shape)                    
                    
        return fused_shape
    
    def __get_voxel_shape(self, gap: float, start_pnt: gp_Pnt, x: int, y: int, z: int) -> TopoDS_Shape:    
        """
        Creates a voxel shape.

        Args:
            gap (float): The gap between voxels.
            start_pnt (gp_Pnt): Grids box's starting point using calculate initial point.
            x (int): The x-coordinate of the voxel.
            y (int): The y-coordinate of the voxel.
            z (int): The z-coordinate of the voxel.

        Returns:
            TopoDS_Shape: The voxel shape.
        """
        min_x, min_y, min_z = start_pnt.X() + x * gap, start_pnt.Y() + y * gap, start_pnt.Z() + z * gap
        max_x, max_y, max_z = min_x + gap, min_y + gap, min_z + gap   
        
        conrner_min, conrner_max = gp_Pnt(min_x, min_y, min_z), gp_Pnt(max_x, max_y, max_z) 
        
        voxel_shape = BRepPrimAPI_MakeBox(conrner_min, conrner_max).Shape()
        
        return voxel_shape

    
    def __fuse_voxel_shape(self,fused_shape: TopoDS_Shape, voxel_shape: TopoDS_Shape) -> TopoDS_Shape:
        """
        Fuses the voxel shape with the fused shape.

        Args:
            fused_shape (TopoDS_Shape): The fused shape.
            voxel_shape (TopoDS_Shape): The voxel shape.

        Returns:
            TopoDS_Shape: The fused shape.
        """
        if fused_shape.IsNull():
            fused_shape = voxel_shape
        else:
            fused_shape = BRepAlgoAPI_Fuse(fused_shape, voxel_shape).Shape()
            
        return fused_shape
        

