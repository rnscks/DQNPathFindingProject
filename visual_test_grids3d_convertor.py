from OCC.Core.gp import gp_Pnt
from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Display.SimpleGui import init_display

from rl_env.env_util.grids_converter import TopoDSShapeConvertor
from rl_env.env_util.grids_generator import RandomBoxGenerator3D
from rl_env.grids_map.grids3d import Grids3D


map_size, box_number = 10, 10
grids: Grids3D = RandomBoxGenerator3D(map_size, box_number).generate((0, 0, 0), (9, 9, 9))   
start_pnt, end_pnt = gp_Pnt(0, 0, 0), gp_Pnt(10, 10, 10)    
fused_shape: TopoDS_Shape = TopoDSShapeConvertor(grids).convert(start_pnt, end_pnt)

display, start_display, add_menu, add_function_to_menu = init_display()

display.DisplayShape(fused_shape, update=True)
display.FitAll()
start_display()