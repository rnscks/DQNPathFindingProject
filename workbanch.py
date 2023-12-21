from rl_env.env_util.grids_generator import RandomBoxGenerator3D


grids3d = RandomBoxGenerator3D(10, 10).generate((1, 1, 1), (9, 9, 9))

