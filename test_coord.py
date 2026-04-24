import sys
import os
sys.path.append(os.path.abspath('RLSTCcode'))
sys.path.append(os.path.abspath('RLSTCcode/subtrajcluster'))
import traj, point, point_xy
from MDP import TrajRLclus

env = TrajRLclus('RLSTCcode/data/geolife_testdata', 'RLSTCcode/data/geolife_clustercenter', 'RLSTCcode/data/geolife_clustercenter')
t = env.trajsdata[0]
print("p.x:", t.points[0].x, "p.y:", t.points[0].y)
