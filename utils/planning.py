import math
import cv2
import matplotlib.pyplot as plt
import numpy as np
from utils.geometry import compute_distance



######################## Astar planner ##########################  
#
# The implementation of the AstarPlanner class is based on
#      https://github.com/AtsushiSakai/PythonRobotics
#
#################################################################
class AStarPlanner:

    def __init__(self, ox, oy, reso, rr,h,w):
        """
        Initialize grid map for a star planning

        ox: x position list of Obstacles
        oy: y position list of Obstacles 
        reso: grid resolution
        rr: robot radius
        h,w: dimensions of the input mask
        """

        self.reso = reso
        self.rr = rr
        self.h=h
        self.w=w
        self.calc_obstacle_map(ox, oy,h,w)
        self.motion = self.get_motion_model()

    class Node:
        def __init__(self, x, y, cost, pind):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.pind = pind

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.pind)

    def planning(self, sx, sy, gx, gy,img_index):
        """
        input:
            sx: start x position
            sy: start y position 
            gx: goal x position
            gy: goal y position 

        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """

        nstart = self.Node(self.calc_xyindex(sx), self.calc_xyindex(sy), 0.0, -1)
        ngoal = self.Node(self.calc_xyindex(gx),self.calc_xyindex(gy), 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(nstart)] = nstart

        while 1:
            if len(open_set) == 0:
                print(img_index,"Open set is empty..")
                break

            c_id = min(
                open_set,
                key=lambda o: open_set[o].cost + self.calc_heuristic(ngoal,
                                                                     open_set[
                                                                         o]))
            current = open_set[c_id]


            if current.x == ngoal.x and current.y == ngoal.y:
                #goal reached
                ngoal.pind = current.pind
                ngoal.cost = current.cost
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # expand_grid search grid based on motion model
            for i, _ in enumerate(self.motion):
                node = self.Node(current.x + self.motion[i][0],
                                 current.y + self.motion[i][1],
                                 current.cost + self.motion[i][2], c_id)
                n_id = self.calc_grid_index(node)

                # If the node is not safe, do nothing
                if not self.verify_node(node):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # discovered a new node
                else:
                    if open_set[n_id].cost > node.cost:
                        # This path is the best until now. record it
                        open_set[n_id] = node

        rx, ry = self.calc_final_path(ngoal, closed_set)

        return rx, ry

    def calc_final_path(self, ngoal, closedset):
        # generate final course
        rx, ry = [ngoal.x*self.reso], [ngoal.y*self.reso]
        pind = ngoal.pind
        while pind != -1:
            n = closedset[pind]
            rx.append(n.x*self.reso)
            ry.append(n.y*self.reso)
            pind = n.pind

        return rx, ry

    @staticmethod
    def calc_heuristic(n1, n2):     
        #euclidean works better than manhattan distance heuristic
        w = 10000  # weight of heuristic 
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d


    def calc_xyindex(self, position):
        return round(position / self.reso)

    def calc_grid_index(self, node):
        return node.y * self.xwidth + node.x 

    def verify_node(self, node):
        px = node.x*self.reso
        py = node.y*self.reso

        if px < 0:
            return False
        elif py < 0:
            return False
        elif px >= self.w:
            return False
        elif py >= self.h:
            return False

        # collision check
        if (self.obmap[node.x][node.y]==1):
            #print("obstacle")
            return False
        
        #print("no obs")
        return True


    def calc_obstacle_map(self, ox, oy,h,w):
        
        self.xwidth = round(w / self.reso)
        self.ywidth = round(h / self.reso)
        #print("mappa init")
        #obsta map generation
        self.obmap = [[False for i in range(self.ywidth)] for i in range(self.xwidth)]
        for iox, ioy in zip(ox, oy):
            self.obmap[iox][ioy]=True
            #xgrid,ygrid= self.calc_xyindex(iox),self.calc_xyindex(ioy) #handle case =0 or h,w
            #for xx in range(xgrid-1,xgrid+1):
             #   for yy in range(ygrid-4,ygrid+6):
              #      self.obmap[xx][yy]=True
                    #print(xx,yy)

    @staticmethod
    def get_motion_model():
        # dx, dy, cost
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [-1, -1, math.sqrt(2)],
                  [-1,1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)],
                  [1, 1, math.sqrt(2)]]

        return motion


######################## Plan the path, given the ordered wp ########################    
    
def plan_path(wp,a_star,smooth=False,img_index=0):
    path = np.empty((0,2),"int")
    for index in range (len(wp)-1):
        rx, ry = a_star.planning((wp[index,0]),(wp[index,1]),(wp[index+1,0]),(wp[index+1,1]),img_index)
        rx = np.flip(rx,axis=0)
        ry = np.flip(ry,axis=0)
        p = np.stack([rx,ry],axis=-1)
        path = np.append(path,p,axis=0)
    if smooth:
        path=smooth_path(path)
    return path



######################## Path smoothing ########################    


def smooth_path(path, weight_data=0.2, weight_smooth=0.7, tolerance=0.000001):
    """
        parameters:
        path: List containing coordinates of a path
        weight_data: how much weight to update the data (alpha)
        weight_smooth:how much weight to smooth the coordinates (beta).
        tolerance: how much change per iteration is necessary to keep iterating.

    """
    path = path.astype("float")
    smoothed = path.copy()
    dims = len(path[0])
    change = tolerance
    k = 0
    
    while change >= tolerance:
        change = 0.0
        for i in range(1, len(smoothed) - 1):
            for j in range(dims):
                x_i = path[i][j]
                y_i, y_prev, y_next = smoothed[i][j], smoothed[i - 1][j], smoothed[i + 1][j]

                y_i_saved = y_i.copy()
                y_i += weight_data * (x_i - y_i) + weight_smooth * (y_next + y_prev - (2 * y_i))
                smoothed[i][j] = y_i

                change += abs(y_i - y_i_saved)
        k += 1

    return smoothed.astype("int")



############################ Coverage metric #####################################

def field_coverage(mask,path,wp,clusters,target_wp,return_tp=False):
    """
        Computes the field covergae metric.
    """
    total_true_positive=0
    index=np.where(clusters[1:]!=clusters[:-1])[0][0]
    start_wp_index = np.where(np.all(path==wp[index],axis=1))[0][0]
    end_wp_index = np.where(np.all(path==wp[index+1],axis=1))[0][0]
    points_old = np.linspace(start_wp_index,end_wp_index,6,dtype="int")[1:-1][::-1]

    for index in range(index+1,len(wp)-1):
        if clusters[index]==clusters[index+1]:
            continue
        start_wp_index = np.where(np.all(path==wp[index],axis=1))[0][0]
        end_wp_index = np.where(np.all(path==wp[index+1],axis=1))[0][0]
        points_new= np.linspace(start_wp_index,end_wp_index,6,dtype="int")[1:-1]
        for j in range(4):
            a = path[points_old[j]]; b = path[points_new[j]];
            l = (compute_distance(a,b)+1).astype("int")
            xy = np.round(np.linspace(a,b,l)).astype("int")
            occ = mask[xy[:,1],xy[:,0]]
            if occ.any():
                total_true_positive+=1
                break
        points_old = points_new[::-1]

    n_rows = len(target_wp)//2
    if return_tp:
        return total_true_positive/(n_rows-1),total_true_positive
    else:
        return total_true_positive/(n_rows-1)

