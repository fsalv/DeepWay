import os
import cv2
import numpy as np
import random
import scipy.stats as stats

from utils.geometry import *


##################### Borders generation #####################

def gen_p(start=40,limit=760):
    """
    Generate a number with exponentially deacying probability between start and limit
    """
    p = (np.random.exponential(2,1)+1)*start
    if p>limit:
        return gen_p(start,limit)
    else:
        return p[0]

    

def gen_borders(border=40,H=800,W=800):
    """
    Generate linear random borders (50% straight, 50% with random points on the border frame)
    """
    border = (float)(border)
    if np.random.uniform()<0.5:
        border_points = (gen_p(border,W-border),H-gen_p(border,H-border),W-gen_p(border,W-border),gen_p(border,H-border))
        border_points = [(border_points[0],border),(border,border_points[1]),
                         (border_points[2],H-border),(W-border,border_points[3])]
    else:
        border_points = (gen_p(border,H-border),gen_p(border,W-border),H-gen_p(border,H-border),W-gen_p(border,W-border))
        border_points = [(border,border_points[0]),(border_points[1],H-border),
                         (W-border,border_points[2]),(border_points[3],border)]
        
    border_points.append(border_points[0])
    border_points = np.array(border_points)
    
    alpha = []
    for i in range(4):
        dy = (border_points[i+1][1]-border_points[i][1])
        dx = (border_points[i+1][0]-border_points[i][0])
        alpha.append(np.arctan2(dy,dx))
    
    return [(a,p) for a,p in zip(alpha,border_points[:-1])]



def check_borders_area(borders,reference_area,fraction=0.65):
    """
    Compute area inside border and compare it with a reference value
    """
    x = [b[1][0] for b in borders]
    y = [b[1][1] for b in borders]
    return (PolyArea(x,y)/reference_area)>fraction



def intersect_border(alpha,p,border):
    """
    Find intersection point of a line with the border (given as distance l from point p with angle alpha)
    """
    return find_intersect(alpha,p,border[0],border[1],ret_xy=False)



##################### Synthetic points generation #####################

def gen_start_and_end(alpha,center,borders=40,H=800,W=800,angle_var=0.005,border_var=40):
    """
    Find line starting and ending points (intersections with borders) 
    """
    alpha = -alpha # convert alpha to get the right view in imshow mode
    alpha += stats.truncnorm(-2,2,loc=0, scale = angle_var).rvs(1)[0]
    
    l = []
    for border in borders:
        l.append(intersect_border(alpha,center,border))
    
    l = np.array(l)
    lmax = np.min(l[l>0])
    lmin = np.max(l[l<0])
    
    l1 = lmax - border_var/2 + stats.truncnorm(-2,2, loc=0, scale=border_var/2).rvs(1)[0]
    l2 = lmin + border_var/2 + stats.truncnorm(-2,2, loc=0, scale=border_var/2).rvs(1)[0]
        
    x1 = int(round(l1*np.cos(alpha)+center[0]))
    y1 = int(round(l1*np.sin(alpha)+center[1]))
    x2 = int(round(l2*np.cos(alpha)+center[0]))
    y2 = int(round(l2*np.sin(alpha)+center[1]))

    return ((x1,y1),(x2,y2)) 



def find_intrarow_distance(nrows,alpha,borders,p=(400,400),lower_limit=12):
    """
    Compute intra-row distance Q randomly in a range depending on nrows
    """
    alpha = alpha - np.pi/2       # perpendicular to the orientation
    (x1,y1),(x2,y2) = gen_start_and_end(alpha,p,borders=borders,angle_var=0,border_var=0)
    dist1 = compute_distance((x1,y1),p)
    dist2 = compute_distance((x2,y2),p)
    dist = min(dist1,dist2)
    Qmax = 2*(dist-5)/(nrows-0.3)   # heuristic formula to have at least 5 pixels between the border and the last possible point
    
    if Qmax > lower_limit:
        Q = np.random.uniform(lower_limit,Qmax)
    else:
        Q = lower_limit
        nrows = 2*int(dist/Q)    # too many rows with the current border, reduce them
        
    return Q,nrows



def find_centers(nrows,alpha,image_center=(400,400),Q=20):
    """
    Find rows pivot points starting from image_center with intra-row distance Q
    """
    alpha = -alpha         # convert alpha to get the right view in imshow mode (y -> -y)
    
    l = np.arange(nrows/2-0.5,-nrows/2,-1)*Q+random_displ(Q/5,shape=nrows) # random displacement of Q/5 along the perpendicular line
    x_c = l*np.cos(alpha-np.pi/2)+image_center[0]
    y_c = l*np.sin(alpha-np.pi/2)+image_center[1]

    return [(x,y) for x,y in zip(x_c,y_c)] 



##################### Mask creation #####################

def get_row_line(p1,p2):
    """
    Compute x,y values and alpha angle for the line between start (p1) and ending (p2) points
    """
    l,alpha = line_polar(p1,p2)
    x,y = line_polar_to_cart(l,alpha,p1)
    return (np.round(x).astype("int"),np.round(y).astype("int"),alpha)



def generate_holes(row_len,hole_prob=0.5,hole_dim=[2,4],hole_frame=12):
    """
        Randomly generate holes in the row. For each x,y a hole in range hole_dim is generated with hole_prob.
        If a hole is too close (within hole_frame) to starting/ending points of the line, the hole is extended
        to avoid border effects. Returns the boolean mask to select final row points.
    """
    indexes = np.ones(row_len) # points mask
    
    holes_pos = np.random.choice([0,1],row_len,p=(1-hole_prob/100,hole_prob/100))
    holes_pos = np.where(holes_pos == 1)[0]

    for h in holes_pos:
        hole = random.randint(hole_dim[0],hole_dim[1])  #the hole dimension is random
        indexes[h:h+hole]=0
    
    if len(holes_pos):
        if holes_pos[0]<=hole_frame:          #avoid to leave few points at the beginning
            indexes[:holes_pos[0]]=0
        if holes_pos[-1]>=row_len-hole_frame: #avoid to leave few points at the end
            indexes[holes_pos[-1]:]=0
    
    return indexes.astype("bool")



def create_mask(points,H=800,W=800,radius=[3,4],hole_prob=0.5,hole_dim=[4,6],hole_frame=12):
    """
        Compute x-y points of all the rows and draw them in the mask
    """
    mask = np.ones((H,W),"float")
    
    row_lines = []
    for p in points:
        row_line = get_row_line(p[0],p[1])
        indexes = generate_holes(len(row_line[0]),hole_prob,hole_dim,hole_frame)
        row_line = row_line[0][indexes],row_line[1][indexes],row_line[2]

        for x,y in zip(row_line[0],row_line[1]):
            cv2.circle(mask,(x,y),random.randint(radius[0],radius[1]),color=0,thickness=-1)
        
        row_lines.append(row_line)
    
    return mask.astype("bool"),row_lines



##################### Waypoints creation #####################

def gen_wp(line1,line2,index=0):
    """
        Compute wp between two adjacent lines from the points in index position (0: line starting point, -1: line ending point)
    """
    p0 = (line1[0][index],line1[1][index])
    p1 = (line2[0][index],line2[1][index])
    
    mx = np.mean((p0[0],p1[0]))               # middle point x
    my = np.mean((p0[1],p1[1]))               # middle point y
    alpha = np.mean((line1[-1],line2[-1]))    # mean angle
        
    dist = compute_distance(p0,p1)
    l = dist/2
    
    if index<0:  # at the end of the line, we should move in the opposite direction
        l = -l
    
    x,y = line_polar_to_cart(l,alpha,(mx,my))    
    return (int(round(x)),int(round(y)))




def gen_waypoints(row_lines):
    """
        Generate wp for all the rows
    """
    waypoints = []
    for row in range(1,len(row_lines)):    # no wp before the first and after the last row
        waypoints.append(gen_wp(row_lines[row-1],row_lines[row],index=0))
        waypoints.append(gen_wp(row_lines[row-1],row_lines[row],index=-1))
    return waypoints



    
##################### Ground truths managing #####################
      

def get_points(file,img_shape,mirror=False):
    """
        Read wp from a YOLO style file: each wp are the upper-left and the lower-right points of the bounding box
    """
    points = []
    img_shape = img_shape[:-1][::-1]
    file = open(file).read().split('\n')[:-1]
    for r in file:
        r = r.split()
        center = (float(r[1])*img_shape[0],float(r[2])*img_shape[1])
        width = float(r[3])*img_shape[0]
        height = float(r[4])*img_shape[1]
        if mirror:   # depends on orientation of rows
            p1 = round(center[0]+width/2),round(center[1]-height/2)
            p2 = round(center[0]-width/2),round(center[1]+height/2)
        else:
            p1 = round(center[0]-width/2),round(center[1]-height/2)
            p2 = round(center[0]+width/2),round(center[1]+height/2)
        points.append((p1,p2))
    return points




def rescale_points(points,pad,pad_ax,point_pad_ax,r):
    """
        Compute wp positions after image padding and rescaling
    """
    points = np.array(points.copy())
    points[:,:,point_pad_ax] += pad[pad_ax][0]
    return np.round(points/r).astype("int")



def rescale_img(img,points=None,H=800,W=800):
    """
        Rescale the image and the wp to the target dimension (PLEASE USE INTEGER W/H RATIOS ONLY)
    """
    ratio = W/H

    w_new = img.shape[0]*ratio
    h_new = img.shape[1]/ratio

    p = np.max([h_new - img.shape[0],w_new - img.shape[1]])
    pad_ax = np.argmax([h_new - img.shape[0],w_new - img.shape[1]])
    pad = [(0,0)]*3
    pad[pad_ax] = ((int)(np.ceil(p/2)),(int)(np.floor(p/2)))

    img2 = np.pad(img,pad)
    img2 = cv2.resize(img2,(W,H),interpolation=cv2.INTER_AREA)
    img2 = np.clip(img2, 0, 1)
    
    if points is not None:
        if pad_ax:
            r = img.shape[0]/H
            point_pad_ax = 0
        else:
            r = img.shape[1]/W
            point_pad_ax = 1
            
        points = rescale_points(points,pad,pad_ax,point_pad_ax,r) 
        return img2,points
    return img2



##################### Others #####################


def random_zoom(mask,wp,centers,points,zoom_ratio=[80,100],H=800,W=800):
    """
       Apply random out-zoom to the generated mask. Max zoom_ratio must be <= 100
    """  
    ratio = stats.truncnorm((zoom_ratio[0]-100)/5,(zoom_ratio[1]-100)/5,loc=100,scale=5).rvs(1)[0]/100

    H_reshaped,W_reshaped = (int)(H*ratio),(int)(W*ratio)
    mask_reshaped = np.clip(cv2.resize(mask.astype("uint8"),(W_reshaped,H_reshaped),interpolation=cv2.INTER_NEAREST),0,1)
    
    #white padding around the parcel
    delta = (H-H_reshaped,W-W_reshaped)
    pad_y = ((int)(np.ceil(delta[0]/2)),(int)(np.floor(delta[0]/2)))
    pad_x = ((int)(np.ceil(delta[1]/2)),(int)(np.floor(delta[1]/2)))
    padding = (pad_y,pad_x)
    
    mask_reshaped = np.pad(mask_reshaped,padding,'constant', constant_values=1.)

    # waypoints transformation
    wp_reshaped = np.round(np.array(wp)*ratio+(pad_x[0],pad_y[0])).astype("int")
    centers_reshaped = np.round(np.array(centers)*ratio+(pad_x[0],pad_y[0])).astype("int")
    points_reshaped = np.round(np.array(points)*ratio+(pad_x[0],pad_y[0])).astype("int")
    
    return mask_reshaped, wp_reshaped, centers_reshaped, points_reshaped



def save_img(img,num,extension='png',data_path="mask_datasets"):
    """
       Save the mask as uint8 image
    """  
    final_path = os.path.join(data_path,'img{}.{}'.format(num,extension))
    cv2.imwrite(final_path,(img*255).astype("uint8"))
    
    
    