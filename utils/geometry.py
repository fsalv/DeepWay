import numpy as np


##################### Points #####################

def compute_distance(p1,p2=np.array([0,0])):
    """
        Euclidean distance between p1 and p2
    """
    p1 = np.array(p1)
    p2 = np.array(p2)   
    return np.sqrt(np.sum((p1-p2)**2))



def random_displ(max_pix=5,min_pix=0,shape=None):
    """
        Get random displacements between min_pix and max_pix
    """
    disp = np.random.randint(min_pix,max_pix,shape)
    disp *= np.random.choice([-1,1],shape)
    return disp



##################### Lines #####################

def line_polar(p1,p2):
    """
        Compute distance coordinates and angle for the line connecting p1 and p2
    """
    dy = (p2[1]-p1[1])
    dx = (p2[0]-p1[0])
    alpha = np.arctan2(dy,dx)
    l = np.arange(compute_distance(p1,p2)+1)
    return l,alpha



def line_polar_to_cart(l,alpha,point=(0,0)):
    """
        Line points from polar to cartesian coordinates
    """
    return point[0]+l*np.cos(alpha),point[1]+l*np.sin(alpha)



def find_intersect(alpha1,p1,alpha2,p2,ret_xy=True):
    """
        Compute intersection between two lines in polar coordinates. Returns either x,y of the intersection or 
        its distance from p1 along line1
    """
    if alpha1 == alpha2:
        if ret_xy:
            return np.inf,np.inf
        else:
            return np.inf
    
    p1 = np.array(p1)
    p2 = np.array(p2)
    
    p21 = p2-p1
    l_inter = (np.tan(alpha2)*p21[0]-p21[1])/(np.tan(alpha2)*np.cos(alpha1)-np.sin(alpha1))
    if ret_xy:
        return l_inter*np.cos(alpha1)+p1[0],l_inter*np.sin(alpha1)+p1[1]
    else:
        return l_inter



##################### Other #####################
    
    
def PolyArea(x,y):
    """
        Compute area of polygon defined by x,y points 
    """
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
    
