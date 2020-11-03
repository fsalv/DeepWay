import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import cv2

from utils.geometry import line_polar_to_cart

##################### Network #####################

def plotImages(x, y, img_n, WAYP_VALUE = 255):
    """
    BATCH Take as input a batch from the generator and plt a number of images equal to img_n
    Default columns equal to max_c. At least inputs of batch equal two
    """
    max_c = 5
    
    if img_n <= max_c:
        r = 1
        c = img_n
    else:
        r = math.ceil(img_n/max_c)
        c = max_c
        
    fig, axes = plt.subplots(r, c, figsize=(15,15))
    axes = axes.flatten()
    for x, y, ax in zip(x, y, axes):
        canvas = x.copy()[...,None].astype('uint8')
        canvas = np.concatenate((canvas,canvas,canvas), axis=-1)
        row, col = np.where(y==WAYP_VALUE)
        for r, c in zip(row, col):
            canvas = cv2.circle(canvas, (c,r), 3, (50,255,250), -1)
        ax.imshow(cv2.bitwise_not(canvas))
        ax.grid()
    plt.tight_layout()
    plt.show()
    
def plotHistory(history):
    """
    Plot the loss and accuracy curves for training and validation 
    """
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.show()
    
def plotData(x, y, WAYP_VALUE = 255):
    """
    NO BATCH Plot mask and waypoints in a single image
    """
    plt.figure(figsize=(12,12))
    canvas = x.copy()[...,None].astype('uint8')
    canvas = np.concatenate((canvas,canvas,canvas), axis=-1)
    row, col = np.where(y==WAYP_VALUE)
    for r, c in zip(row, col):
        canvas = cv2.circle(canvas, (c,r), 2, (50,255,250), -1)
    plt.imshow(cv2.bitwise_not(canvas))
    plt.show()
    
def plotDataClass(x, y, WAYP_VALUE = 255):
    """
    NO BATCH Plot mask and waypoints in a single image
    """
    plt.figure(figsize=(12,12))
    canvas = x.copy()[...,None].astype('uint8')
    canvas = np.concatenate((canvas,canvas,canvas), axis=-1)
    row, col = np.where(y==WAYP_VALUE)
    row_c, col_c = np.where(y==127)
    for r, c in zip(row, col):
        canvas = cv2.circle(canvas, (c,r), 2, (50,255,250), -1)
    for r, c in zip(row_c, col_c):
        canvas = cv2.circle(canvas, (c,r), 2, (255,255,0), -1)
    plt.imshow(cv2.bitwise_not(canvas))
    plt.show()
    
def plotDataAside(x, y_pred, y_true, WAYP_VALUE = 255):
    """
    NO BATCH Plot mask and waypoints in a single image for GT and Pred
    """
    fig, ax = plt.subplots(1, 2, figsize=(18,18))
    canvas_pred = x.copy()[...,None].astype('uint8')
    canvas_pred = np.concatenate((canvas_pred,canvas_pred,canvas_pred), axis=-1)
    row, col = np.where(y_pred==WAYP_VALUE)
    for r, c in zip(row, col):
        canvas_pred = cv2.circle(canvas_pred, (c,r), 2, (50,255,250), -1)
    ax[0].imshow(cv2.bitwise_not(canvas_pred))
    ax[0].set_title('Prediction')
    
    canvas_gt = x.copy()[...,None].astype('uint8')
    canvas_gt = np.concatenate((canvas_gt,canvas_gt,canvas_gt), axis=-1)
    row, col = np.where(y_true==WAYP_VALUE)
    for r, c in zip(row, col):
        canvas_gt = cv2.circle(canvas_gt, (c,r), 2, (50,255,250), -1)
    ax[1].imshow(cv2.bitwise_not(canvas_gt))
    ax[1].set_title('Ground Truth')

def plotDataRes(index, x, y):
    """
    NO BATCH but index
    """
    plt.figure(figsize=(12,12))
    plt.imshow(cv2.resize(x[index], (y.shape[1], y.shape[2])))
    plt.imshow(y[index], alpha=(0.5), cmap='ocean')
    plt.show()

    
    
    
##################### Masks #####################

def visualize_mask(mask=None,points=None,wp=None,wp_class=None,centers=None,borders=None,path=None,rad=6,dim=(7,7),axis=True):
    """
        Visualize a binary mask (0:rows, 1:background) and possibly starting/ending points, wp, centers and borders
    """
    if mask is None:
        img = np.ones((800,800,3)).astype("float32")
    elif not(len(mask.shape)==3 and mask.shape[-1]==3):
        img = np.tile(mask[...,None],3).astype("float32")
    else:
        img=mask.copy()
    if borders is not None:
        img = plot_border(img,borders,show_img=False)
    if points is not None:
        for p in points:
            cv2.circle(img,(p[0][0],p[0][1]),rad,color=(1,0,0),thickness=-1)
            cv2.circle(img,(p[1][0],p[1][1]),rad,color=(0,0.5,1),thickness=-1)
    if wp is not None:
        for i,p in enumerate(wp):
            if wp_class is not None:
                if wp_class[i]:
                    color = (0,1,0)
                else:
                    color = (0,1,1)
            else:
                color = (0,204/255,0)
            cv2.circle(img,(p[0],p[1]),rad,color=color,thickness=-1)
    if centers is not None:
        for p in centers:
            cv2.circle(img,(int(round(p[0])),int(round(p[1]))),rad,color=(1,0.8,0),thickness=-1)   
    show(img,dim=dim,path=path,axis=axis,markersize=rad)
    


def visualize_image_with_mask(img,mask=None,points=None,wp=None,path=None,rad=6,dim=(7,7),axis=True):
    """
        Visualize the binary mask (0:rows, 1:background) together with the original image
    """
    img2 = img.copy()
    if mask is not None:
        img2[np.bitwise_not(mask.astype("bool"))] = 1   
    if points is not None:
        for p in points:
            cv2.circle(img2,(p[0][0],p[0][1]),rad,color=(1,0,0),thickness=-1)
            cv2.circle(img2,(p[1][0],p[1][1]),rad,color=(0,0,1),thickness=-1)
    if wp is not None:
        for p in wp:
            cv2.circle(img2,(p[0],p[1]),rad,color=(0,1,0),thickness=-1)        
    show(img2,path=path,dim=dim,axis=axis,markersize=rad)


    

def visualize_points(points,centers,H=800,W=800,dim=(7,7)):
    """
        Visualize centers and starting/ending points used to generate the mask
    """
    fig = np.ones((H,W,3),"float")

    for c in centers:
        c = (int(round(c[0])),int(round(c[1])))
        cv2.circle(fig,(c[0],c[1]),5,color=(0,1,0),thickness=-1)

    for p1,p2 in points:
        cv2.circle(fig,(p1[0],p1[1]),5,color=(1,0,0),thickness=-1)
        cv2.circle(fig,(p2[0],p2[1]),5,color=(0,0,1),thickness=-1)
    show(fig,dim)
                     


def draw_line(img,x,y,radius=3,color=(0,0,0)):
    """
        Draw a line in xy coordinates
    """
    img2 = img.copy()
    
    x,y = np.round(x).astype("int"),np.round(y).astype("int")
    for i,j in zip(x,y):
        cv2.circle(img2,(i,j),radius,color,thickness=-1)
    return img2


                     
def plot_border(img,lines,show_img=True):
    """
        Add borders to image: lines should be a list of (alpha,point)
    """
    img2 = img.copy()

    for l in lines:
        r = np.arange(800)
        x,y = line_polar_to_cart(r,l[0],l[1])
        img2 = draw_line(img2,x,y)

    if show_img:
        show(img2)
    return img2

                     
    
def show(img,path=None,dim=(7,7),markersize=3,axis=True):
    """
        Show img with dim
    """
    plt.figure(figsize=dim)
    plt.imshow(img)
    if path is not None:
        plt.plot(path[:,0],path[:,1],'--r.',markersize=markersize)
    if not axis:
        plt.gca().axis("off")
    plt.show()
