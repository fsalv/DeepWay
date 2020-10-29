import numpy as np
import cv2
import json
from utils.tools_net import interpret



######################## Read masks and predict wp ########################    


def read_and_predict(img,model,conf_thresh,dist_thresh,K):
    """ 
    Read an image and predict the waypoints.
    """
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    img[img<255/2] = 0
    img[img>255/2] = 255
    mask_dim = img.shape[0]
    pred = interpret(model.predict(cv2.bitwise_not(img)[None].astype('float32')),
                 conf_thresh = conf_thresh, dist_thresh = dist_thresh, waypoint_prox_sup=True, K=K, MASK_DIM=mask_dim)[0]
    y,x = np.where(pred)
    wp = [(i,j) for i,j in zip(x,y)]
    wp = np.array(wp)
    return img/255,wp


######################## Configuration ########################    


def load_config(config_path='config.json'):
    """
    Load config file
    """
    with open(config_path) as json_data_file:
        config = json.load(json_data_file)
    
    return config
    
    
######################## Derive the y mask from wp ########################    


def resizeAddCorrection(y, WAYP_VALUE, K, correction=True, normalize=True):
    """
    Take a waypoint mask and transform it in a scaled (K) version with three channels. First channels rescaled
    waypoints, second one x correction coordinates and last onne y correction coordinates.
    """
    if normalize:
        norm = K // 2
    else:
        norm = 1
    if correction:
        y_complete = np.zeros((y.shape[0], y.shape[1]//K, y.shape[2]//K, 3))
    else:
        y_complete = np.zeros((y.shape[0], y.shape[1]//K, y.shape[2]//K))
    for index in range(y.shape[0]):
        row, col = np.where(y[index] == WAYP_VALUE)
        row_res, col_res = row // K, col //K
        if correction:
            y_complete[index, row_res, col_res, 0] = WAYP_VALUE
            y_complete[index, row_res, col_res, 1] = (((row % K) - ((K // 2) + 1)) / norm)
            y_complete[index, row_res, col_res, 2] = (((col % K) - ((K // 2) + 1)) / norm)
        else:
            y_complete[index, row_res, col_res] = WAYP_VALUE
    return y_complete
