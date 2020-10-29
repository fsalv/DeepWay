import numpy as np
from tqdm.notebook import tqdm
import tensorflow as tf


def deepPathLoss(y_true, y_pred):
    """
    MSE weighted for true and false waypoints
    """
    mse_zero = tf.keras.losses.MeanSquaredError()
    
    mse_one = tf.keras.losses.MeanSquaredError()
    
    # find zero values in y_ture
    zero = tf.constant(0, dtype=tf.float32)
    where = tf.equal(y_true[...,0], zero)
    indices_zero = tf.where(where)
    
    y_true_zero = tf.gather_nd(y_true, indices_zero)
    y_pred_zero = tf.gather_nd(y_pred, indices_zero)
    
    # find one values in y_ture
    one = tf.constant(1, dtype=tf.float32)
    where = tf.equal(y_true[...,0], one)
    indices_one = tf.where(where)
    
    y_true_one = tf.gather_nd(y_true, indices_one)
    y_pred_one = tf.gather_nd(y_pred, indices_one)
        
    return mse_zero(y_true_zero, y_pred_zero) * 0.3 + mse_one(y_true_one, y_pred_one) * 0.7



def waypointProxSup(row, col, d, conf_map):
    """
    Eliminate points too near to each other using the Euclidean distance
    """
    p = np.concatenate((row[...,None], col[...,None]), axis=-1)
    indices_all = []
    indices_good = set()
    for index_1, p_1 in enumerate(p):
        indices_temp = []
        indices_all.append(index_1) # append index_1 to the list of all checks
        indices_temp.append((index_1, conf_map[p_1[0], p_1[1]]))
        for index_2, p_2 in enumerate(p):
            if (np.linalg.norm(p_1-p_2)) < d:
                if index_1 != index_2:
                    indices_all.append(index_2)
                    indices_temp.append((index_2, conf_map[p_2[0], p_2[1]]))
        max_index = 0
        max_conf = 0
        if len(indices_temp) > 1:
            for index, conf in indices_temp:
                if conf > max_conf:
                    max_conf = conf
                    max_index = index
            indices_good.add(max_index)
        else:
            indices_good.add(index_1)
    return p[list(indices_good), 0], p[list(indices_good), 1]



def interpret(y_pred, conf_thresh, dist_thresh = 3, normalize=True, 
              waypoint_prox_sup=True, MASK_DIM = 400, K = 8, WAYP_VALUE = 255):
    """ BATCH Interpret predictions rescaling to the original dimension usinng correction coordinates. 
    If waypoint_prox_sup=True it applies a proximity suppression for the points in the scaled dimension.
    """
    if normalize:
        norm = K // 2
    else:
        norm = 1
    y_up = np.zeros((y_pred.shape[0], MASK_DIM, MASK_DIM, 2))
    y_pred_up = np.zeros((y_pred.shape[0], MASK_DIM, MASK_DIM))
    for i in range(y_pred.shape[0]):
        row, col = np.where(y_pred[i,:,:,0]>conf_thresh)
        for r, c in zip(row, col):
            coors = y_pred[i,r,c,1:]
            r_o = int((r * K) + coors[0] * norm + ((K // 2) + 1))
            c_o = int((c * K) + coors[1] * norm + ((K // 2) + 1))
            conf = y_pred[i,r,c,0]
            y_up[i,r_o,c_o,0] = WAYP_VALUE
            y_up[i,r_o,c_o,1] = conf
        if waypoint_prox_sup:
            row_pre, col_pre = np.where(y_up[i,:,:,1]>conf_thresh)
            row, col = waypointProxSup(row_pre, col_pre, dist_thresh, y_up[i,:,:,1])
            for r, c in zip(row, col):
                y_pred_up[i,r,c] = WAYP_VALUE
    return y_pred_up



def upRes(y_pred, normalize=True, MASK_DIM = 800, K = 8, WAYP_VALUE = 255):
    """BATCH Rescale prediction taking into account correction coordinates"""
    if normalize:
        norm = (K // 2)
    else:
        norm = 1
    y_up = np.zeros((y_pred.shape[0], MASK_DIM, MASK_DIM))
    for i in range(y_pred.shape[0]):
        row, col = np.where(y_pred[i,:,:,0]==WAYP_VALUE)
        for r, c in zip(row, col):
            coors = y_pred[i,r,c,1:]
            r_o = int((r * K) + coors[0] * norm + ((K // 2) + 1))
            c_o = int((c * K) + coors[1] * norm + ((K // 2) + 1))
            y_up[i,r_o,c_o] = WAYP_VALUE
    return y_up



def AP(X_test, y_test, model, WAY_POINT_SUPR, DIST_RANGE = 8, K = 8, MASK_DIM = 800, WAYP_VALUE = 1, dist_thresh = 8):
    prec_tot = []
    rec_tot = []
    max_range = DIST_RANGE # minimum interow distance

    
    for c in tqdm(np.arange(0.1,1,0.1)):
        count_fp = 0
        true_p = 0
        gt = 0
        pred = 0
        for index in range(X_test.shape[0]):
            y_pred = interpret(model.predict(X_test[index:index+1]), conf_thresh = c,
                           dist_thresh = dist_thresh, waypoint_prox_sup=WAY_POINT_SUPR, K=K, MASK_DIM=MASK_DIM)

            row_pred, col_pred = (np.where(y_pred[0] > 0))
            row_true, col_true = (np.where(upRes(y_test[index:index+1], K=K, MASK_DIM=MASK_DIM, WAYP_VALUE=WAYP_VALUE)[0] > 0))
            
            p_pred = np.concatenate((row_pred[...,None], col_pred[...,None]), axis=-1)
            p_true = np.concatenate((row_true[...,None], col_true[...,None]), axis=-1)
            
            gt += len(p_true)
            pred += len(p_pred)
            
            indices = []

            for i_gt in p_true:
                for idx,i_pred in enumerate(p_pred):
                    if (np.linalg.norm(i_gt-i_pred)) < max_range and (idx not in indices):
                        indices.append(idx)
                        true_p += 1
                        break

        false_p = pred - true_p
        false_n = gt - true_p
        
    
        recall = true_p/(true_p + false_n)
        precision = true_p/(true_p + false_p)

        rec_tot.append(recall)
        prec_tot.append(precision)
    
    
    return rec_tot, prec_tot


