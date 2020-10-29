import numpy as np
import cv2
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

from utils.geometry import *



##################### Post-processing #####################


def wp_post_processing(mask,wp,angle_est_iter=30):
    """
        Complete wp post-processing. Returns the ordered wp.
    """
    # row angle estimation
    row_angle,row_vector,row_normal = estimate_row_angle(mask)
    # wp clustering
    labels,lab_sorted = cluster_wp(wp,eps=40)
    
    # clusters ordering and correction
    wp_cl = []
    for label in lab_sorted:
        wp_cli = order_cluster(wp[labels==label],row_normal)
        wp_cli = correct_wp(mask,wp_cli,row_angle)
        wp_cl.append(wp_cli)

    for p in wp[labels==-1]:   # add non-clustered points
        wp_cl.append(np.array([p]))  
        
    # merge clusters
    cluster_a,cluster_b = get_principal_clusters(mask,wp_cl,row_angle,row_normal)
    
    #wp ordering
    path,cluster_order = order_wp(cluster_a,cluster_b,mask,row_angle)
    #path,cluster_order = order_wp_OLD(cluster_a,cluster_b)
    return path,cluster_order
    
    

##################### Row angle estimation #####################

def find_point(mask,p1=(400,400)):
    """
        Find a suitable point out where mask is 1 starting from p1
    """
    p1+=random_displ(5,shape=(2)) 
    while not mask[p1[1],p1[0]]:
        p1+=random_displ(5,shape=(2))
    return p1



def check_edges(mask,p1,p2,l=300):
    """
        Computes falling edges number between p1 and p2
    """
    xy = np.linspace(p1,p2,l)
    x_line = np.round(xy[...,0]).astype(int)
    y_line = np.round(xy[...,1]).astype(int)
    a = mask[y_line,x_line]
    #mask_falling = (a[:-1] > 0.5) & (a[1:] < 0.5)
    #falling_edges = np.flatnonzero(mask_falling)
    return np.sum(a==0)
    #return len(falling_edges)



def minimize_edges(mask,p1=(400,400),l=300):
    """
        Estimate row angle by minimizing the number of edges found from p1
    """
    min_cross = np.inf
    angles = np.linspace(-np.pi/2,np.pi/2,500)
    np.random.shuffle(angles)
    for alpha in angles:
        p2 = line_polar_to_cart(l,alpha,p1)
        cross = check_edges(mask,p1,p2,l)
        if cross<min_cross:
            alpha_row = alpha
            min_cross = cross
            if not min_cross:
                break
    return alpha_row




def estimate_row_angle(img,verbose=False):
    edges = np.bitwise_not((img*255).astype("uint8"))
    linesP = cv2.HoughLinesP(edges,1,np.pi/180,50,None,50,1)
    
    if linesP is None:   #use the iterative random approach
        if verbose:
            print("Hough tranform failed, iterative approach.")
        return estimate_row_angle_iterative(img)
    
    row_angle = []
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        row_angle.append(line_polar(l[0:2],l[2:4])[1])
        if abs(row_angle[-1] - np.mean(row_angle))>np.pi/10:      # to avoid errors if the angle is close to -pi or pi
            row_angle.pop(-1)

    row_angle = np.mean(row_angle)
    row_vector = np.array(line_polar_to_cart(1,row_angle)) # vector in direction of alpha
    row_normal = np.array([-row_vector[1],row_vector[0]])
    return row_angle,row_vector,row_normal



def estimate_row_angle_iterative(mask,iterations=30):
    """
        Estimate row angle by several iterations of edge cross minimization
    """
    mask_center = mask.shape[1]//2,mask.shape[0]//2
    l = int(0.75*min(mask.shape)//2)
    
    row_angle = []
    for _ in range(iterations):
        p1 = find_point(mask,mask_center)
        row_angle.append(minimize_edges(mask,p1,l))
        if abs(row_angle[-1] - np.mean(row_angle))>np.pi/4:      # to avoid errors if the angle is close to -pi/2 or pi/2
            row_angle.pop(-1)

    row_angle = np.mean(row_angle)
    row_vector = np.array(line_polar_to_cart(1,row_angle)) # vector in direction of alpha
    row_normal = np.array([-row_vector[1],row_vector[0]])    # normal vector
    return row_angle,row_vector,row_normal

##################### Wp clustering #####################


def cluster_wp(wp,eps=40,min_samples=2,verbose=False):
    """
        Cluster wp with DBSCAN and sort the labels wrt cluster dimension
    """
    clust = DBSCAN(eps=eps,min_samples=min_samples)
    labels = clust.fit_predict(wp)
    lab,count = np.unique(labels,return_counts=True)
    if (count[np.where(lab==-1)]>0.5*len(wp)).any(): # too much unclustered points
        if verbose:
            print(f"Too much unclustered points, increase eps to {eps+5}.")
        return cluster_wp(wp,eps+5,min_samples,verbose=verbose)
    if len(lab[lab!=-1])<2 or max(count[lab!=-1])>0.6*len(wp):  # unbalanced clustering
        if verbose:
            print(f"Dominant cluster found, reducing eps to {eps-5}.")
        return cluster_wp(wp,eps-5,min_samples,verbose=verbose)
    lab_sorted = lab[np.argsort(count)[::-1]]
    lab_sorted = lab_sorted[lab_sorted>=0]
    return labels,lab_sorted


def order_cluster(wp,row_normal):
    """
        Project cluster along row noraml direction
    """
    wp_tr = np.dot(wp,row_normal)
    sorted_indexes = np.argsort(wp_tr)
    return wp[sorted_indexes]


##################### Wp correction #####################

def move_wp(wp,alpha,mask,direction=1):
    """
        Move wp along alpha until mask is 1
    """
    l = 0
    while not mask[wp[1],wp[0]]:
        l+=1*direction
        x,y = line_polar_to_cart(l,alpha,wp)
        wp = int(round(x)),int(round(y))
    return wp



def check_connection_OLD(mask,p1,p2,row_angle):
    """
        Compute the number of crossed rows between p1 and p2
    """
    alpha = line_polar(p1,p2)[1]
    if not mask[p1[1],p1[0]]:   # check if p1 is on row
        p1 = move_wp(p1,alpha,mask,direction=-1)
    if not mask[p2[1],p2[0]]:   # check if p2 is on row
        p2 = move_wp(p2,alpha,mask)

    l = (compute_distance(p1,p2)+1).astype("int")
    xy = np.linspace(p1,p2,l)
    x_line = np.round(xy[...,0]).astype(int)
    y_line = np.round(xy[...,1]).astype(int)

    angle_diff = alpha-row_angle
    cooldown = abs(find_intersect(angle_diff,(0,0),0,(0,1),ret_xy=False))
    cooldown = int(np.round(np.clip(cooldown,1,min(l/10,100))))

    a = mask[y_line,x_line]
    a = np.insert(a,len(a),0)
    
    mask_falling = (a[:-1] > 0.5) & (a[1:] < 0.5)
    falling_edges = np.flatnonzero(mask_falling)
    mask_rising = (a[:-1] < 0.5) & (a[1:] > 0.5)
    rising_edges = np.flatnonzero(mask_rising)
    
    cross = 0
    inverted = False
    row_centers = []
    for i in range(len(rising_edges)):
        if not inverted:
            if falling_edges[i]+cooldown<=rising_edges[i]:
                center = np.mean([falling_edges[i],rising_edges[i]])+1
                row_centers.append(center.astype(int))
                cross +=1
                inverted = True
        if inverted:
            if rising_edges[i]+cooldown*3<=falling_edges[i+1]: # minum of 3 pixels for intrarow
                inverted = False
    
    return cross,p1,p2,xy[row_centers].astype(int)



def check_connection(mask,p1,p2,row_angle):
    """
        Compute the number of crossed rows between p1 and p2
    """
    alpha = line_polar(p1,p2)[1]
    if not mask[p1[1],p1[0]]:   # check if p1 is on row
        p1 = move_wp(p1,alpha,mask,direction=-1)
    if not mask[p2[1],p2[0]]:   # check if p2 is on row
        p2 = move_wp(p2,alpha,mask)

    l = (compute_distance(p1,p2)+1).astype("int")
    xy = np.linspace(p1,p2,l)
    x_line = np.round(xy[...,0]).astype(int)
    y_line = np.round(xy[...,1]).astype(int)

    angle_diff = alpha-row_angle
    cooldown = abs(find_intersect(angle_diff,(0,0),0,(0,1),ret_xy=False))
    cooldown = int(np.round(np.clip(cooldown,1,min(l/3,100))))

    a = mask[y_line,x_line]
    a = np.insert(a,len(a),0)
    
    mask_falling = (a[:-1] > 0.5) & (a[1:] < 0.5)
    falling_edges = np.flatnonzero(mask_falling)
    mask_rising = (a[:-1] < 0.5) & (a[1:] > 0.5)
    rising_edges = np.flatnonzero(mask_rising)
    
    b = rising_edges-falling_edges[:-1]
    c = falling_edges[1:]-rising_edges
    d = np.where((c<=cooldown*2)==0)[0]+1
    
    row_cross_sum  = []
    row_centers = []
    
    if rising_edges.size:
        i_old = 0
        for i in d:
            row_cross_sum.append(np.sum(b[i_old:i]))
            row_centers.append(np.mean((falling_edges[i_old],rising_edges[i-1]+1)).astype(int))
            i_old = i
        row_cross_sum.append(np.sum(b[i_old:]))
        row_centers.append(np.mean((falling_edges[i_old],rising_edges[-1]+1)).astype(int))
    
    row_cross_sum = np.array(row_cross_sum,"int")
    row_centers = np.array(row_centers,"int")
    
    cross = np.sum(row_cross_sum>cooldown)
    row_centers = row_centers[row_cross_sum>cooldown]
    return cross,p1,p2,xy[row_centers].astype(int)



def correct_wp(mask,wp,row_angle,verbose=False):
    """
        Analyze the wp in a cluster and add/remove/move to correctly cover the whole parcel
    """
    wp_corrected = []    
    for i in range(len(wp)-1):
        cross,p1,p2,row_centers = check_connection(mask,wp[i],wp[i+1],row_angle)
        if cross > 0:
            wp[i+1] = p2
            wp_corrected.append(p1)
            if cross > 1:
                wp_new_x = np.round(np.convolve(row_centers[:,0],[0.5,0.5],mode="valid")).astype(int)[:,None]
                wp_new_y = np.round(np.convolve(row_centers[:,1],[0.5,0.5],mode="valid")).astype(int)[:,None]
                wp_new = np.concatenate([wp_new_x,wp_new_y],axis=1)
                for p in wp_new:
                    wp_corrected.append(p)
                if verbose:
                    print(f"[{i}]: missing wp. Adding {len(wp_new)} points at: {wp_new}")
        else:
            wp[i+1] = np.mean((p1,p2),axis=0)  # if both in the same row, take the mean of the two
            if verbose:
                print(f"[{i}]: more wp in the same row: averaging.")
    wp_corrected.append(wp[-1])
    return np.array(wp_corrected)  



##################### Clusters merging #####################

def coverage(cl1,cl2,plot=False):
    """
        Compute the coverages of cl1 and cl2
    """
    if max(cl2)<min(cl1) or max(cl1)<min(cl2):
        common = None
    else:
        common = (max(min(cl1),min(cl2)),min(max(cl1),max(cl2)))

    if common:
        #cov1 = np.sum([1 for p in cl1 if common[0]<=p<=common[1]])
        #cov2 = np.sum([1 for p in cl2 if common[0]<=p<=common[1]])
        #cov1 = max(cov1,1)/cl1.size
        #cov2 = max(cov2,1)/cl2.size
        cov1 = 1/cl1.size
        cov2 = 1/cl2.size
        if common[1]-common[0]:
            cov1 = np.max(((common[1]-common[0])/(cl1.max()-cl1.min()),cov1))
            cov2 = np.max(((common[1]-common[0])/(cl2.max()-cl2.min()),cov2))
    else:
        cov1 = cov2 = 0
    if plot:
        plt.figure(figsize=(20,3))
        plt.plot(cl1,np.zeros(len(cl1))+1,'o-',linewidth=5,markersize=10)
        plt.plot(cl2,np.zeros(len(cl2)),'o-',linewidth=5,markersize=10)

        plt.ylim((-1,2))
        plt.xticks([])
        plt.yticks([])
        plt.show()
    return cov1,cov2



def compute_coverage(cl1,CL2):
    """
        Compute the coverages of cl with super-cluster CL
    """
    if not cl1.size or not np.sum([cl.size for cl in CL2]):
        return 0,0
    cov_cl1 = []
    cov_CL2 = []
    for cl in CL2:
        cov1,cov2 = coverage(cl1,cl)
        cov_cl1.append(cov1)
        cov_CL2.append(cov2)
    
    cov_cl1 = min(np.sum(cov_cl1),1)
    sizes = [cl.size for cl in CL2]
    
    cov_CL2 = np.sum(np.array(cov_CL2)*np.array(sizes))/np.sum(sizes)
    return cov_cl1,cov_CL2



def merge_clusters(mask,CL,row_angle,row_normal,verbose=False):
    """
        Merge clusters in the correct order and analyze borders
    """
    if len(CL)>1:
        cl_proj = []
        for c in CL:
            cl_proj.append(np.dot(c,row_normal))
        order = np.argsort([c[0] for c in cl_proj])
        merged_cl = np.empty((0,2))
        for i in range(len(order[:-1])):
            wp_border = correct_wp(mask,[CL[order[i]][-1],CL[order[i+1]][0]],row_angle,verbose)
            CL[order[i+1]][0] = wp_border[-1]
            merged_cl = np.append(merged_cl,CL[order[i]][:-1],axis=0)
            merged_cl = np.append(merged_cl,wp_border[:-1],axis=0)
        merged_cl = np.append(merged_cl,CL[order[-1]],axis=0)
    else:
        merged_cl = CL[0]
    return merged_cl.astype("int")



def get_principal_clusters(mask,wp_cl,row_angle,row_normal,verbose=False):
    """
        Merge all the minor clusters to the principal ones to get two final clusters
    """
    wp_cl = np.array(wp_cl,dtype=object)
    cl_proj = []
    for cl in wp_cl:
        cl_proj.append(np.dot(cl,row_normal))
    cl_proj = np.array(cl_proj,dtype=object)

    CLA = [0]
    CLB = []
    indexes = list(range(1,len(cl_proj)))

    while len(indexes):
        nA= np.sum([len(cl) for cl in cl_proj[CLA]]) # wp in CLA
        nB= np.sum([len(cl) for cl in cl_proj[CLB]]) # wp in CLB
        if verbose:
            print(f"To be assigned: {indexes}")
            print(f"\tCLA: {nA} points\n\tCLB: {nB} points")

        # compute coverages
        covA = np.array([compute_coverage(cl,cl_proj[CLA]) for cl in cl_proj[indexes]])
        covB = np.array([compute_coverage(cl,cl_proj[CLB]) for cl in cl_proj[indexes]])

        #remove clusters totally covered by both
        bad_indexes = np.where(np.bitwise_and(np.bitwise_and(covA[:,0]>0.5,covB[:,0]>0.5),np.array([cl.size for cl in cl_proj[indexes]])>0))[0]
        indexes = list(np.delete(indexes,bad_indexes))
        covA = np.delete(covA,bad_indexes,axis=0)
        covB = np.delete(covB,bad_indexes,axis=0)
        
        if not len(indexes):
            break
        
        #product of coverages (cumulative coverage)
        covA = covA[:,0]*covA[:,1]
        covB = covB[:,0]*covB[:,1]
        measure = "coverage"
        
        if nA<nB: # append to CLA
            if not np.sum(covB):   # if all not covered -> assign the most distant from B
                mean_CL = np.sum([np.sum(cl,axis=0) for cl in wp_cl[CLB]],axis=0)/np.sum([len(cl) for cl in wp_cl[CLB]])
                clusters_means = [np.mean(cl,axis=0) for cl in wp_cl[indexes]]
                covB = [np.sum((cl-mean_CL)**2) for cl in clusters_means]  # new measure: distances
                measure = "distance"   
            
            i = np.argmax(covB)
            index = indexes.pop(i)
            CLA.append(index)
            if verbose:
                print(f"\tAssigning to CLA cluster {index} of {len(cl_proj[index])} points with {measure} {np.around(covB[i],2)}")
        elif nA>nB:       # append to CLB
            if not np.sum(covA):   # if all not covered -> assign the most distant from A
                mean_CL = np.sum([np.sum(cl,axis=0) for cl in wp_cl[CLA]],axis=0)/np.sum([len(cl) for cl in wp_cl[CLA]])
                clusters_means = [np.mean(cl,axis=0) for cl in wp_cl[indexes]]
                covA = [np.sum((cl-mean_CL)**2) for cl in clusters_means]  # new measure: distances
                measure = "distance"
            
            i = np.argmax(covA)
            index = indexes.pop(i)
            CLB.append(index)
            if verbose:
                print(f"\tAssigning to CLB cluster {index} of {len(cl_proj[index])} points with {measure} {np.around(covA[i],2)}") 
        else:
            if not np.sum(covA) and not np.sum(covB):   # if all not covered
                mean_CLA = np.sum([np.sum(cl,axis=0) for cl in wp_cl[CLA]],axis=0)/np.sum([len(cl) for cl in wp_cl[CLA]])
                mean_CLB = np.sum([np.sum(cl,axis=0) for cl in wp_cl[CLB]],axis=0)/np.sum([len(cl) for cl in wp_cl[CLB]])
                clusters_means = [np.mean(cl,axis=0) for cl in wp_cl[indexes]]
                covA = [np.sum((cl-mean_CLA)**2) for cl in clusters_means]  # new measures: distances
                covB = [np.sum((cl-mean_CLB)**2) for cl in clusters_means]
                measure = "distance"
            
            iA = np.argmax(covA)
            iB = np.argmax(covB)

            if covB[iB]>covA[iA]: # append to CLA
                index = indexes.pop(iB)
                CLA.append(index)
                if verbose:
                    print(f"\tAssigning to CLA cluster {index} of {len(cl_proj[index])} points with {measure} {np.around(covB[iB],2)}") 
            else: # append to CLB
                index = indexes.pop(iA)
                CLB.append(index)
                if verbose:
                    print(f"\tAssigning to CLB cluster {index} of {len(cl_proj[index])} points with {measure} {np.around(covA[iA],2)}")                
            
            
    cluster_a = merge_clusters(mask,wp_cl[CLA],row_angle,row_normal,verbose)
    cluster_b = merge_clusters(mask,wp_cl[CLB],row_angle,row_normal,verbose)
    return cluster_a,cluster_b



##################### Wp ordering #####################

def angle_deviation(p1,p2,row_angle):
    """
        Compute how much the angle between two wp is deviated from the estimated row_angle.
    """
    alpha1 = line_polar(p1,p2)[1]
    alpha2 = line_polar(p2,p1)[1]
    alpha = np.array((alpha1,alpha2))
    ind = np.argmin(np.abs(row_angle-alpha))
    alpha = alpha[ind]
    return row_angle-alpha,ind


def order_wp(cluster_a,cluster_b,mask,row_angle,verbose=False):
    """
        Compute the final order of the wp for path planning.
    """
    clusters = [cluster_a,cluster_b]
    counter = [0,0]
    names = ["A","B"]
    
    p1 = clusters[0][0]
    p2 = clusters[1][0]
    dev,ind = angle_deviation(p1,p2,row_angle)
    if dev<0:
        actual_clus = ind
    else:
        actual_clus = not ind
        
    p1 = clusters[actual_clus][counter[actual_clus]]
    wp_ordered = [p1]
    cluster_order = [actual_clus]
    counter[actual_clus]+=1
    
    while True:
        p2 = clusters[not actual_clus][counter[not actual_clus]]
        dev,ind = angle_deviation(p1,p2,row_angle)
        cross = check_connection(mask,p1,p2,row_angle)[0]

        if dev<-0.1 or (cross>0 and dev<=0):
            if not ind: # append in the same cluster
                if counter[actual_clus]<len(clusters[actual_clus]):
                    p1 = clusters[actual_clus][counter[actual_clus]]
                    wp_ordered.append(p1)
                    cluster_order.append(actual_clus)
                    counter[actual_clus]+=1
                else: #append p2 even if not optimal
                    wp_ordered.append(p2)
                    p1 = p2
                    actual_clus = not actual_clus
                    cluster_order.append(actual_clus)
                    counter[actual_clus]+=1
            else: # check the following p2
                counter[not actual_clus]+=1
        elif dev>0.1 or (cross>0 and dev>0):
            if ind: # append in the same cluster
                if counter[actual_clus]<len(clusters[actual_clus]):
                    p1 = clusters[actual_clus][counter[actual_clus]]
                    wp_ordered.append(p1)
                    cluster_order.append(actual_clus)
                    counter[actual_clus]+=1
                else: #append p2 even if not optimal
                    wp_ordered.append(p2)
                    p1 = p2
                    actual_clus = not actual_clus
                    cluster_order.append(actual_clus)
                    counter[actual_clus]+=1 
            else: # check the following p2
                counter[not actual_clus]+=1
        else: # append p2
            wp_ordered.append(p2)
            p1 = p2
            actual_clus = not actual_clus
            cluster_order.append(actual_clus)
            counter[actual_clus]+=1        

        if counter[not actual_clus]>=len(clusters[not actual_clus]): #other cluster ended
            while(counter[actual_clus]<len(clusters[actual_clus])):
                wp_ordered.append(clusters[actual_clus][counter[actual_clus]])
                cluster_order.append(actual_clus)
                counter[actual_clus]+=1
                if verbose:
                    print(counter,np.array(names)[np.array(cluster_order).astype("int")])
            break

        if verbose:
            print(counter,np.array(names)[np.array(cluster_order).astype("int")])
    
    return np.array(wp_ordered),np.array(cluster_order,"int")




def order_wp_OLD(cluster_a,cluster_b,mask,row_angle,verbose=False):
    """
        Computes the final order of the wp for path planning.
    """
    clusters = [cluster_a,cluster_b]
    names = ["A","B"]

    if len(cluster_a)>=len(cluster_b):
        wp_ordered = [cluster_a[0]]
        indexes = [1,0]
        next_clus = 1
        actual_clus = 0
    else:
        wp_ordered = [cluster_b[0]]
        cluster_order = [1]
        indexes = [0,1]
        next_clus = 0
        actual_clus = 1
        
    cluster_order = [actual_clus]
    while indexes[0]<len(cluster_a) or indexes[1]<len(cluster_b):
        if indexes[next_clus]<len(clusters[next_clus]):
            if actual_clus == next_clus:
                wp_ordered.append(clusters[next_clus][indexes[next_clus]])
                indexes[next_clus] += 1
                next_clus = not(next_clus)
                cluster_order.append(actual_clus)
                if verbose:
                    print(f"Adding point {indexes[actual_clus]-1} from cluster {names[actual_clus]}. Targets: {indexes}")
                continue

            else:
                if not check_connection(mask,wp_ordered[-1],clusters[next_clus][indexes[next_clus]],row_angle)[0]:
                    wp_ordered.append(clusters[next_clus][indexes[next_clus]])
                    indexes[next_clus] += 1
                    actual_clus = not(actual_clus)
                    cluster_order.append(actual_clus)
                    if verbose:
                        print(f"Adding point {indexes[next_clus]-1} from cluster {names[actual_clus]}. Targets: {indexes}")
                    continue

                if indexes[next_clus]+1<len(clusters[next_clus]):
                    if not check_connection(mask,wp_ordered[-1],clusters[next_clus][indexes[next_clus]+1],row_angle)[0]:
                        wp_ordered.append(clusters[next_clus][indexes[next_clus]+1])
                        indexes[next_clus] += 2
                        actual_clus = not(actual_clus)
                        cluster_order.append(actual_clus)
                        if verbose:
                            print(f"Skipping, adding point {indexes[next_clus]-2} from cluster {names[actual_clus]}. Targets: {indexes}")
                        continue

                if indexes[actual_clus]+1<len(clusters[actual_clus]):
                    wp_ordered.append(clusters[actual_clus][indexes[actual_clus]])
                    indexes[actual_clus] += 1
                    cluster_order.append(actual_clus)
                    if verbose:
                        print(f"Remaining, adding point {indexes[actual_clus]-1} from cluster {names[actual_clus]}. Targets: {indexes}")
                else:
                    if verbose:
                        print("I can't connect to the next point. Finishing.")
                    return np.array(wp_ordered),np.array(cluster_order)
        else:
            next_clus = not(next_clus)
    
    return np.array(wp_ordered),np.array(cluster_order)

