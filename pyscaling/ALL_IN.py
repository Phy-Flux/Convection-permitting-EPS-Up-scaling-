# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 10:57:43 2020

@author: tizia

This script contains the three methodology used to up-scale ensamble based 
forecast for precipitation. 
"""

import numpy as np
import statistics as stat
from scipy.ndimage import convolve, median_filter
from collections import Counter
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering


def GaussianKernel(r,bound=0.5):
    '''
    
    Parameters
    ----------
    r : integer
        length of matrix side
    bound : float 
        Boundary value to select the distribution's range
    sig : float number
        Standard deviation of the Normal distribution.
        Default set to be 1/2pi (in this way, the higher value it's one)

    Returns
    -------
    g = 2D array, a kernel with the shape of a Gaussian (normal distribution) curve.
    k2.max() = maximum convolved value between a kernel of probability 1 and Gaussian
    '''
    x = np.linspace(-bound,bound,r) ;  y= x
    X,Y = np.meshgrid(x,y)
    
    sig=1/np.sqrt(2*(np.pi))
    norm = 1/(2*(np.pi)*(sig**2))
    g = norm*np.exp(-(X**2 + Y**2)/(2*sig**2))
    
    k1 = np.ones((r,r))
    k2 = convolve(k1,g, mode='constant')

    return(g,k2.max())

def ClusterUpScale(prob_matrix,r_list,up = 6):
    '''
    
    Parameters
    ----------
    prob_matrix : 2D array like
        A fraction probability matrix
    r_list : list like
        Each element of the list must containg the radius to up-scale the n-th cluster
    up : integer, optional
        Border adjustments distance. 
        The default is 6.

    Returns
    -------
    Up-scaled probability matrix - 2D array like.
    cluster matrix - 2D array like
    '''
    # Store sides length of the input matrix 
    lx = prob_matrix.shape[0] ; ly = prob_matrix.shape[1]
    X_ie = prob_matrix.reshape(lx*ly,1) 
    
    # Perform the hierarchic clustering
    link_ie = linkage(X_ie, method='single')
    d_ie = dendrogram(link_ie, truncate_mode='lastp')#, no_plot=True)

    # Get the number of cluster without plotting the dendrongram using the color_list 
    list_col = d_ie['color_list']

    if len(Counter(list_col).keys()) == 1:
        up_scaled_precipitation = FixedUpScaling(prob_matrix,rad=5)[up:lx-up,up:ly-up]
        cluster = 'None'
    else:
        nc = list_col.count('b')

        num_clusters = nc + 1 
        ag = AgglomerativeClustering(n_clusters=num_clusters)
        Y_ie = ag.fit_predict(X_ie)
    
        cluster= Y_ie.reshape(lx,ly)[up:lx-up,up:ly-up]
        
        up_scaled_precipitation = np.zeros((lx-2*up, ly-2*up))
        for c in range(num_clusters):
            
    #        N = len(X_ie[np.where(Y_ie == c)])
    #        mean_of_cth_cluster = np.mean(X_ie[np.where(Y_ie == c)])
            
            r = r_list[c]
            centre = int(r/2)
            kernel, k_max = GaussianKernel(r=r,bound=0.1)
            #kernel = np.ones((r,r)) 
        # Find all indexes of the positions belonging to a cluster
            index = np.where(cluster == c)
        # Indexes rows and columns in lists
            list_row = list(index[0])
            list_column = list(index[1])
            l = len(list_row)
                
            for p in range(l):
                n = list_row[p]
                m = list_column[p]  
                
    #            mini_matrix = prob_matrix[n+up:n+1+up,m-centre+up:m+1+up]
    #            mini_matrix_mean = mini_matrix.mean()
                
                # Choose the spread metrics ---------------------------------------------------------------------------
    #            N = (up*2 +1)**2
    #            variance = np.sum((mini_matrix - mini_matrix_mean)**2)/(N-1)
    #            standard_deviation = np.sqrt(variance)
                 
            # Pointwise upscaling
                temp = convolve(prob_matrix[(n- centre)+up:(n+ centre+1)+up,((m-centre))+up:(m+centre+1)+up], kernel )[centre,centre]
                up_scaled_precipitation[n, m] = temp/k_max
    return(up_scaled_precipitation, cluster)


def DynamicUpScaling(matrice_prob_ie,sides_list, method_spread='sd', up = 6):
    '''
    The function perform the up-scaling using a dynamic radius.
    Output matrix calle: up_scaled_precipitation has the shape of the input matrix minus the border
    Borders can be set using the argument "up".

   Parameters
    ----------
    matrice_prob_ie : array like
        A fraction probability matrix
    sides_list : array/list like
        DESCRIPTION.
    method_spread : string like
        Method used to calculate the spread in the pointwise neighborhood.
        Options -> 
        1) 'sd' = Standard Deviation
        2) 'var' = Variance
        3) 'mad' = mean_absolute_deviation
        Default method is Standard Deviation
    up : integer, optional
        Border adjustments distance. 
        The default is 6.

    Returns
    -------
    Up-scaled probability matrix - array like.
    spread matrix - array like
    
    '''
    # Dictionary to contain: radius, center and maximum
    dic_r3 = {'radius':3,'center':1,'kernel':GaussianKernel(r=3)[0],'max':GaussianKernel(r=3)[1]}
    dic_r5 = {'radius':5,'center':2,'kernel':GaussianKernel(r=5)[0],'max':GaussianKernel(r=5)[1]}
    dic_r7 = {'radius':7,'center':3,'kernel':GaussianKernel(r=7)[0],'max':GaussianKernel(r=7)[1]}
    dic_r9 = {'radius':9,'center':4,'kernel':GaussianKernel(r=9)[0],'max':GaussianKernel(r=9)[1]}
    dic_r11 = {'radius':11,'center':5,'kernel':GaussianKernel(r=11)[0],'max':GaussianKernel(r=11)[1]}
    dic_r13 = {'radius':13,'center':6,'kernel':GaussianKernel(r=13)[0],'max':GaussianKernel(r=13)[1]}

#    dic_r3 = {'radius':3,'center':1,'kernel':np.ones((3,3))}
#    dic_r5 = {'radius':5,'center':2,'kernel':np.ones((5,5))}
#    dic_r7 = {'radius':7,'center':3,'kernel':np.ones((7,7))}
#    dic_r9 = {'radius':9,'center':4,'kernel':np.ones((9,9))}
#    dic_r11 = {'radius':11,'center':5,'kernel':np.ones((11,11))}
#    dic_r13 = {'radius':13,'center':6,'kernel':np.ones((13,13))}
    
    list_radius = [0]*len(sides_list)
    for l in range(len(sides_list)):
        if sides_list[l] ==3: 
            list_radius[l]= dic_r3 
        elif sides_list[l] ==5: 
            list_radius[l]= dic_r5
        elif sides_list[l] ==7: 
            list_radius[l]= dic_r7
        elif sides_list[l] ==9: 
            list_radius[l]= dic_r9
        elif sides_list[l] ==11: 
            list_radius[l]= dic_r11
        else: 
            list_radius[l]= dic_r13
  
    lx = int(matrice_prob_ie.shape[0]) ;  ly = int(matrice_prob_ie.shape[1])
    up_scaled_precipitation = np.zeros((lx-2*up, ly-2*up))
    spread = []
    
    for i in range(up,lx-up):
        for j in range(up,ly-up):
    
            mini_matrix = matrice_prob_ie[(i - up ):(i + up+1),(j - up):(j + up +1)]
            mini_matrix_mean = mini_matrix.mean()
            
            # Choose the spread metrics ---------------------------------------------------------------------------
            N = (up*2 +1)**2
            mean_absolute_deviation = np.sum(np.abs(mini_matrix - mini_matrix_mean))/N
            variance = np.sum((mini_matrix - mini_matrix_mean)**2)/(N-1)
            standard_deviation = np.sqrt(variance)
            
            spread_dic = { 'mad':mean_absolute_deviation , 'var':variance, 'sd':standard_deviation}
            method = method_spread        
            spread_metric = spread_dic[method]
            spread.append(spread_metric)
    
    
            if spread_metric == .0:
                
                dic_radius = list_radius[0]
                radius = int(dic_radius['radius'])
                centre = dic_radius['center']
                kernel = dic_radius['kernel']

                # Storing the convolution
                temp = convolve(matrice_prob_ie[(i - centre):(i + centre+1),(j - centre):(j + centre+1)],kernel)[centre, centre]
                up_scaled_precipitation[i - up, j - up] = temp /dic_radius['max']
                    
            elif ((spread_metric <= .05) and (spread_metric != 0)):
                
                dic_radius = list_radius[1]
                radius = int(dic_radius['radius'])
                centre = dic_radius['center']
                kernel = dic_radius['kernel']
                
                # Storing the convolution
                temp = convolve(matrice_prob_ie[(i - centre):(i + centre+1), (j - centre):(j + centre+1)], kernel)[centre, centre]
                up_scaled_precipitation[i - up, j - up] = temp / dic_radius['max']
    
            elif ((spread_metric <= .06) and (spread_metric > 0.05)):
                
                dic_radius = list_radius[2]
                radius = int(dic_radius['radius'])
                centre = dic_radius['center']
                kernel = dic_radius['kernel']
                
                # Storing the convolution
                temp = convolve(matrice_prob_ie[(i - centre):(i + centre+1),(j - centre):(j + centre+1)],kernel)[centre, centre]
                up_scaled_precipitation[i - up, j - up] = temp / dic_radius['max']
    
            elif ((spread_metric < .075) and (spread_metric > .06)):
                dic_radius = list_radius[3]
                radius = int(dic_radius['radius'])
                centre = dic_radius['center']
                kernel = dic_radius['kernel']
    
    
                # Storing the convolution
                temp = convolve(matrice_prob_ie[(i - centre):(i + centre+1),(j - centre):(j + centre+1)],kernel)[centre, centre]
                up_scaled_precipitation[i - up, j - up] = temp / dic_radius['max']
    
            elif ((spread_metric < .2) and (spread_metric > .075)):
                dic_radius = list_radius[4]
                radius = int(dic_radius['radius'])
                centre = dic_radius['center']
                kernel = dic_radius['kernel']
    
                # Storing the convolution
                temp = convolve(matrice_prob_ie[(i - centre):(i + centre+1),(j - centre):(j + centre+1)],kernel)[centre, centre]
                up_scaled_precipitation[i - up, j - up] = temp / dic_radius['max']
    
            else:
                dic_radius = list_radius[5]
                radius = int(dic_radius['radius'])
                centre = dic_radius['center']
                kernel = dic_radius['kernel']
                
                # Storing the convolution
                temp = convolve(matrice_prob_ie[(i - centre):(i + centre+1),(j - centre):(j + centre+1)],kernel)[centre, centre]
                up_scaled_precipitation[i - up, j - up] = temp / dic_radius['max']
                    
    
    return(up_scaled_precipitation, spread)

def MedianFiltering(prob_matrix, up=6):
    '''
    Parameters
    ----------
    prob_matrix : 2D array like
        DESCRIPTION.
    up : integer, optional
        Border adjustments distance. The default is 6.

    Returns
    -------
    median_matrix : 

    '''
  # Kernel size
    side = up*2 + 1
    median_matrix = median_filter(prob_matrix, side)
    return(median_matrix[up:-up, up:-up])

def FixedUpScaling(prob_matrix,rad=5,coeff='Gaussian', Gauss_bound = 1):
    dic_kernel = {'Ones': np.ones((rad,rad)),'Gaussian': GaussianKernel(r=rad, bound=Gauss_bound)[0]}
    kernel = dic_kernel[coeff]
    up_scaled_fix_radius = convolve(prob_matrix, kernel, mode='constant')
    dic_max = {'Ones': up_scaled_fix_radius.max(),'Gaussian': GaussianKernel(r=rad, bound=Gauss_bound)[1]}
    normalization = dic_max[coeff]
    return(up_scaled_fix_radius/normalization)
