# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 11:43:52 2020

@author: tizia
SYNOP (surface synoptic observations) is a numerical code (called FM-12 by WMO) 
used for reporting weather observations made by manned and automated weather stations.
"""


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
#from Clustering_Up_scaling_NO_CONVULUTION import ClusteringUPS
from collections import Counter
import ALL_IN as ai
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import brier_score_loss
import warnings

warnings.filterwarnings('ignore')
#%%
# Observation tables paths

path_to_stations = "F:/Met_Eireann/Station_Data/"
path_to_stations_JUne = "F:/Met_Eireann/dataset_June_2020/"

os.chdir(path_to_stations)
# Stations details
data = pd.read_csv('synop.txt')

stations_coord = data.iloc[:,1:3].values
stations_synop = data.iloc[:,0].values

# Storing latitudes and longitudes matrices
lat = pd.read_csv('latitudes.csv', sep=';', header= None).values
lon = pd.read_csv('longitudes.csv', sep=';', header = None).values

obs_May_df = pd.read_csv('pe20200509.csv')

os.chdir(path_to_stations_JUne)
obs_June_df = pd.read_csv('OBS202006.csv')

#%%
# June DATA -------------------------------------------------------------
'''
Observation in June table are structured by sinoptic number:
    for each synop we have the values of hourly precipitation per day
    We want to get a separate table for each day
'''
# June has 30days, lets create an array of days in the month
June_days_list = np.arange(20200601,obs_June_df['date'].max()+1)

June_obs_by_day_list = []

for day in June_days_list:
    temp_d = obs_June_df[obs_June_df['date']==day]
    June_obs_by_day_list.append(temp_d)



#%%
# DATA ---------------------------------------------------------------

# Select May (only one day, no need to choose) or June's day of interest 


bs_NON_up_scaled = []
bs_fixed_up_scaling =  []
bs_spread_up_scaled =  []
bs_cluster_up_scaled =  []
bs_clustering =  []
bs_median_filter =  []


ROC_score_NON_up_scaled = []
ROC_score_fixed_up_scaling =  []
ROC_score_spread_up_scaled =  []
ROC_score_cluster_up_scaled =  []
ROC_score_clustering =  []
ROC_score_median_filter =  []

time_step = []
# Select the day: 
#[0] -> 01 June 2020
#[1] -> 02 June 2020
#[2] -> 03 June 2020
#[3] -> 04 June 2020
#[4] -> 05 June 2020
#[5] -> 06 June 2020
# ...
#[n-1] -> n-th June 2020

d = 27# +1 Day for June 
#days_list=[6,8,11,17,25,26,27]
#for d in days_list:
    # In that day, choose the h-step of accumulated precipitation
t=23
times = t
# Eqivalent for the model's member
forecast = times     

# Based on the prescribed day, set the index of folder's list for models files
# Remark: if the time-step is greater than 23h means that I need the obs of 24h in the future
# which is the next day, but the model gives 54h of forecast starting from the same day

if d==6: dd = 0
elif d==8: dd = 1 
elif d==11: dd = 2 
elif d==17: dd = 3
elif d==25: dd = 4 
elif d==26: dd = 5
else: dd = 6

if t>23:
    d=d+1    
    if d==7: dd = 0
    elif d==9: dd = 1 
    elif d==12: dd = 2 
    elif d==18: dd = 3
    elif d==26: dd = 4 
    elif d==27: dd = 5
    else: dd = 6
    t = t - 24
print(d)

month_dic = {'May':obs_May_df, 'June':June_obs_by_day_list[d]}

month = 'June'
obs_df = month_dic[month]
obs_data = np.zeros(obs_df.shape)

#Check if ALL the stations have 24 hours#
num_h_stations = list(Counter(obs_df['synop']).values())
name_station = list(Counter(obs_df['synop']).keys())

if (len(Counter(num_h_stations).values()) != 1):
    
    idx_no_24 = np.where(np.array(num_h_stations) != 24)[0]
    
    for ind in idx_no_24:    
        obs_df = obs_df[obs_df.synop != name_station[ind]]  
        num_h_stations = list(Counter(obs_df['synop']).values())
        name_station = list(Counter(obs_df['synop']).keys())

obs_data = np.zeros(obs_df.shape)   
# I want to know how many station I have
# Create the SYNOP array
synop_array = obs_df['synop'].values
#Use Counter() method. It return a dictionary which provide a number of recurence for each synop

n_stations = len((Counter(synop_array)).keys())


for h in range(24):
    temp = obs_df[obs_df['hour'] == h].values
    dist = len(temp)
    obs_data[dist*h:dist*h+dist,:] = temp
# Sort observation by hour


obs_by_h = pd.DataFrame(obs_data, columns=list(obs_df))    

# Accumulated observation
tp = [0]*n_stations
for h in range(1,24):
    for s in range(n_stations):  
        temp_h = obs_data[n_stations*h +s,6] + tp[n_stations*(h-1) +s]
        tp.append(temp_h)

obs_by_h['tp'] = tp
tp_array = np.array(tp).reshape(n_stations*24,1)
obs_data = np.hstack((obs_data,tp_array))


y = obs_data[n_stations*(t):n_stations*(t+1),1]
x = obs_data[n_stations*(t):n_stations*(t+1),2]
#z = obs_data[0+t:23+t,6]
z = obs_data[n_stations*(t):n_stations*(t+1),7]

#%%---------------------------------------------------------------------
# Select dataset based on the chosen obs and create a list of its member
# PROBABILITY MATRIX
# SET THE THRESHOLD
    #th = 5

threshold_list = [0,1,5,7.5,10,15,20,30,45]

for th in threshold_list:

# In this list we will append all matrix of 0 and 1

    if month=='May':
        path_to_data = "F:/Met_Eireann/Prog/Data/"
        
        os.chdir(path_to_data)
        
        list_files = sorted(os.listdir())
        
        df = 0
        numer_of_member = 11
        somma = 0
    
        for n in range(0, numer_of_member):
            df= (pd.read_csv('Forecast+{}_member'.format(forecast) + str(n) + '.csv', sep=' ', header=None).values)
            # Create an empy matrix that can be full for 0 and 1
            df_previus_step = (pd.read_csv('Forecast+{}_member'.format(forecast-1) + str(n) + '.csv', sep=' ', header=None).values)
            df_diff = df - df_previus_step
            a = np.zeros(df.shape)
            bolean = np.where(df < th, a, 1)
            somma = somma + bolean
        probability_matrix = somma / numer_of_member
    else: 
        path_to_July_data = "F:/Met_Eireann/dataset_June_2020/"
        os.chdir(path_to_July_data)
        list_folder = sorted(os.listdir())
        
        
        complete_path = path_to_July_data+list_folder[dd]+"/"
        os.chdir(complete_path)
        list_files = sorted(os.listdir())
        #print(list_files)
    
        number_of_members = 11
        if dd ==6: number_of_members = 5
        somma = 0
        for n in range(number_of_members):
            array =  pd.read_csv(list_files[forecast*number_of_members+n], sep=' ', header=None).values    
            matrix_0_1 = np.where(array > th , 1, 0)
            somma = somma + matrix_0_1
        probability_matrix = somma/number_of_members
            
        
    # Probability of observatiom
    zz = np.where(z > th , 1,0)
    
        
    i1 = 308; i2 = 524; j1 = 572 ; j2 = 724
    up=6
    probability_matrix_ie = probability_matrix[i1-up:i2+up,j1-up:j2+up]
    lat_ie = lat[i1:i2,j1:j2]
    lon_ie = lon[i1:i2,j1:j2]

    up=6
    A,B = ai.DynamicUpScaling(probability_matrix_ie,[13,11,9,7,5,5])
    C = ai.MedianFiltering(probability_matrix_ie)
    D = ai.FixedUpScaling(probability_matrix[i1:i2,j1:j2])
    E, F = ai.ClusterUpScale(probability_matrix_ie,[13,11,9,7,9,11,13,11,9,7,9,11])
#    G, H = ClusteringUPS(probability_matrix_ie,[3,5,7,9,7,5,3,5,7,9,5,3,5,7])
    
    #-----------------------------------------------------------------
    station_location = obs_data[0:n_stations,1:3]
    
    index_row=[]
    index_column=[]
    
    array_grid_obs_point = np.zeros((n_stations,3))
    A_array = np.zeros(n_stations)
    C_array = np.zeros(n_stations)
    D_array = np.zeros(n_stations)
    E_array = np.zeros(n_stations)
    G_array = np.zeros(n_stations)
    
    for s in range(n_stations):
        accuracy = 1
        idx_row = [0,0]
        idx_column = [0,0]
        while ( ((len(idx_row)) != 1) and ((len(idx_column)) != 1) ): 
                accuracy = accuracy + 0.01
                idx_row = np.where((lat_ie < station_location[s,0]+10**(-accuracy))&(lat_ie > station_location[s,0]-10**(-accuracy)) &
                                        (lon_ie < station_location[s,1]+10**(-accuracy))&(lon_ie > station_location[s,1]-10**(-accuracy)))[0]
                idx_column = np.where((lat_ie < station_location[s,0]+10**(-accuracy))&(lat_ie > station_location[s,0]-10**(-accuracy)) &
                                          (lon_ie < station_location[s,1]+10**(-accuracy))&(lon_ie > station_location[s,1]-10**(-accuracy)))[1]
       # print(idx_row,idx_column)
       # print("after:" ,len(idx_row), len(idx_column))
        index_row.append(np.asscalar(idx_row))
        index_column.append(np.asscalar(idx_column))
        
        # Put the latitude
        array_grid_obs_point[s,0] = lat_ie[idx_row,idx_column]
        #Put the longitude
        array_grid_obs_point[s,1] = lon_ie[idx_row,idx_column]
        # Put the value
        array_grid_obs_point[s,2] = probability_matrix_ie[idx_row,idx_column]
    
        A_array[s] = A[idx_row,idx_column]
        C_array[s] = C[idx_row,idx_column]
        D_array[s] = D[idx_row,idx_column]
        E_array[s] = E[idx_row,idx_column]
#        G_array[s] = E[idx_row,idx_column]
        
    #----------------------------------------------------------------------
    plt.figure(figsize=(30,20))
    
    plt.subplot(231)
    plt.pcolor(lon_ie,lat_ie, probability_matrix_ie[up:(i2-i1)+up, up:(j2-j1)+up], cmap='Blues_r')
    plt.colorbar()
    sns.scatterplot(x=x,y=y, hue=zz , legend="full", palette='Reds', marker='o')
    sns.scatterplot(x=array_grid_obs_point[:,1],y=array_grid_obs_point[:,0],
                    hue=array_grid_obs_point[:,2] , marker='x', palette='Greens')
    plt.title("Stations position", fontsize=20)
    
    plt.subplot(232)
    plt.pcolor(lon_ie,lat_ie, D, cmap='Blues_r')
    plt.colorbar()
    sns.scatterplot(x=x,y=y, hue=zz , legend="full", palette='Reds', marker='o')
    sns.scatterplot(x=array_grid_obs_point[:,1],y=array_grid_obs_point[:,0],
                    hue=D_array , marker='x', palette='Greens')
    plt.title("Fixed radius (R=2)", fontsize=20)
    
    
    plt.subplot(233)
    plt.pcolor(lon_ie,lat_ie, A, cmap='Blues_r')
    plt.colorbar()
    sns.scatterplot(x=x,y=y, hue=zz , legend="full", palette='Reds', marker='o')
    sns.scatterplot(x=array_grid_obs_point[:,1],y=array_grid_obs_point[:,0],
                    hue=A_array , marker='x', palette='Greens')
    plt.title("Spread-based Up-scaling", fontsize=20)
    
    if F=='None':
        plt.subplot(234)
        plt.pcolor(lon_ie,lat_ie, np.array(B).reshape((i2-i1),(j2-j1)), cmap='Blues_r')
        plt.colorbar()
        plt.title("Spread", fontsize=20)
        
    else:    
        plt.subplot(234)
        plt.pcolor(lon_ie,lat_ie, F, cmap='Blues_r')
        plt.colorbar()
        plt.title("Cluster", fontsize=20)


    plt.subplot(235)
    plt.pcolor(lon_ie,lat_ie, E, cmap='Blues_r')
    plt.colorbar()
    sns.scatterplot(x=x,y=y, hue=zz , legend="full", palette='Reds', marker='o')
    sns.scatterplot(x=array_grid_obs_point[:,1],y=array_grid_obs_point[:,0],
                    hue=E_array , marker='x', palette='Greens')
    plt.title("Clustering-based Up-scaling", fontsize=20)


        
    plt.subplot(236)
    plt.pcolor(lon_ie,lat_ie, C, cmap='Blues_r')
    plt.colorbar()
    sns.scatterplot(x=x,y=y, hue=zz , legend="full", palette='Reds', marker='o')
    sns.scatterplot(x=array_grid_obs_point[:,1],y=array_grid_obs_point[:,0],
                    hue=C_array , marker='x', palette='Greens')
    plt.title("Median Filtering", fontsize=20)
    
    
    plt.show()
    
    
    #%%------------------------------------------------------------------------
    
    
    fss_NON_up_scaled = ai.BrierScore(zz, array_grid_obs_point[:,2])
    
    fss_fixed_up_scaling = ai.BrierScore(zz, D_array)
    
    fss_spread_up_scaled = ai.BrierScore(zz, A_array)
    
    fss_cluster_up_scaled = ai.BrierScore(zz, E_array)
    
#    fss_clustering = ai.BrierScore(zz, G_array)
    
    fss_median_filter = ai.BrierScore(zz,C_array)
    
    print('BS for probability matrix not up-scaled:',fss_NON_up_scaled, '\n',
          'BS for fixed up-scaled:',fss_fixed_up_scaling,'\n',
          'BS for spread up-scaled:',fss_spread_up_scaled ,'\n',
          'BS for clustering based up-scaled:',fss_cluster_up_scaled,'\n',
#          'BS for clustering without convolution:',fss_clustering,'\n',
          'BS for median filtering:',fss_median_filter)
       
    
    
    bs_NON_up_scaled.append(brier_score_loss(zz, array_grid_obs_point[:,2]))
    
    bs_fixed_up_scaling.append(brier_score_loss(zz, D_array))
    
    bs_spread_up_scaled.append(brier_score_loss(zz, A_array))
    
    bs_cluster_up_scaled.append(brier_score_loss(zz, E_array))
    
#   bs_clustering.append(brier_score_loss(zz, G_array))
    
    bs_median_filter.append(brier_score_loss(zz,C_array))
    
    #BS_list = [str(th)+'mm',fss_NON_up_scaled,fss_fixed_up_scaling,fss_spread_up_scaled,fss_cluster_up_scaled, fss_median_filter]
    if np.sum(zz) != 0:
    
    
        ROC_score_NON_up_scaled.append(roc_auc_score(zz, array_grid_obs_point[:,2]))
        
        ROC_score_fixed_up_scaling.append(roc_auc_score(zz, D_array))
        
        ROC_score_spread_up_scaled.append(roc_auc_score(zz, A_array))
        
        ROC_score_cluster_up_scaled.append(roc_auc_score(zz, E_array))
        
#        ROC_score_clustering.append(roc_auc_score(zz, G_array))
        
        ROC_score_median_filter.append(roc_auc_score(zz,C_array))
        
        print('AUC for probability matrix not up-scaled:',roc_auc_score(zz, array_grid_obs_point[:,2]), '\n',
          'AUC for fixed up-scaled:',roc_auc_score(zz, D_array),'\n',
          'AUC for spread up-scaled:',roc_auc_score(zz, A_array) ,'\n',
          'AUC for clustering based up-scaled:',roc_auc_score(zz, E_array),'\n',
          'AUC for median filtering:',roc_auc_score(zz,C_array))
        
        
        time_step.append(int(th))
        

    #%%----------------------------------------------------------------------
    os.chdir("F:/Met_Eireann/prog/imm")
    sns.set(style="darkgrid")
    
    plt.figure(figsize =(10,8))
    #ns_probs
    #ns_fpr, ns_tpr, _ = roc_curve(zz, ns_probs)
    aaa , bbb , _ = roc_curve(zz, array_grid_obs_point[:,2])
    ccc , ddd , _ = roc_curve(zz, D_array)
    eee , fff , _ = roc_curve(zz, A_array)
    ggg , hhh , _ = roc_curve(zz, E_array)
    iii , lll , _ = roc_curve(zz, C_array)
#    mmm , nnn , _ = roc_curve(zz, G_array)
    
    plt.plot(aaa , bbb, marker='o', label='No up-scaling')
    plt.plot(ccc , ddd , marker='x', label='Fixed up-scaling')
    plt.plot(eee , fff , marker='^', label='Spread-based')
    plt.plot(ggg , hhh , marker='*', label='Clustering-based')
    plt.plot(iii , lll , marker='>', label='Median Filtering')
    #plt.plot(mm , nn , marker='*', label='Sum-Clustering')
    
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC curve - {},{} - +{}h - {}mm'.format(month,d+1,times, th))
    # show the legend
    plt.legend()
    plt.show()
    #plt.savefig("ROC_curve_{}{}+{}h_th{}mm.jpg".format(month,d+1,times,th))
    

    print(f"{th}, done")

#%%
plt.figure(figsize =(10,8))

days_array = np.array(threshold_list) 
plt.plot(days_array[1:], bs_NON_up_scaled[1:] , marker='o', label='FPM')
plt.plot(days_array[1:], bs_fixed_up_scaling[1:]  , marker='x', label='Fixed up-scaling')
plt.plot(days_array[1:], bs_spread_up_scaled[1:] , marker='^', label='Spread-based')
plt.plot(days_array[1:], bs_cluster_up_scaled[1:] , marker='*', label='Clustering-based')
#plt.plot(days_array, bs_median_filter , marker='>', label='Median Filtering')
#plt.plot(threshold_list, bs_clustering , marker='*', label='Sum-Clustering')

plt.xlabel('Precipitation Thresholds (mm)', fontsize = 16)
plt.ylabel('BS', fontsize = 18)
#plt.title('Brier Score - {},{},{}h '.format(month,d+1,times), fontsize = 20)

plt.legend(loc='best')

plt.savefig("BS_{}{}_{}h.jpg".format(month,d+1,times))
plt.show()

plt.figure(figsize =(10,8))
#time_step = np.arange( 0,len(ROC_score_NON_up_scaled))
#time_step = threshold_list
plt.plot(time_step[1:], ROC_score_NON_up_scaled[1:] , marker='o', label='FPM')
plt.plot(time_step[1:], ROC_score_fixed_up_scaling[1:]  , marker='x', label='Fixed up-scaling')
plt.plot(time_step[1:], ROC_score_spread_up_scaled[1:] , marker='^', label='Spread-based')
plt.plot(time_step[1:], ROC_score_cluster_up_scaled[1:] , marker='*', label='Clustering-based')
#plt.plot(time_step, ROC_score_median_filter , marker='>', label='Median Filtering')
#plt.plot(time_step, ROC_score_clustering , marker='*', label='Sum-Clustering')

plt.xlabel('Precipitation Thresholds (mm)', fontsize = 16)
plt.ylabel('AUC', fontsize = 18)
#plt.title('AUC Score - {},{},{}h '.format(month,d+1,times), fontsize = 20)

plt.legend(loc='best')
plt.savefig("AUC_{}{}_{}h.jpg".format(month,d+1,times))
plt.show()


data=list(zip(threshold_list,bs_NON_up_scaled, bs_fixed_up_scaling, bs_spread_up_scaled, bs_cluster_up_scaled, bs_median_filter, ROC_score_NON_up_scaled, ROC_score_fixed_up_scaling, ROC_score_spread_up_scaled ,ROC_score_cluster_up_scaled, ROC_score_median_filter))
df = pd.DataFrame(data, columns = ['Threshold_BS', 'Original_BS','Fixed_BS','Spread_BS','Clusering_BS', 'Median_BS', 'Original_AUC','Fixed_AUC','Spread_AUC','Clusering_AUC', 'Median_AUC']) 
df.to_csv(f'{month}_{d+1}+{times+1}h.csv', index=False)  
#%%
os.chdir("F:/Met_Eireann/prog/Results")


all_BS_AUC = pd.read_excel('All_BS_AUC_Copy.xlsx')
BS_original = (np.mean(all_BS_AUC.iloc[:,1]), stats.sem(all_BS_AUC.iloc[:,1])) 
BS_fix = (np.mean(all_BS_AUC.iloc[:,2]), stats.sem(all_BS_AUC.iloc[:,2]))
BS_spread = (np.mean(all_BS_AUC.iloc[:,3]), stats.sem(all_BS_AUC.iloc[:,3]))
BS_cluster = (np.mean(all_BS_AUC.iloc[:,4]),stats.sem(all_BS_AUC.iloc[:,4]))
BS_median = (np.mean(all_BS_AUC.iloc[:,5]), stats.sem(all_BS_AUC.iloc[:,5]))

AUC_original = (np.mean(all_BS_AUC.iloc[:,6]), stats.sem(all_BS_AUC.iloc[:,6])) 
AUC_fix = (np.mean(all_BS_AUC.iloc[:,7]), stats.sem(all_BS_AUC.iloc[:,7]))
AUC_spread = (np.mean(all_BS_AUC.iloc[:,8]), stats.sem(all_BS_AUC.iloc[:,8]))
AUC_cluster = (np.mean(all_BS_AUC.iloc[:,9]), stats.sem(all_BS_AUC.iloc[:,9]))
AUC_median = (np.mean(all_BS_AUC.iloc[:,10]), stats.sem(all_BS_AUC.iloc[:,10]))

