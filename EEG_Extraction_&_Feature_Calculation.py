import os 
import mne
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import scipy
import math 
import pywt
import emd
import numpy as np
import pandas as pd
import seaborn as sns
from random import randint
import entropy as ent 
import warnings
import matplotlib as mpl
import seaborn as sns
import pathlib
warnings.filterwarnings("ignore")
from glob import glob
from scipy.signal import filtfilt
from scipy import stats
from numpy.fft import fft
from tqdm import tqdm_notebook
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold, cross_val_score
from pyinform.dist import Dist
from pyinform.shannon import entropy

#----------------Reading, Filtering, Segmentation of EEG Signal----------------
file_path_non_seizure=glob('Non seizure_50/*.edf')
file_path_seizure=glob('Seizure_50/*.edf')
length_path1=len(file_path_non_seizure)
length_path2=len(file_path_seizure)



def read_data(filepath,start_time,end_time):
    raw=mne.io.read_raw_edf(filepath,preload=True)
    specific_ch = raw.copy().pick_channels(['F3-C3', 'C3-P3', 'F4-C4', 'C4-P4'])
    filtering=specific_ch.filter(l_freq=0.1,h_freq=60)
    data_frame=filtering[:,int(256*start_time):int(256*end_time)]
    return np.transpose(np.asarray(data_frame)[0])

# onesec_array=read_data('file_path_non_seizure',0,1)
# plt.plot(onesec_array)

#-----------------------non Seizure Data---------------------------------------
non_seizure_file_list=[read_data(i,0,60) for i in file_path_non_seizure]
# onesec_array_0=non_seizure_file_path_list[0]
# plt.plot(onesec_array_0)

#----------------------------Seizure Data--------------------------------------
get_timestamp=np.genfromtxt("seizure_time_stamp.csv",delimiter=",", dtype=int)
start_time=[]
for i in range(0,50):
    j=0
    get=get_timestamp[i,j]
    start_time.append(get)
start_time_array=np.asarray(start_time)    


end_time=[]
for i in range(0,50):
    j=1
    get=get_timestamp[i,j]
    end_time.append(get)
end_time_array=np.asarray(end_time)

seizure_file_list=[]
for k in range(0,50): 
    read_seizure=read_data(file_path_seizure[k],start_time[k],end_time[k])
    seizure_file_list.append(read_seizure)


###############################################################################
#----------------------------------DWT-----------------------------------------
##############################################################################
#--------------------non seizure DWT Coefficients Calculation------------------
non_seizure_DWT_Coeffs_list=[];
def get_DWT(i,j):
    z=pywt.wavedec(non_seizure_file_list[i][:,j],'db4',level=4,axis=0)
    return z
for i in range(0,50):
    for j in range(0,4):
        non_seizure_zz=get_DWT(i,j)
        non_seizure_DWT_Coeffs_list.append(non_seizure_zz)
    
non_seizure_DWT_Coeffs_array=np.asarray(non_seizure_DWT_Coeffs_list);

#------------------------seizure DWT Coefficients Calculation------------------
seizure_DWT_Coeffs_list=[];
def get_DWT(i,j):
    z=pywt.wavedec(seizure_file_list[i][:,j],'db4',level=4,axis=0)
    return z
for i in range(0,50):
    for j in range(0,4):
        seizure_zz=get_DWT(i,j)
        seizure_DWT_Coeffs_list.append(seizure_zz)
    
seizure_DWT_Coeffs_array=np.asarray(seizure_DWT_Coeffs_list);

#---------------------Statistical Features Calculations------------------------
def mean(data):
    return np.mean(data,axis=0)
    
def std(data):
    return np.std(data,axis=0)

def ptp(data):
    return np.ptp(data,axis=0)

def var(data):
        return np.var(data,axis=0)

def minim(data):
      return np.min(data,axis=0)


def maxim(data):
      return np.max(data,axis=0)

def argminim(data):
      return np.argmin(data,axis=0)


def argmaxim(data):
      return np.argmax(data,axis=0)

def mean_square(data):
      return np.mean(data**2,axis=0)

def rms(data): #root mean square
      return  np.sqrt(np.mean(data**2,axis=0))  

def abs_diffs_signal(data):
    return np.sum(np.abs(np.diff(data,axis=0)),axis=0)


def skewness(data):
    return stats.skew(data,axis=0)

def concatenate_features(data):
    return np.stack((mean(data),std(data),ptp(data),var(data),minim(data),maxim(data),argminim(data),argmaxim(data),mean_square(data),rms(data),abs_diffs_signal(data),skewness(data)),axis=0)




#------------------------------------------------------------------------------
DWT_feature_non_seizure=[]
for i in range(0,200):
    for j in range(0,5):
        DWT_features_n_seizure=concatenate_features(non_seizure_DWT_Coeffs_array[i,j])
        DWT_feature_non_seizure.append(DWT_features_n_seizure)
DWT_feature_non_seizure=np.asarray(DWT_feature_non_seizure)
DWT_feature_non_seizure=DWT_feature_non_seizure.reshape([200,12,5])
DWT_stat_feature_non_seizure=DWT_feature_non_seizure.reshape([200,60])       
#------------------------------------------------------------------------------        
DWT_feature_seizure=[]
for i in range(0,200):
    for j in range(0,5):
        DWT_features_seizure=concatenate_features(seizure_DWT_Coeffs_array[i,j])
        DWT_feature_seizure.append(DWT_features_seizure)
DWT_feature_seizure=np.asarray(DWT_feature_seizure)
DWT_feature_seizure=DWT_feature_seizure.reshape([200,12,5])
DWT_stat_feature_seizure=DWT_feature_seizure.reshape([200,60]) 

# #-----------------Concatenating non seizure & seizure stat features----------------
DWT_stat_features=np.concatenate((DWT_stat_feature_non_seizure,DWT_stat_feature_seizure),axis=0)
# #------------------------Seizure Label-----------------------------------------
# seizure_label=np.hstack([[len(i)*[1] for i in  seizure_All_features]])

# #------------------------Non Seizure Label-------------------------------------
# non_seizure_label=np.hstack([[len(i)*[0] for i in  non_seizure_All_features_]])

#------------------------Seizure Label + Non Seizure Label--------------------
label=[[len(i)*[0] for i in DWT_feature_non_seizure]]+[[len(i)*[1] for i in  DWT_feature_seizure]]
label=np.vstack(label)
label_DWT=label[:,0]

# -----------------------DWT entropy calculation (seizure)-----------------------------------

reshape_DWT_Coeffs=seizure_DWT_Coeffs_array.reshape([1000])
permutation_entropy=[]
spectral_entropy=[]
singular_v_d_entropy=[]
approximate_entropy=[]
sample_entropy=[]
hjorth=[]
no_z_c=[]

for a in range(0,1000):
    per=ent.perm_entropy(reshape_DWT_Coeffs[a], normalize=True)
    permutation_entropy.append(per)
    
    
for b in range(0,1000):
    spe=ent.spectral_entropy(reshape_DWT_Coeffs[b], sf=100, method='welch', normalize=True)
    spectral_entropy.append(spe)
    
    
for c in range(0,1000):
    sing=ent.svd_entropy(reshape_DWT_Coeffs[c], normalize=True)
    singular_v_d_entropy.append(sing)
    
    
for d in range(0,1000):
    appr=ent.app_entropy(reshape_DWT_Coeffs[d])
    approximate_entropy.append(appr)
    
    
for e in range(0,1000):
    samp=ent.sample_entropy(reshape_DWT_Coeffs[e])
    sample_entropy.append(samp)
    
    
for f in range(0,1000):
    hjo=ent.hjorth_params(reshape_DWT_Coeffs[f])
    hjorth.append(hjo)

for g in range(0,1000):
    hjo=ent.num_zerocross(reshape_DWT_Coeffs[g])
    hjorth.append(hjo)

for h in range(0,1000):
    nzc=ent.num_zerocross(reshape_DWT_Coeffs[h])
    no_z_c.append(nzc)
    

permutation_entropy=np.asarray(permutation_entropy)
permutation_entropy=permutation_entropy.reshape([200,5])  
   
spectral_entropy=np.asarray(spectral_entropy)
spectral_entropy=spectral_entropy.reshape([200,5])

singular_v_d_entropy=np.asarray(singular_v_d_entropy)
singular_v_d_entropy=singular_v_d_entropy.reshape([200,5])

approximate_entropy=np.asarray(approximate_entropy)
approximate_entropy=approximate_entropy.reshape([200,5])

sample_entropy=np.asarray(sample_entropy)
sample_entropy=sample_entropy.reshape([200,5])

hjorth=np.asarray(hjorth)
hjorth_complexity=np.asarray([float(x) for x in hjorth[1000:,]])
hjorth_mobility=np.asarray([float(x) for x in hjorth[1000:2000,]])
hjorth_complexity= hjorth_complexity.reshape([200,5])
hjorth_mobility=hjorth_mobility.reshape([200,5])

no_z_c=np.asarray(no_z_c)
no_z_c=no_z_c.reshape([200,5])

    
entropy_features=np.hstack((permutation_entropy,spectral_entropy,singular_v_d_entropy,approximate_entropy,sample_entropy,hjorth_complexity,hjorth_mobility,no_z_c))
entropy_features=entropy_features.reshape([200,5,5])
DWT_entropy_features_seizure=entropy_features.reshape([200,40])
DWT_entropy_features_seizure=np.asarray(DWT_entropy_features_seizure).astype(np.float)


#----------------------------DWT Fractal dimension (seizure)--------------------------------

petrosian_fd=[]
katz_fd=[]
higuchi_fd=[]
detrended_fa=[]

for i in range(0,1000):
    pet=ent.petrosian_fd(reshape_DWT_Coeffs[i])
    petrosian_fd.append(pet)
    
for j in range(0,1000):   
    katz=ent.katz_fd(reshape_DWT_Coeffs[j])
    katz_fd.append(katz)
    
for k in range(0,1000):    
    higuchi=ent.higuchi_fd(reshape_DWT_Coeffs[k])
    higuchi_fd.append(higuchi)
    
for l in range(0,1000):    
    detrended=ent.detrended_fluctuation(reshape_DWT_Coeffs[l])
    detrended_fa.append(detrended)
    
petrosian_fd=np.asarray(petrosian_fd)
petrosian_fd=petrosian_fd.reshape([200,5])    

katz_fd=np.asarray(katz_fd)
katz_fd=katz_fd.reshape([200,5])

higuchi_fd=np.asarray(higuchi_fd)
higuchi_fd=higuchi_fd.reshape([200,5])

detrended_fa=np.asarray(detrended_fa)
detrended_fa=detrended_fa.reshape([200,5])

fractal_dim_features=np.stack((petrosian_fd,katz_fd,higuchi_fd,detrended_fa),axis=0)
fractal_dim_features=fractal_dim_features.reshape([200,4,5])
DWT_fractal_dim_features_seizure=fractal_dim_features.reshape([200,20])

####Concatenate SEIZURE####

DWT_entropy_fractal_seizure_features=np.hstack((DWT_entropy_features_seizure,DWT_fractal_dim_features_seizure))
DWT_entropy_fractal_seizure_features=np.asarray(DWT_entropy_fractal_seizure_features).astype(float)
###############################################################################

# -----------------------DWT entropy calculation (non seizure)-----------------------------------

reshape_non_seizure_DWT_Coeffs=non_seizure_DWT_Coeffs_array.reshape([1000])
permutation_entropy=[]
spectral_entropy=[]
singular_v_d_entropy=[]
approximate_entropy=[]
sample_entropy=[]
hjorth=[]
no_z_c=[]

for a in range(0,1000):
    per=ent.perm_entropy(reshape_non_seizure_DWT_Coeffs[a], normalize=True)
    permutation_entropy.append(per)
    
    
for b in range(0,1000):
    spe=ent.spectral_entropy(reshape_non_seizure_DWT_Coeffs[b], sf=100, method='welch', normalize=True)
    spectral_entropy.append(spe)
    
    
for c in range(0,1000):
    sing=ent.svd_entropy(reshape_non_seizure_DWT_Coeffs[c], normalize=True)
    singular_v_d_entropy.append(sing)
    
    
for d in range(0,1000):
    appr=ent.app_entropy(reshape_non_seizure_DWT_Coeffs[d])
    approximate_entropy.append(appr)
    
    
for e in range(0,1000):
    samp=ent.sample_entropy(reshape_non_seizure_DWT_Coeffs[e])
    sample_entropy.append(samp)
    
    
for f in range(0,1000):
    hjo=ent.hjorth_params(reshape_non_seizure_DWT_Coeffs[f])
    hjorth.append(hjo)

for g in range(0,1000):
    hjo=ent.num_zerocross(reshape_non_seizure_DWT_Coeffs[g])
    hjorth.append(hjo)

for h in range(0,1000):
    nzc=ent.num_zerocross(reshape_non_seizure_DWT_Coeffs[h])
    no_z_c.append(nzc)
    

permutation_entropy=np.asarray(permutation_entropy)
permutation_entropy=permutation_entropy.reshape([200,5])  
   
spectral_entropy=np.asarray(spectral_entropy)
spectral_entropy=spectral_entropy.reshape([200,5])

singular_v_d_entropy=np.asarray(singular_v_d_entropy)
singular_v_d_entropy=singular_v_d_entropy.reshape([200,5])

approximate_entropy=np.asarray(approximate_entropy)
approximate_entropy=approximate_entropy.reshape([200,5])

sample_entropy=np.asarray(sample_entropy)
sample_entropy=sample_entropy.reshape([200,5])

hjorth=np.asarray(hjorth)
hjorth_complexity=np.asarray([float(x) for x in hjorth[1000:,]])
hjorth_mobility=np.asarray([float(x) for x in hjorth[1000:2000,]])
hjorth_complexity= hjorth_complexity.reshape([200,5])
hjorth_mobility=hjorth_mobility.reshape([200,5])

no_z_c=np.asarray(no_z_c)
no_z_c=no_z_c.reshape([200,5])

    
entropy_features1=np.stack((permutation_entropy,spectral_entropy,singular_v_d_entropy,approximate_entropy,sample_entropy,hjorth_complexity,hjorth_mobility,no_z_c),axis=0)
entropy_features1=entropy_features1.reshape([200,5,5])
DWT_entropy_features_non_seizure=entropy_features1.reshape([200,40])

DWT_ent_features=np.concatenate((DWT_entropy_features_non_seizure,DWT_entropy_features_seizure),axis=0)
#----------------------------DWT Fractal dimension (non seizure) --------------------------------

petrosian_fd=[]
katz_fd=[]
higuchi_fd=[]
detrended_fa=[]

for i in range(0,1000):
    pet=ent.petrosian_fd(reshape_non_seizure_DWT_Coeffs[i])
    petrosian_fd.append(pet)
    
for j in range(0,1000):   
    katz=ent.katz_fd(reshape_non_seizure_DWT_Coeffs[j])
    katz_fd.append(katz)
    
for k in range(0,1000):    
    higuchi=ent.higuchi_fd(reshape_non_seizure_DWT_Coeffs[k])
    higuchi_fd.append(higuchi)
    
for l in range(0,1000):    
    detrended=ent.detrended_fluctuation(reshape_non_seizure_DWT_Coeffs[l])
    detrended_fa.append(detrended)
    
petrosian_fd=np.asarray(petrosian_fd)
petrosian_fd=petrosian_fd.reshape([200,5])    

katz_fd=np.asarray(katz_fd)
katz_fd=katz_fd.reshape([200,5])

higuchi_fd=np.asarray(higuchi_fd)
higuchi_fd=higuchi_fd.reshape([200,5])

detrended_fa=np.asarray(detrended_fa)
detrended_fa=detrended_fa.reshape([200,5])

fractal_dim_features=np.stack((petrosian_fd,katz_fd,higuchi_fd,detrended_fa),axis=0)
fractal_dim_features=fractal_dim_features.reshape([200,4,5])
DWT_fractal_dim_features_non_seizure=fractal_dim_features.reshape([200,20])

DWT_fra_features=np.concatenate((DWT_fractal_dim_features_non_seizure,DWT_fractal_dim_features_seizure),axis=0)

####Concatenate NON SEIZURE####

DWT_entropy_fractal_non_seizure_features=np.hstack((DWT_entropy_features_non_seizure,DWT_fractal_dim_features_non_seizure))
DWT_entropy_fractal_non_seizure_features=np.asarray(DWT_entropy_fractal_non_seizure_features).astype(float)
###############################################################################
#--------------------IMF----------------------------------------------------#
##############################################################################
non_seizure_IMF_Coeffs_list=[];
def get_IMF(i,j):
    IMF_ns=emd.sift.sift(non_seizure_file_list[i][:,j])
    return IMF_ns
for i in range(0,50):
    for j in range(0,4):
        non_seizure_zzz=get_IMF(i,j)
        non_seizure_IMF_Coeffs_list.append(non_seizure_zzz)

# non_seizure_IMF_Coeffs_array=np.hstack(non_seizure_IMF_Coeffs_list);

#----------------------seizure IMF Coefficients Calculation--------------------
seizure_IMF_Coeffs_list=[];
def get_IMF(i,j):
    IMF_s=emd.sift.sift(seizure_file_list[i][:,j])
    return IMF_s
for i in range(0,50):
    for j in range(0,4):
        seizure_zzz=get_IMF(i,j)
        seizure_IMF_Coeffs_list.append(seizure_zzz)

# seizure_IMF_Coeffs_array=np.hstack(seizure_IMF_Coeffs_list);

#----------Taking Only five non_seizure IMF Coefficeient from Top--------------
five_non_seizure_IMF_Coeffs_list=[]
for i in range(0,200): 
    non_seizure=non_seizure_IMF_Coeffs_list[i][:,:5]
    five_non_seizure_IMF_Coeffs_list.append(non_seizure)
    five_non_seizure_IMF_Coeffs_array=np.asarray(five_non_seizure_IMF_Coeffs_list)

#----------Taking Only five seizure IMF Coefficeient from Top--------------
five_seizure_IMF_Coeffs_list=[]
for i in range(0,200): 
    seizure=seizure_IMF_Coeffs_list[i][:,:5]
    five_seizure_IMF_Coeffs_list.append(seizure) 
    five_seizure_IMF_Coeffs_array=np.asarray(five_seizure_IMF_Coeffs_list)
#---------------------IMF Statistical Features Calculations------------------------
def mean(data):
    return np.mean(data,axis=1)
    
def std(data):
    return np.std(data,axis=1)

def ptp(data):
    return np.ptp(data,axis=1)

def var(data):
        return np.var(data,axis=1)

def minim(data):
      return np.min(data,axis=1)


def maxim(data):
      return np.max(data,axis=1)

def argminim(data):
      return np.argmin(data,axis=1)


def argmaxim(data):
      return np.argmax(data,axis=1)

def mean_square(data):
      return np.mean(data**2,axis=1)

def rms(data): #root mean square
      return  np.sqrt(np.mean(data**2,axis=1))  

def abs_diffs_signal(data):
    return np.sum(np.abs(np.diff(data,axis=1)),axis=1)


def skewness(data):
    return stats.skew(data,axis=1)

def concatenate_features(data):
    return np.stack((mean(data),std(data),ptp(data),var(data),minim(data),maxim(data),argminim(data),argmaxim(data),mean_square(data),rms(data),abs_diffs_signal(data),skewness(data)),axis=1)

    
#-------------non_seizure_Features Calculation From IMF coefficients----------

non_seizure_IMF_stat_features=concatenate_features(five_non_seizure_IMF_Coeffs_array)
non_seizure_IMF_stat_features=non_seizure_IMF_stat_features.reshape([200,60])

 #--------------seizure_Features Calculation From IMF coefficients-------------
seizure_IMF_features=concatenate_features(five_seizure_IMF_Coeffs_array)
seizure_IMF_stat_features=seizure_IMF_features.reshape([200,60])

IMF_stat_features=np.concatenate((non_seizure_IMF_stat_features,seizure_IMF_features),axis=0)
IMF_stat_features=IMF_stat_features.reshape([400,96])
# IMF_stat_features=np.reshape(IMF_non_seizure_seizure_features,(400,60))
#------------------------Seizure Label + Non Seizure Label--------------------
# label=[[len(i)*[0] for i in  non_seizure_IMF_stat_features]]+[[len(i)*[1] for i in  seizure_IMF_stat_features]]
# label=np.vstack(label)
# label_IMF=label[:,0]


DWT_IMF_stat_features=np.concatenate((DWT_stat_features,IMF_stat_features),axis=1)
# label_DWT_IMF=np.concatenate((label_DWT,label_IMF),axis=0)


# -----------------------IMF entropy calculation (non-seizure)-----------------------------------
permutation_entropy=[]
spectral_entropy=[]
singular_v_d_entropy=[]
approximate_entropy=[]
sample_entropy=[]
hjorth=[]
no_z_c=[]

for a in range(0,200):
    for b in range(0,5):
        loop=eight_non_seizure_IMF_Coeffs_array[a,:,b]
        per=ent.perm_entropy(loop, normalize=True)
        permutation_entropy.append(per)
    
    
for c in range(0,200):
    for d in range(0,5):
        loop=eight_non_seizure_IMF_Coeffs_array[c,:,d]
        spe=ent.spectral_entropy(loop, sf=100, method='welch', normalize=True)
        spectral_entropy.append(spe)
    
for e in range(0,200):
    for f in range(0,5):
        loop=eight_non_seizure_IMF_Coeffs_array[e,:,f]
        sing=ent.svd_entropy(loop, normalize=True)
        singular_v_d_entropy.append(sing)
        
    
for g in range(0,200):
    for h in range(0,5):
        loop=eight_non_seizure_IMF_Coeffs_array[g,:,h]  
        appr=ent.app_entropy(loop)
        approximate_entropy.append(appr)
    
for i in range(0,200):
    for j in range(0,5):
        loop=eight_non_seizure_IMF_Coeffs_array[i,:,j]
        samp=ent.sample_entropy(loop)
        sample_entropy.append(samp)

for k in range(0,200):
    for l in range(0,5):
        loop=eight_non_seizure_IMF_Coeffs_array[k,:,l]
        hjo=ent.hjorth_params(loop)
        hjorth.append(hjo)
    
for m in range(0,200):
    for n in range(0,5):
        loop=eight_non_seizure_IMF_Coeffs_array[m,:,n]
        hjo=ent.num_zerocross(loop)
        hjorth.append(hjo)
  
for o in range(0,200):
    for p in range(0,5):
        loop=eight_non_seizure_IMF_Coeffs_array[o,:,p]
        nzc=ent.num_zerocross(loop)
        no_z_c.append(nzc)

permutation_entropy=np.asarray(permutation_entropy)
permutation_entropy=permutation_entropy.reshape([200,5])  
   
spectral_entropy=np.asarray(spectral_entropy)
spectral_entropy=spectral_entropy.reshape([200,5])

singular_v_d_entropy=np.asarray(singular_v_d_entropy)
singular_v_d_entropy=singular_v_d_entropy.reshape([200,5])

approximate_entropy=np.asarray(approximate_entropy)
approximate_entropy=approximate_entropy.reshape([200,5])

sample_entropy=np.asarray(sample_entropy)
sample_entropy=sample_entropy.reshape([200,5])

hjorth=np.asarray(hjorth)
hjorth_complexity=np.asarray([float(x) for x in hjorth[1000:,]])
hjorth_mobility=np.asarray([float(x) for x in hjorth[1000:2000,]])
hjorth_complexity= hjorth_complexity.reshape([200,5])
hjorth_mobility=hjorth_mobility.reshape([200,5])

no_z_c=np.asarray(no_z_c)
no_z_c=no_z_c.reshape([200,5])

    
entropy_features=np.stack((permutation_entropy,spectral_entropy,singular_v_d_entropy,approximate_entropy,sample_entropy,hjorth_complexity,hjorth_mobility,no_z_c),axis=0)
entropy_features=entropy_features.reshape([200,8,5])
IMF_entropy_features_non_seizure=entropy_features.reshape([200,40])

IMF_entropy_features=np.concatenate((IMF_entropy_features_non_seizure,IMF_entropy_features_seizure),axis=0)
#----------------------------IMF Fractal dimension non seizure --------------------------------

petrosian_fd=[]
katz_fd=[]
higuchi_fd=[]
detrended_fa=[]

for q in range(0,200):
    for r in range(0,5):
        loop=eight_non_seizure_IMF_Coeffs_array[q,:,r]
        pet=ent.petrosian_fd(loop)
        petrosian_fd.append(pet)
    
    
for u in range(0,200):
    for v in range(0,5):
        loop=eight_non_seizure_IMF_Coeffs_array[u,:,v]
        katz=ent.katz_fd(loop)
        katz_fd.append(katz)
    
    
for w in range(0,200):
    for x in range(0,5):
        loop=eight_non_seizure_IMF_Coeffs_array[w,:,x]
        higuchi=ent.higuchi_fd(loop)
        higuchi_fd.append(higuchi)
   
    
for y in range(0,200):
    for z in range(0,5):
        loop=eight_non_seizure_IMF_Coeffs_array[y,:,z]
        detrended=ent.detrended_fluctuation(loop)
        detrended_fa.append(detrended)
    
    
petrosian_fd=np.asarray(petrosian_fd)
petrosian_fd=petrosian_fd.reshape([200,5])    

katz_fd=np.asarray(katz_fd)
katz_fd=katz_fd.reshape([200,5])

higuchi_fd=np.asarray(higuchi_fd)
higuchi_fd=higuchi_fd.reshape([200,5])

detrended_fa=np.asarray(detrended_fa)
detrended_fa=detrended_fa.reshape([200,5])

fractal_dim_features=np.stack((petrosian_fd,katz_fd,higuchi_fd,detrended_fa),axis=0)
fractal_dim_features=fractal_dim_features.reshape([200,4,5])
IMF_fractal_dim_features_non_seizure=fractal_dim_features.reshape([200,20])


####Concatenate NON SEIZURE IMF####


IMF_entropy_fractal_non_seizure_features=np.hstack((IMF_entropy_features_non_seizure,IMF_fractal_dim_features_non_seizure))
IMF_entropy_fractal_non_seizure_features=np.asarray(IMF_entropy_fractal_non_seizure_features).astype(float)

DWT_IMF_entropy_features=np.concatenate((DWT_ent_features,IMF_entropy_features),axis=1)
###############################################################################
# -----------------------IMF entropy calculation (seizure)----------------------
###############################################################################
permutation_entropy=[]
spectral_entropy=[]
singular_v_d_entropy=[]
approximate_entropy=[]
sample_entropy=[]
hjorth=[]
no_z_c=[]

for a in range(0,200):
    for b in range(0,5):
        loop=eight_seizure_IMF_Coeffs_array[a,:,b]
        per=ent.perm_entropy(loop, normalize=True)
        permutation_entropy.append(per)
    
    
for c in range(0,200):
    for d in range(0,5):
        loop=eight_seizure_IMF_Coeffs_array[c,:,d]
        spe=ent.spectral_entropy(loop, sf=100, method='welch', normalize=True)
        spectral_entropy.append(spe)
    
for e in range(0,200):
    for f in range(0,5):
        loop=eight_seizure_IMF_Coeffs_array[e,:,f]
        sing=ent.svd_entropy(loop, normalize=True)
        singular_v_d_entropy.append(sing)
        
    
for g in range(0,200):
    for h in range(0,5):
        loop=eight_seizure_IMF_Coeffs_array[g,:,h]  
        appr=ent.app_entropy(loop)
        approximate_entropy.append(appr)
    
for i in range(0,200):
    for j in range(0,5):
        loop=eight_seizure_IMF_Coeffs_array[i,:,j]
        samp=ent.sample_entropy(loop)
        sample_entropy.append(samp)

for k in range(0,200):
    for l in range(0,5):
        loop=eight_seizure_IMF_Coeffs_array[k,:,l]
        hjo=ent.hjorth_params(loop)
        hjorth.append(hjo)
    
for m in range(0,200):
    for n in range(0,5):
        loop=eight_seizure_IMF_Coeffs_array[m,:,n]
        hjo=ent.num_zerocross(loop)
        hjorth.append(hjo)
  
for o in range(0,200):
    for p in range(0,5):
        loop=eight_seizure_IMF_Coeffs_array[o,:,p]
        nzc=ent.num_zerocross(loop)
        no_z_c.append(nzc)

permutation_entropy=np.asarray(permutation_entropy)
permutation_entropy=permutation_entropy.reshape([200,5])  
   
spectral_entropy=np.asarray(spectral_entropy)
spectral_entropy=spectral_entropy.reshape([200,5])

singular_v_d_entropy=np.asarray(singular_v_d_entropy)
singular_v_d_entropy=singular_v_d_entropy.reshape([200,5])

approximate_entropy=np.asarray(approximate_entropy)
approximate_entropy=approximate_entropy.reshape([200,5])

sample_entropy=np.asarray(sample_entropy)
sample_entropy=sample_entropy.reshape([200,5])

hjorth=np.asarray(hjorth)
hjorth_complexity=np.asarray([float(x) for x in hjorth[1000:,]])
hjorth_mobility=np.asarray([float(x) for x in hjorth[1600:2000,]])
hjorth_complexity= hjorth_complexity.reshape([200,5])
hjorth_mobility=hjorth_mobility.reshape([200,5])

no_z_c=np.asarray(no_z_c)
no_z_c=no_z_c.reshape([200,5])

    
entropy_features=np.stack((permutation_entropy,spectral_entropy,singular_v_d_entropy,approximate_entropy,sample_entropy,hjorth_complexity,hjorth_mobility,no_z_c),axis=0)
entropy_features=entropy_features.reshape([200,8,5])
IMF_entropy_features_seizure=entropy_features.reshape([200,40])



#################################################################################
#----------------------------IMF Fractal dimension seizure ---------------------
###################################################################################
petrosian_fd=[]
katz_fd=[]
higuchi_fd=[]
detrended_fa=[]

for q in range(0,200):
    for r in range(0,5):
        loop=eight_seizure_IMF_Coeffs_array[q,:,r]
        pet=ent.petrosian_fd(loop)
        petrosian_fd.append(pet)
    
    
for u in range(0,200):
    for v in range(0,5):
        loop=eight_seizure_IMF_Coeffs_array[u,:,v]
        katz=ent.katz_fd(loop)
        katz_fd.append(katz)
    
    
for w in range(0,200):
    for x in range(0,5):
        loop=eight_seizure_IMF_Coeffs_array[w,:,x]
        higuchi=ent.higuchi_fd(loop)
        higuchi_fd.append(higuchi)
   
    
for y in range(0,200):
    for z in range(0,5):
        loop=eight_seizure_IMF_Coeffs_array[y,:,z]
        detrended=ent.detrended_fluctuation(loop)
        detrended_fa.append(detrended)
    
    
petrosian_fd=np.asarray(petrosian_fd)
petrosian_fd=petrosian_fd.reshape([200,5])    

katz_fd=np.asarray(katz_fd)
katz_fd=katz_fd.reshape([200,5])

higuchi_fd=np.asarray(higuchi_fd)
higuchi_fd=higuchi_fd.reshape([200,5])

detrended_fa=np.asarray(detrended_fa)
detrended_fa=detrended_fa.reshape([200,5])

fractal_dim_features=np.stack((petrosian_fd,katz_fd,higuchi_fd,detrended_fa),axis=0)
fractal_dim_features=fractal_dim_features.reshape([200,4,5])
IMF_fractal_dim_features_seizure=fractal_dim_features.reshape([200,20])

vIMF_fractal_features=np.concatenate((IMF_fractal_dim_features_non_seizure,IMF_fractal_dim_features_seizure),axis=0)
DWT_IMF_fractal_features=np.concatenate((DWT_fra_features,IMF_fractal_features),axis=1)
####Concatenate SEIZURE IMF####


IMF_entropy_fractal_seizure_features=np.hstack((IMF_entropy_features_seizure,IMF_fractal_dim_features_seizure))
IMF_entropy_fractal_seizure_features=np.asarray(IMF_entropy_fractal_seizure_features).astype(float)


#---------------------------------FEATURE MATRIX (Three Matrix)------------#####

                      # DWT (Statistical,Entropy & Fractal Dimension Features)
feature_matrix_non_seizure=np.concatenate((DWT_stat_feature_non_seizure,DWT_entropy_fractal_non_seizure_features),axis=1)
feature_matrix_seizure=np.concatenate((DWT_stat_feature_seizure,DWT_entropy_fractal_seizure_features),axis=1)
feature_matrix_DWT=np.concatenate((feature_matrix_non_seizure,feature_matrix_seizure),axis=0)


                      # EMD (Statistical,Entropy & Fractal Dimension Features)

feature_matrix_non_seizure=np.concatenate((non_seizure_IMF_stat_features,IMF_entropy_fractal_non_seizure_features),axis=1)
feature_matrix_seizure=np.concatenate((seizure_IMF_stat_features,IMF_entropy_fractal_seizure_features),axis=1)
feature_matrix_EMD=np.concatenate((feature_matrix_non_seizure,feature_matrix_seizure),axis=0)

                # DWT + EMD (Statistical,Entropy & Fractal Dimension Features)

feature_matrix_non_seizure=np.concatenate((DWT_stat_feature_non_seizure,non_seizure_IMF_stat_features,DWT_entropy_fractal_non_seizure_features,IMF_entropy_fractal_non_seizure_features),axis=1)
feature_matrix_seizure=np.concatenate((DWT_stat_feature_seizure,seizure_IMF_stat_features,DWT_entropy_fractal_seizure_features,IMF_entropy_fractal_seizure_features),axis=1)
feature_matrix_DWT_EMD=np.concatenate((feature_matrix_non_seizure,feature_matrix_seizure),axis=0)


#----------------LABEL OF FEATURE MATRIX---------------------------------------

label=[[len(i)*[0] for i in  feature_matrix_non_seizure]]+[[len(i)*[1] for i in  feature_matrix_seizure]]
label_non_seizure=label=np.vstack(label)
feature_matrix_label=label[:,0]