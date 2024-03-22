from scipy.signal import filtfilt
from scipy import stats
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import scipy
import math 

def bandpassfilter(signal):
    
    fs=256
    low_cut=0.1
    high_cut=30
    
    nyq=0.5*fs
    low=low_cut/nyq
    high=high_cut/nyq
    
    order=3
    
    b,a=scipy.signal.butter(order,[low,high],'bandpass',analog=False)
    y=scipy.signal.filtfilt(b,a,signal,axis=0)
    return(y)

    

