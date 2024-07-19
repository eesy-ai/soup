from pyhrv.frequency_domain import ar_psd
import numpy as np 
import pandas as pd 


def get_HF(nni):
    try:
        output = ar_psd(nni.values, mode='dev')[0]['ar_abs']
        hf = output[2]
    except:
        hf = np.nan
    
    return hf

def get_LF(nni):
    try:
        output = ar_psd(nni=nni.values, mode='dev')[0]['ar_abs']
        lf = output[1]
    except:
        lf = np.nan
    
    return lf

def _pass_variation_criterion(value,average, thredshold):
    return np.abs(value - average) < thredshold


def fix_variation_outliers(data_,
                           window_size = 5, # samples
                           window_step = 1,
                           min_windows=1,
                           max_iters=5,
                           data_normalized=False):
    data = data_.copy()

    outliers = []
    n_iter = 0 
    while n_iter < max_iters:
        act_outliers = []
        for i in range(0,len(data)-window_size,window_step):
            act_av = np.mean(data[i:i+window_size])
            act_thredshold = act_av *0.3

            for j in range(i,i+window_size):
                if not _pass_variation_criterion(data[j],act_av,act_thredshold):
                    act_outliers.append(j)
        act_outliers, counts = np.unique(act_outliers,return_counts=True)
        act_outliers = act_outliers[counts>min_windows]
        act_outliers[~np.isin(act_outliers,outliers)]

        if len(act_outliers) == 0:
            break 

        data[act_outliers] = np.nan
        data = pd.DataFrame(data).interpolate().values

        outliers += list(act_outliers)
        n_iter+= 1
    return data



