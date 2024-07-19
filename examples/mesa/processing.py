from pathlib import Path
import pandas as pd
import os
import numpy as np
from datetime import datetime
from utils import get_HF, get_LF, fix_variation_outliers
from progress.bar import ChargingBar
from scipy.signal import lombscargle



#####################################################################################
# Path and global variables definition
#####################################################################################
INPUT_PATH = Path('../mesa_raw/') # Update with the path where MESA dataset was download
OUTPUT_PATH = Path('examples/mesa/data/')
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

DISEASES = ['apnea35', 'rstlesslgs5', 'slpapnea5', 'insmnia5']
CLINICAL2KEEP = ['disease','race1c','sleepage5c','mesaid','gender1']

input_acti = INPUT_PATH / 'actigraphy'
input_poli = INPUT_PATH / 'polysomnography/annotations-rpoints'
features_30S = pd.DataFrame([])


#####################################################################################
# Load data
#####################################################################################
overlap = pd.read_csv(INPUT_PATH / 'overlap/mesa-actigraphy-psg-overlap.csv')
clinical = pd.read_csv(INPUT_PATH / 'datasets/mesa-sleep-dataset-0.6.0.csv')

#####################################################################################
# Keep just synchronized patients
#####################################################################################
clinical = clinical.loc[clinical['mesaid'].isin(overlap['mesaid'])]

#####################################################################################
# Process patients
#####################################################################################
mins = []

with ChargingBar('\rProcessing MESA patients...', suffix='%(percent).1f%% - %(eta)ds',max=len(clinical)) as bar:
    for index, row in clinical.iterrows():
        bar.next()
        # Skip if no bed time specified
        if not isinstance(row['bedtmwkday5c'],str) and np.isnan(row['bedtmwkday5c']):
            continue
        
        patient = str(row['mesaid'])
        act_overlap = overlap.loc[overlap['mesaid'] == row['mesaid']]

        #####################################################################################
        # Read (if exists) polisomnography 
        #####################################################################################
        # Same length for ids
        while len(patient) < 4:
            patient = '0' + patient
        # print(patient)
        try:
            poli_file = pd.read_csv(input_poli / ('mesa-sleep-' + patient + '-rpoint.csv'))[['seconds', 'stage', 'epoch']]
        except:
            continue

        #####################################################################################
        # Read and filter HR data
        ######################################################################################
        if not poli_file['seconds'].is_monotonic_increasing:
            continue
        ibi = poli_file['seconds'].diff().values[1:]
        ibi = fix_variation_outliers(ibi.reshape(-1,1))
        hr = 60 / ibi
        hrv = np.append([np.nan],np.diff(ibi,axis=0))

        act_data = pd.DataFrame(dict(IBI=ibi.flatten(), 
                                     HR=hr.flatten(), 
                                     HRV=hrv.flatten(), 
                                     Stage=poli_file['stage'].iloc[1:].values, 
                                     epoch=poli_file['epoch'].iloc[1:].values),
                                index = poli_file['seconds'].iloc[1:].values)
        act_data.index = pd.to_timedelta(act_data.index,unit='s')
        
        act_data_hr = act_data.groupby('epoch').mean()
        act_data_hr['HRV'] = np.abs(act_data_hr['HRV'])


        act_data_hr[['LF','HF']] = act_data.groupby('epoch')['IBI'].agg([get_LF,get_HF])

        act_data_hr['LF:HF'] = act_data_hr['LF'].values / act_data_hr['HF'].values
        #####################################################################################
        # Read and filter activity data
        #####################################################################################
        acti_file = pd.read_csv(input_acti / ('mesa-sleep-' + patient + '.csv'))
        acti_file = acti_file[['linetime', 'activity', 'wake']]
        acti_file['linetime'] = acti_file['linetime'].apply(lambda x: datetime.strptime(x, '%H:%M:%S').time())

        # Find the first overlap
        acti_file = acti_file.iloc[act_overlap['line'].iloc[0] -1 :]  # Until the end of the hypnogram
        acti_file = acti_file.iloc[:len(act_data_hr)]
        acti_file.rename(columns={'activity': 'AC'}, inplace=True)
        # acti_file['AC'] = get_acitivity_level(acti_file['activity'])

        #####################################################################################
        # Concatenate
        #####################################################################################
        act_data = pd.concat([act_data_hr.reset_index(), acti_file.reset_index()], axis=1)
        act_data.set_index('linetime', inplace=True)

        #####################################################################################
        # Keep just bed time
        #####################################################################################
        night_i0 = np.where(act_data.index >= datetime.strptime(row['bedtmwkday5c'], '%H:%M:%S').time())[0]
        # No bed time
        if len(night_i0) == act_data.shape[0] or len(night_i0) == 0:
            continue
        night_i0 = night_i0[0]
        night_i1 = np.where(acti_file['linetime'] <= datetime.strptime(row['waketmwkday5c'], '%H:%M:%S').time())[0]
        act_data = act_data.iloc[night_i0:night_i1[-1]+1]

        #####################################################################################
        # Stages 3 and 4 are N3
        #####################################################################################
        act_data.loc[act_data['Stage'] == 4,'Stage'] = 3

        #####################################################################################
        # Add id
        #####################################################################################
        act_data['id'] = row['mesaid']

        #####################################################################################
        # Clean and save
        #####################################################################################
        # Drop generated nan values
        if act_data.isna().any().any():
            act_data = act_data.dropna()
        # Drop useless columns
        act_data = act_data.drop(columns=[ 'index', 'wake', 'epoch'])
        # Save global info
        features_30S = pd.concat([features_30S, act_data])
        features_30S.to_csv(OUTPUT_PATH / 'mesa_features_30S.csv')

features_30S.to_csv(OUTPUT_PATH / 'mesa_features_30S.csv')

#####################################################################################
# Process relevant clinical information
#####################################################################################
# Keep just synchronized patients
clinical = clinical.loc[clinical['mesaid'].isin(features_30S['id'].unique())]

# Add disease column
clinical['disease'] = 'healthy'
for d in DISEASES:
    clinical.loc[clinical[d] == 1,'disease'] = d

# Keep just relevant columns
clinical = clinical[CLINICAL2KEEP]

# Rename columns
clinical.rename(columns={'mesaid':'id', 
                         'sleepage5c':'age',
                         'race1c':'race',
                         'gender1':'sex'},
                inplace=True)

# Discrete age for disemination
clinical.loc[clinical['age'] <= 64,'age'] = 0 # Early Senior
clinical.loc[(64 < clinical['age']) & (clinical['age'] <= 74),'age'] = 1 # Middle Senior
clinical.loc[(74 < clinical['age']) & (clinical['age'] <= 84),'age'] = 2 # Late Senior
clinical.loc[clinical['age'] >= 85,'age'] = 3  # Advanced Senior
clinical['age'] = clinical['age'].replace({0: '[54-64]', 1: '[65-74]', 2: '[75-84]', 3: '[85-94]'})

# Decode race for disemination
clinical.loc[clinical['race']==1, 'race'] = 'White'
clinical.loc[clinical['race']==2, 'race'] = 'Chinese American' 
clinical.loc[clinical['race']==3, 'race'] = 'Black, African-American'
clinical.loc[clinical['race']==4, 'race'] = 'Hispanic'

# Decode sex for disemination
clinical.loc[clinical['sex']==0, 'sex'] = 'Female'
clinical.loc[clinical['sex']==1, 'sex'] = 'Male'

# Rename disesases for disemination
clinical.loc[clinical['disease']=='slpapnea5', 'disease'] = 'Sleep Apnea'
clinical.loc[clinical['disease']=='apnea35', 'disease'] = 'Sleep Apnea'
clinical.loc[clinical['disease']=='rstlesslgs5', 'disease'] = 'Restless Legs'
clinical.loc[clinical['disease']=='insmnia5', 'disease'] = 'Insomnia'
# Save clinical data
clinical.to_csv(OUTPUT_PATH / 'mesa_clinical.csv')
