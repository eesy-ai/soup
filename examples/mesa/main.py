from pathlib import Path
import pandas as pd 
import sys, os

INPUT_PATH = Path('examples/mesa/data')
OUTPUT_PATH = Path('examples/mesa/output')

sys.path.insert(1, str(Path(os.path.dirname(sys.argv[0]), '../..')))
from soup.HRTime import HRTime
from soup.AC import AC
from soup.HRFreq import HRFreq
from soup.config import STAGES

################################################################################################
# Load data
################################################################################################
data = pd.read_csv(INPUT_PATH / 'mesa_features_30S.csv')
clinical = pd.read_csv(INPUT_PATH / 'mesa_clinical.csv',index_col=0)

################################################################################################
# Ensure data stage is integer
################################################################################################

data['Stage'] = data['Stage'].astype(int)

################################################################################################
# Single patient
################################################################################################
patients_ids = [27,275]
for act_id in patients_ids:
    act_data = data.loc[data['id'] == act_id]
    for key in ['HRFreq','AC','HRTime']:
        if key == 'AC':
            tool = AC()
        elif key == 'HRTime':
            tool = HRTime()
        elif key == 'HRFreq':
            tool = HRFreq()

        act_output_path = OUTPUT_PATH / str(act_id) 
        if not os.path.exists(act_output_path):
            os.makedirs(act_output_path)

        tool.run( act_data, clinical=None, output_path = act_output_path)


################################################################################################
# Full dataset
################################################################################################

for key in ['HRTime', 'HRFreq', 'AC']:
    print(key)
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    if key == 'AC':
        tool = AC()
    elif key == 'HRTime':
        tool = HRTime()
    elif key == 'HRFreq':
        tool = HRFreq()

    act_output_path = OUTPUT_PATH / key 
    if not os.path.exists(act_output_path):
        os.makedirs(act_output_path)

    tool.run( data, clinical, output_path = act_output_path)
    
    
'''
data = data.loc[data['id'].isin(patients_checked)]
data = data.loc[data['Stage'].isin([1,2,3,5])]
# Calculate frequency of each stage
stage_counts = data['Stage'].value_counts()

aux = data.merge(clinical,on='id')

stage_counts = aux.groupby(['age', 'Stage']).size().reset_index(name='Count')
total_counts = aux.groupby('age').size().reset_index(name='TotalCount')
stage_counts = pd.merge(stage_counts, total_counts, on='age')
stage_counts['Percentage'] = (stage_counts['Count'] / stage_counts['TotalCount']) * 100
from Tool.config import STAGES 
stage_labels = {v: k for k, v in STAGES.items()}
# Replace integers in the 'Stage' column with corresponding string labels
data['Stage'] = data['Stage'].map(stage_labels)

# Calculate frequency of each stage
stage_counts = data['Stage'].value_counts()

import matplotlib.pyplot as plt
# Plot pie chart
fig, ax = plt.subplots()
ax.pie(stage_counts, labels=stage_counts.index, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12})
ax.axis('equal')  # Equal aspect ratio ensures the pie is drawn as a circle.

plt.title('Distribution of Sleep Stages',fontsize=15)
plt.savefig('cake.png')
plt.show()


# Calculate frequency of each stage by ID
stage_counts = data.groupby(['id', 'Stage']).size().reset_index(name='Count')

# Calculate total counts (total stages per ID)
total_counts = stage_counts.groupby('id')['Count'].sum()

# Calculate percentage of each stage by ID
stage_counts['Percentage'] = stage_counts.apply(lambda row: row['Count'] / total_counts[row['id']] * 100, axis=1)

stage_counts.groupby('Stage')['Percentage'].mean()
stage_counts.groupby('Stage')['Percentage'].std()
'''