from pathlib import Path
import pandas as pd 
import sys, os
sys.path.insert(1, str(Path(os.path.dirname(sys.argv[0]), '../..')))
from soup.HRTime import HRTime
from soup.AC import AC
from soup.HRFreq import HRFreq

################################################################################################
# Global variables definition
################################################################################################
INPUT_PATH = Path('examples/plug&play/input/')
OUTPUT_PATH = Path('examples/plug&play/output/')

STAGES = {
    "N1": 3,
    "N2": 2,
    "N3": 1,
    "REM": 4,
    "AWAKE": 5
}

################################################################################################
# Load data
################################################################################################
data = pd.read_csv(INPUT_PATH / 'features.csv')
clinical = pd.read_csv(INPUT_PATH / 'clinical.csv')


################################################################################################
# Ensure data stage is integer
################################################################################################
data['Stage'] = data['Stage'].astype(int)


################################################################################################
# Full dataset
################################################################################################

for key in ['AC','HRTime','HRFreq']:
    print(key)
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    if key == 'AC':
        tool = AC(stages=STAGES)
    elif key == 'HRTime':
        tool = HRTime(stages=STAGES)
    elif key == 'HRFreq':
        tool = HRFreq(stages=STAGES)

    act_output_path = OUTPUT_PATH / key 
    if not os.path.exists(act_output_path):
        os.makedirs(act_output_path)

    tool.run(data, clinical=clinical, output_path = act_output_path)


