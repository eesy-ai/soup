# the keys should remain the same, the number can change but are expected to be in [1,5].
STAGES = {
    "N1": 1,
    "N2": 2,
    "N3": 3,
    "REM": 5,
    "AWAKE": 0
}


# [5] Boudreau2013CircadianStages
# [6] scholz1997vegetative
# [7] stein2008cardiac
HR_FREQ_CONFIG = {
    "rel_cols": ["LF","HF", "LF:HF", "Stage", "id"],
    "conditions": {
        "hf_n3_highest": { 
            "description": "The is highest HF power during NREM is found in N3 sleep [?]",
            "alpha": 0.05,
            "generate":{
                "operation": "max",
                "column": "HF", 
                "stage": "N3",
                "stage2compare": "",
                "stages2ignore":["REM","AWAKE"],
                "groupby":"",
            }
        },
        "lf_nrem_lower_awake": {
            "description": "LF power is lower during NREM sleep compared to awakeness [?]",
            "alpha": 0.05,
            "generate":{
                "operation": "min",
                "column": "LF", 
                "stage": "N1",
                "stage2compare": "AWAKE",
                "stages2ignore":["REM"],
                "groupby":"ns",
            }
        },
        "lf_nrem_lower_rem": { 
            "description": "LF power is lower during NREM sleep compared to REM sleep [?]",
            "alpha": 0.05,
            "generate": {
                "operation": "min",
                "column": "LF", 
                "stage": "N1",
                "stage2compare": "REM",
                "stages2ignore":["AWAKE"],
                "groupby":"ns",
            }
        },
        "lf_lowest_n3": {  
            "description": "",
            "alpha": 0.05, 
            "generate": {
                "operation": "min",
                "column": "LF", 
                "stage": "N3",
                "stage2compare": "",
                "stages2ignore":[],
                "groupby":"",
            }
        },
        "ratio_lowest_n3": {  
            "description": "The lowest LF:HF ratio is found in the N3 stage [5][7]. Statistically significant with all sleep stages but N2. LF/HF decrease with the deepeining of sleep[?]",
            "alpha": 0.05
        },
        "ratio_rem_lower_awake": {  
            "description": "LF:HF ratio is higher during awakeness compared to the REM stage [5]",
            "alpha": 0.05, 
            "generate": {
                "operation": "min",
                "column": "LF:HF", 
                "stage": "REM",
                "stage2compare": "AWAKE",
                "stages2ignore":["N1","N2","N3"],
                "groupby":"",
            }
        },
        "ratio_rem_higher_nrem": {  
            "description": "LF:HF ratio during the REM stage is expected to be higher than in NREM stages [5][6]",
            "alpha": 0.05, 
            "generate": {
                "operation": "max",
                "column": "LF:HF", 
                "stage": "REM",
                "stage2compare": "N1",
                "stages2ignore":["AWAKE"],
                "groupby":"ns",
            }
        },
    },
}

# [4] Stein2012HeartDisorders
HR_TIME_CONFIG = {
    "rel_cols": ["HR","HRV","Stage", "id"],
    "conditions": {
        "hr_decrease_nrem": { 
            "description": "heart rate decreases during NREM stage [4]",
            "alpha": 0.05,
            "threshold": 0.1,
            "min_samples": 2
        },
        "hr_increase_rem": { 
            "description": "heart rate increases during the REM stage [4]",
            "alpha": 0.05,
            "threshold": 0.1,
            "min_samples": 2
        },
        "hrv_nrem_lower_rem": {  
            "description": "HR more stable during NREM [?]",
            "alpha": 0.05, 
            "generate": {
                "operation": "min",
                "column": "HRV",
                "stage": "N1",
                "stage2compare": "REM",
                "stages2ignore":["AWAKE"],
                "groupby":"ns",
            }
        },
        "hrv_awake_higher_nrem":{
            "description": "HRV during NREM sleep is lower than during awakeness [?]",
            "alpha": 0.05, 
            "generate": {
                "operation": "max",
                "column": "HRV", 
                "stage": "AWAKE",
                "stage2compare": "N1",
                "stages2ignore":["REM"],
                "groupby":"ns",
            }
        },
        "hr_awake_higher_nrem": {
            "description": "When moving from wake stage to deeper sleep, heart rate decrease [2]",
            "alpha": 0.05, 
            "generate": {
                "operation": "max",
                "column": "HR", 
                "stage": "AWAKE",
                "stage2compare": "N1",
                "stages2ignore":["REM"],
                "groupby":"ns",
            }
        },
        "hr_rem_higher_nrem": {  
            "description": "HR increased, with increased variability, in REM sleep [4][?]",
            "alpha": 0.05, 
            "generate": {
                "operation": "max",
                "column": "HR", 
                "stage": "REM",
                "stage2compare": "N1",
                "stages2ignore":["AWAKE"],
                "groupby":"ns",
            }
        },
    },
    
}

# Configuration for AC tool
# [1] Gaiduk2018AutomaticSignals
# [2] Gaiduk2022EstimationSignals
# [3] ogilvie1989detection
AC_CONFIG = {
    "rel_cols": ["AC", "Stage", "id"],
    "conditions": {
        "ac_nrem_smaller_awake": { 
            "description": "NREM stages generally exhibit smaller movements compared to awake states [2]",
            "alpha": 0.05, 
            "generate":{
                "operation": "min",
                "column": "AC", 
                "stage": "N1",
                "stage2compare": "AWAKE",
                "stages2ignore":["REM"],
                "groupby":"ns",
            }
        },
        "ac_awake_highest": { 
            "description": "Highest activity is observed during awakeness [2]",
            "alpha": 0.01, 
            "generate":{
                "operation": "max",
                "column": "AC", 
                "stage": "AWAKE",
                "stage2compare": "",
                "stages2ignore":[],
                "ignore_stages":"",
                "groupby":"",
            }
        },
        "ac_awake_sim_n1": {  
            "description": "Awake and N1 stages exhibit the highest similarity [3]",
            "alpha": 0.01, 
            "generate":{
                "operation": "sim",
                "column": "AC", 
                "stage": "AWAKE",
                "stage2compare": "N1",
                "stages2ignore":[],
                "groupby":"",
            }
        },
        "ac_n1_highest": { 
            "description": "Activity levels in the N1 stage are higher than in other sleep stages [1]",
            "alpha": 0.05, 
            "generate":{
                "operation": "max",
                "column": "AC", 
                "stage": "N1",
                "stage2compare": "",
                "stages2ignore":["AWAKE"],
                "groupby":"",
            }
        },
        "ac_nrem_less_freq_awake": { 
            "description": "When sleep deepens from wake stage, body movements become smaller and less frequent [2]",
            "alpha": 0.05
        },
        "ac_post_prev_rem_higher": { 
            "description": "Body movement is tipical for the epochs just before and directly after REM stage [2]",
            "alpha": 0.05,
            "n_epochs": 8   
        },
    }
}