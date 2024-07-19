from soup.soup import soup
import numpy as np 
from .config import HR_TIME_CONFIG
from .config import STAGES

class HRTime(soup):
    def __init__(self,
                 config=None,
                 stages=None):
        """ Class Constructor. The parent class is initialized with the configuration dictionary for the HR in the time domain feature.

        Args:
            config (dict): Configuration dictionary with the conditions to check. Default None, if None the default configuration is used.
            stages (dict): Configuration dictionary with the sleep stages. Default None, if None the default stages are used.
        """
        if stages is None:
            stages = STAGES
        if config is None:
            config = HR_TIME_CONFIG
        super().__init__(config, stages)


    def _check_hr_increase_rem(self,data,config, condition_name='hr_increase_rem'):
        """ Check if heart rate increases during the REM stage.

        Args:
            data (pd.DataFrame): DataFrame with the relevant columns: HR and Stage.
            config (dict): Configuration dictionary with the needed parameters to check the condition. 'alpha' key is mandatory.
        """
        # Remove awake 
        act_data = data.loc[data['Stage'] != self.stages['AWAKE']]
        # Combie
        act_data.loc[:,'Stage'] = self._combine_stages(act_data['Stage'],'ns')
        # TEST
        test_results = self.stats.pairwaise_comparisson(act_data[['HR','Stage']],alpha=config['alpha'])
        # Keep REM 
        act_data = act_data.loc[act_data['Stage'] == self.stages['REM']]

        rem_increase = self._check_hr_increase_by_phase(act_data,stage=self.stages['REM'],config=config)
        act_percent = np.round(np.sum(rem_increase) / len(rem_increase) * 100,2)
        
        self._save_results(condition_name, 
                            test_results['H0_Rejected'].iloc[0] == 'True',
                            test_results['P-Value'].iloc[0],
                            act_percent>config['threshold'])
    

    def _check_hr_decrease_nrem(self,data,config, condition_name='hr_decrease_nrem'):
        """ Check if heart rate decreases during NREM stage.

        Args:
            data (pd.DataFrame): DataFrame with the relevant columns: HR and Stage.
            config (dict): Configuration dictionary with the needed parameters to check the condition. 'alpha' key is mandatory.
        """
        act_data = data.loc[data['Stage'] != self.stages['AWAKE']]
        act_data.loc[:,'Stage'] = self._combine_stages(act_data['Stage'],'ns')
        test_results = self.stats.pairwaise_comparisson(act_data[['HR','Stage']],alpha=config['alpha'])
        act_data = act_data.loc[act_data['Stage'] == self.stages['N1']]

        nrem_decrease = self._check_hr_increase_by_phase(act_data,stage=self.stages['N1'],config=config)
        nrem_decrease = ~ nrem_decrease
        act_percent = np.round(np.sum(nrem_decrease) / len(nrem_decrease) * 100,2)
        
        self._save_results(condition_name, 
                            test_results['H0_Rejected'].iloc[0] == 'True',
                            test_results['P-Value'].iloc[0],
                            act_percent>config['threshold'])


    def _check_hr_increase_by_phase(self,data,stage,config):
        """ Private function to check if heart rate increases during a specific stage.

        Args:
            data (pd.DataFrame): DataFrame with the relevant columns: HR and Stage.
            stage (int): Stage to check the condition.
            config (dict): Configuration dictionary with the needed parameters to check the condition. 'min_samples' key is mandatory, indicating the minimum number of samples to consider a stage.
        
        Returns:
            np.array: Array with as many samples as time slots of the specified stage. Contains True if the heart rate increases during that slot, False otherwise.
        """
        act_data = data.loc[data['Stage']==stage]
        
        data_slots = self._get_intervals(act_data.index)
        data_increase = []
        for slot in data_slots:
            if slot[1]-slot[0] < config['min_samples']:
                continue
            act_stage = act_data.loc[slot[0]:slot[1]]['HR'].values
            data_increase.append(act_stage[0] < act_stage[-1])
    
        return np.array(data_increase)