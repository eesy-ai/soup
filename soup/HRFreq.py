from soup.soup import soup
import numpy as np 
from .config import HR_FREQ_CONFIG
from .config import STAGES

class HRFreq(soup):
    def __init__(self,
                 config=None,
                 stages=None):
        """ Class Constructor. The parent class is initialized with the configuration dictionary for the HR in the frequency domain feature.

        Args:
            config (dict): Configuration dictionary with the conditions to check. Default None, if None the default configuration is used.
            stages (dict): Configuration dictionary with the sleep stages. Default None, if None the default stages are used.
        """
        if stages is None:
            stages = STAGES
        if config is None:
            config = HR_FREQ_CONFIG
        super().__init__(config,stages)

    
    def _check_ratio_lowest_n3(self, data, config, condition_name='ratio_lowest_n3'):
        """ Check if the lowest LF:HF ratio is found in the N3 stage.
        
        Args:
            data (pd.DataFrame): DataFrame with the relevant columns: LF:HF and Stage.
            config (dict): Configuration dictionary with the needed parameters to check the condition. 'alpha' key is mandatory.
        """

        # Remove awake
        act_data = data.loc[data['Stage'] != self.stages['AWAKE']]

        # Average
        mean_ratio = act_data.groupby('Stage')['LF:HF'].mean()
        n3_index = mean_ratio.index.get_loc(self.stages['N3'])
        
        # Remove N2 for statistical analysis 
        act_data = act_data.loc[data['Stage'] != self.stages['N2']]
        
        test_results = self.stats.pairwaise_comparisson(act_data[['LF:HF','Stage']], alpha=config['alpha'])
        test_results = test_results.loc[test_results['groups'].isin(self._get_combinations(np.arange(1,6),self.stages['N3']))]

        self._save_results(condition_name, 
                           (test_results['H0_Rejected'] == 'True').all(),
                           test_results['P-Value'],
                           np.argmin(mean_ratio)==n3_index)


