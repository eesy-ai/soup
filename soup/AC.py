from soup.soup import soup
import numpy as np 
import pandas as pd 
from .config import AC_CONFIG
from .config import STAGES


class AC(soup):
    def __init__(self,
                 config=None,
                 stages=None):
        """ Class Constructor. The parent class is initialized with the configuration dictionary for the AC feature.
        
        Args:
            config (dict, optional): Configuration dictionary with the conditions to check. Default None, if None the default configuration is used.
            stages (dict, optional): Configuration dictionary with the sleep stages. Default None, if None the default stages are used.
        """
        if stages is None:
            stages = STAGES
        if config is None:
            config = AC_CONFIG
        super().__init__(config, stages)
   

    def _check_ac_nrem_smaller_awake(self,data, config, condition_name='ac_nrem_smaller_awake'):
        """ Check if the NREM stages have a smaller activity than the awake stage. 
        
        Args:
            data (pd.DataFrame): DataFrame with the relevant columns: AC and Stage.
            config (dict): Configuration dictionary with the needed parameters to check the condition. 'alpha' key is mandatory.
        """
        act_data = data.loc[data['Stage'].isin([ value for key,value in self.stages.items() if key != 'REM'])]
        act_data.loc[:,'Stage'] = self._combine_stages(act_data['Stage'],'ns')
        test_results = self.stats.pairwaise_comparisson(act_data[['AC','Stage']],alpha=config['alpha'])
    
        mean_ac = act_data.groupby('Stage')['AC'].mean()
        nrem_index = mean_ac.index.get_loc(self.stages['N1'])
        
        self._save_results(condition_name, 
                           test_results['H0_Rejected'].iloc[0] == 'True',
                           test_results['P-Value'].iloc[0],
                           np.argmin(mean_ac)==nrem_index)

    def _check_ac_nrem_less_freq_awake(self,data, config, condition_name='ac_nrem_less_freq_awake'):
        """ Check if the activity during NREM stages have a lower frequency than during awake stage.
        
        Args:
            data (pd.DataFrame): DataFrame with the relevant columns: AC and Stage.
            config (dict): Configuration dictionary with the needed parameters to check the condition. 'alpha' key is mandatory.
            
        """
        # Remove REM
        act_data = data.loc[ data['Stage'] != self.stages['REM']]
        act_data.loc[:,'Stage'] = self._combine_stages(act_data['Stage'],'ns')
        test_results = self.stats.pairwaise_comparisson(act_data[['AC','Stage']],alpha=config['alpha'])

        ac_frequency = act_data.groupby('Stage')['AC'].apply(lambda x: (x != 0).mean()).reset_index()
        ac_frequency.columns = ['Stage', 'AC_Frequency']

        nrem_index = ac_frequency.loc[ac_frequency['Stage'] == self.stages['N1']].index[0]

        self._save_results(condition_name, 
                           test_results['H0_Rejected'].iloc[0] == 'True', 
                           test_results['P-Value'].iloc[0],
                           np.argmin(ac_frequency['AC_Frequency'])==nrem_index)
        
    def _check_ac_post_prev_rem_higher(self,data, config, condition_name='ac_post_prev_rem_higher'):
        """ Check if there is a higher activity right before and right after the REM satage. 
        
        Args:
            data (pd.DataFrame): DataFrame with the relevant columns: AC and Stage.
            config (dict): Configuration dictionary with the needed parameters to check the condition. 'alpha' key is mandatory.
            
        """
        act_data = data.copy()
        act_data.loc[:,'Stage'] = self._combine_stages(act_data['Stage'],'ns')
        act_data = self._get_prev_post_rem_groups(act_data,n_epochs=config['n_epochs'])
            
        test_results = self.stats.pairwaise_comparisson(act_data[['AC','key']],group_id_key='key',alpha=config['alpha'])
        test_results = test_results.loc[test_results['groups'].isin(['other_sleep-post','other_sleep-prev'])]
        
        mean_ac = act_data.groupby('key').mean()
        other_index = mean_ac.index.get_loc('other_sleep')

        self._save_results(condition_name, 
                           np.all(test_results['H0_Rejected'] == 'True'), 
                           test_results['P-Value'].values,
                           np.argmax(mean_ac)!=other_index)
    


    def _get_prev_post_rem_groups(self,data, n_epochs=4):
        """ Private function to get the groups of epochs right before and right after the REM stage.
        
        Args:
            data (pd.DataFrame): DataFrame with the relevant columns: AC and Stage.
            num_epochs (int): Number of epochs to consider before and after the REM stage.

        Returns:
            pd.DataFrame: Input DataFrame including the column 'key' that specify if the epoch is previous to REM, posterior to REM, REM or other. 
        """
        act_data = data.copy()
        act_data = act_data.reset_index()
        # act_data['Stage'] = combine_stages(act_data['Stage'],'ns')
        rem_indexs = np.where(act_data['Stage']==self.stages['REM'])[0]
        rem_slots = self._get_intervals(rem_indexs)

        prev_is = np.array([])
        post_is = np.array([])
        for i in rem_slots: 
            act_data_stage = act_data.iloc[i[0]-n_epochs:i[0]]['Stage']
            if (act_data_stage == 1).all():
                i0 = i[0]-n_epochs if i[0]-n_epochs>0 else 0 
                prev_is = np.append(prev_is,np.arange(i0,i[0]))
            act_data_stage = act_data.iloc[i[1]+1:i[1]+n_epochs+1]['Stage']
            if (act_data_stage == 1).all():
                i1 = i[1]+n_epochs if i[1]+n_epochs<len(act_data) else len(act_data)-1
                post_is = np.append(post_is,np.arange(i[1],i1))
        other_is = [i for i in range(len(data)) if i not in prev_is and i not in post_is and i not in rem_indexs and data['Stage'].iloc[i]!= 5]

        group1 = act_data[['Stage','AC']].iloc[prev_is]
        group1['key'] = 'prev'

        group2 = act_data[['Stage','AC']].iloc[post_is]
        group2['key'] = 'post'

        group3 = act_data[['Stage','AC']].iloc[rem_indexs]
        group3['key'] = 'rem'
        
        group4 = act_data[['Stage','AC']].iloc[other_is]
        group4['key'] = 'other_sleep'
        
        output = pd.concat([group1,group2,group3,group4])
        return output