
from itertools import combinations
from scipy.stats import shapiro, levene, ttest_ind, kruskal
import numpy as np 
import pandas as pd 

class Statistics():
    def __init__(self):
        """ Class to perform statistical tests.
        """
        pass 

    def pairwaise_comparisson(self,data,group_id_key = 'Stage',alpha=0.01):
        """ Perform a pairwaise comparisson between all the groups in the data.

        Args:
            data (pd.DataFrame): DataFrame with the relevant columns to compare.
            group_id_key (str, optional): Column name to group the data. Defaults to 'Stage'.
            alpha (float, optional): Thredshold to stablish if the null hipotesis is rejected. Defaults to 0.01.

        Returns:
            pd.DataFrame: DataFrame with the results of the comparisson. It has the following columns: 'groups', 'P-Value', 'Statistic', 'H0_Rejected'.
        """
        output = []
        
        col2test = [ c for c in data.columns if c != group_id_key]
        group_ids = np.unique(data[group_id_key])

        groups_normality = []
        for group_id in group_ids:
            act_data = data.loc[data[group_id_key] == group_id][col2test].values 
            if len(np.unique(act_data)) == 1:
                p = 1 # For safety I assume it is not normal
            else:
                _, p = shapiro(act_data)
            groups_normality.append([group_id, p<=alpha])
        groups_normality = pd.DataFrame(np.array(groups_normality),columns=['group','is_normal'])
        
        for group1, group2 in combinations(group_ids, 2):
            act_group_pair = str(group1) + '-' + str(group2)
            data_group1 = data.loc[data[group_id_key] == group1]
            data_group2 = data.loc[data[group_id_key] == group2]
            
            groups_normality_filter = groups_normality.loc[(groups_normality['group'].isin([group1, group2]))]
            act_output = self.same_distribution_test(data_group1[col2test].values.flatten(),
                                                     data_group2[col2test].values.flatten(), 
                                                     (groups_normality_filter['is_normal']==1).all() ,
                                                     alpha=alpha)
            output.append([act_group_pair] + act_output)

        output = pd.DataFrame(np.array(output), columns=['groups', 'P-Value', 'Statistic', 'H0_Rejected'])
        return output

        
    def same_distribution_test(self,group1, group2, is_normal_distribution, alpha=0.01):                
        """ Perform a test to check if two populations follow the same distribution.

        Args:
            group1 ('pd.series'): Group1 to compare
            group2 ('pd.series'): Group2 to compare
            is_normal_distribution (bool): Indicates if both populations follow a normal distribution
            alpha (float, optional): Thredshold to stablish if the null hipotesis is accepted. Defaults to 0.01.

        Returns:
            list: Test results: 'P-Value', 'Statistic', 'H0_Rejected'.
        """

        if self._check_identical_values(group1,group2):
            s, p = -1, 0 
        elif is_normal_distribution:
            _, p_var = levene(group1, group2)
            s, p = ttest_ind(list(group1),
                            list(group2),
                            equal_var=p_var > alpha)
        else:
            s, p = kruskal(group1,group2)
        output = [round(p, 3), round(s, 3), p <= alpha]
        return output
    

    def _check_identical_values(self,group1,group2):
        """ Check if two groups are identical.
        Args:
            group1 (np.array): Group1 to compare
            group2 (np.array): Group2 to compare
        
        Returns:
            bool: True if both groups are identical, False otherwise.
        """
        if len(np.unique(group1)) == 1 and (np.unique(group1) == np.unique(group2)).all():
            return True