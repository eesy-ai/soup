import pandas as pd 
import numpy as np 
from itertools import combinations
from .Statistics import Statistics
from .display import Display
from pathlib import Path
import types
import textwrap

class soup():
    def __init__(self, config, stages):
        """ Class Constructor

        Args:
            config (dict): Configuration dictionary with the conditions to check
            stages (dict): Configuration dictionary with the sleep stages
        """
        self.config = config
        self.stages = stages

        self.stats = Statistics()
        self.display = Display()

        self.checking_methods= dict()
        self._check_conditions() 

        self.output_pvalues = self._init_output()
        self.output_statistics = self._init_output()
        self.output_values = self._init_output()

    def run(self,data,clinical=None,output_path=Path('.')):
        """ Run the tool with the data

        Args:
            data (pd.DataFrame): DataFrame with the data to be validated
            clinical (pd.DataFrame, optional): DataFrame with the clinical information to . Defaults to None.
            output_path (_type_, optional): _description_. Defaults to Path('.').
        """
        self.validate(data) 
        
        self.output_statistics.to_csv(output_path / 'statistics.csv',index=False)
        self.output_pvalues.to_csv(output_path / 'pvalues.csv',index=False)
        self.output_values.to_csv(output_path / 'values.csv',index=False)

        self.display.generate_report([self.output_values,self.output_statistics], clinical, output_path)

        
    def validate(self, data):
        """ Validate the data and check the conditions
        
        Args:
            data (pd.DataFrame): DataFrame with the data to be validated
        """     
        self._check_data_columns(data.columns)

        patients_checked = []
        ids = data['id'].unique()

        for patient in ids:
            act_data_patient = data.loc[data['id'] == patient]
            if not self._check_sleep_stages(act_data_patient['Stage']):
                continue
            
            patients_checked.append(patient)
            self._process_patient_data(act_data_patient)
        
        self._format_results(patients_checked)


    def _process_patient_data(self, data_patient):
        """ Process the data of a patient by checking the conditions

        Args:
            data_patient (pd.DataFrame): DataFrame with the patient data to be validated
        """
        for condition in self.config['conditions'].keys():
            self.checking_methods[condition](data_patient[self.config['rel_cols']], self.config['conditions'][condition],condition)

    def _check_conditions(self):
        """ Check if the conditions are implemented in the class and generate the methods if needed.

        Raises:
            ValueError: When conditions are not implemented and the information to generate them is not included in the config.
        """
        class_funcs = [ f for f in dir(self)]
        for condition in self.config['conditions']:
            func_name = '_check_' + condition
            if func_name not in class_funcs:
                if 'generate' not in self.config['conditions'][condition]:
                    raise NotImplementedError(f'Condition {condition} not found in implemented methods. The method should be named _check_{condition}. \nIf condition is not implemented, the information to automatically generate it should be included in config.\nSee documentation. ')

                self.generate_check_function(self.config['conditions'][condition], func_name)
            self.checking_methods[condition] = getattr(self,func_name)


    def _check_data_columns(self,columns):
        """ Check if the columns needed for the conditions are in the data

        Args:
            columns (list): List of columns in the data

        Raises:
            ValueError: When the columns specified in the config file are not in the data
        """
        cols_check = np.array([col in columns for col in self.config['rel_cols']])
        if not cols_check.all():
            raise ValueError(f"Columns {self.config['rel_cols'][~cols_check]} not found in data")

    def _check_sleep_stages(self,stages):
        """ Check if the sleep stages are complete

        Args:
            stages (pd.Series): Sleep stages of a patient during a night.

        Returns:
            boolean: True if the stages are complete, False otherwise.
        """
        values, counts = np.unique(stages,return_counts=True)
        if any(counts < 3):
            return False
        if len(values) == 5:
            return True
        # If just N2 is missing, it is fine
        if len(values) == 4 and self.stages['N2'] not in values:
            return True
        return False

    def _get_intervals(self,nums):
        """ Extract the intervals from a list of numbers: E.g: [1,2,3,5,6,7] -> [(1,3),(5,7)]

        Args:
            nums (list): List of integers to extract the intervals

        Returns:
            list: List of tuples with the extracted intervals
        """
        nums = sorted(set(nums))
        gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s + 1 < e]
        edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
        return list(zip(edges, edges))

    def _init_output(self):
        """ Method to initialize the outputs dictionary

        Returns:
            dict: Dictionary with keys as the conditions and empty lists as values.
        """
        output = dict()
        for key in self.config['conditions']:
            output[key] = []
        return output

    def _create_output_df(self,output,patient_ids):
        """ Method to create a DataFrame from the output dictionary

        Args:
            output (dict): Dictio
            patient_ids (list): List of patients ids

        Returns:
            pd.DataFrame: Formatted output DataFrame
        """
        cols2remove = [ col for col in output if len(output[col]) == 0]
        for col in cols2remove:
            del output[col]
        output = pd.DataFrame(output)
        output['id'] = patient_ids
        return output 
    


    def _format_results(self,patient_ids):
        """ Format the results of the conditions to a DataFrame

        Args:
            patient_ids (list): List of patients ids 
        """
        self.output_pvalues = self._create_output_df(self.output_pvalues,patient_ids)
        self.output_values = self._create_output_df(self.output_values,patient_ids)
        self.output_statistics = self._create_output_df(self.output_statistics,patient_ids)
        self.output_statistics = self.output_statistics & self.output_values[self.output_statistics.columns]


    def _combine_stages(self, stages, type):
        """ Combine stages according to the type. 

        Args:
            stages (pd.Series): Array with the stages to combine
            type (str): Specifies how the sleep stages should be combined. E.G: type = 'light' -> N2 and N1 are combined with N1 ; type = 'ns' -> N2, N1 and N3 are combined with N1 ; type = 'all' -> All stages (except awake) are combined with N1

        Returns:
            pd.Series: combined stages
        """
        output = stages.copy()
        # Combine N1, N2 and N3
        if  type == 'light' or type == 'ns' or type == 'all':
            output.loc[output.values == self.stages['N2']] = self.stages['N1']
        if type == 'ns' or type == 'all':
            output.loc[output.values == self.stages['N3']] = self.stages['N1']
        if type == 'all':
            output.loc[output.values == self.stages['REM']] = self.stages['N1']
        return output


    def _get_combinations(self, stages, main_stage):
        """ Auxiliar method to get the combinations of a set of ints against one. EG: stages=[1,2,3] and main_stage = 1 -> ['1-2','1-3']

        Args:
            stages (np.array): Array with all the stages.
            main_stage (int): Main stage to combine with the rest

        Returns:
            list: List with the combinations
        """
        output = []
        for comb in combinations(stages,2):
            if main_stage not in comb:
                continue
            output.append(str(comb[0]) + '-' + str(comb[1]))
        return output
    

    def generate_check_function(self, config, function_name):
        """ Generate a function to check a condition based on the configuration dictionary

        Args:
            config (dict): Configuration dictionary with the information to generate the function
            function_name (str): Name of the function to generate
        """
        # Generate the function code as a string with correct indentation
        function_code = textwrap.dedent(f"""
        def {function_name}(self, data, config, condition_name):
            act_data = data.copy()
            if len(config['generate']['stages2ignore']) > 0:
                # Remove stages to ignore
                stages2ignore = [self.stages[stage] for stage in config['generate']['stages2ignore']]
                act_data = act_data.loc[~act_data['Stage'].isin(stages2ignore)]

            if config['generate']['groupby'] != '':
                # Group stages 
                act_data.loc[:,'Stage'] = self._combine_stages(act_data['Stage'], config['generate']['groupby'])

            # Statistical Analysis
            test_results = self.stats.pairwaise_comparisson(act_data[[config['generate']['column'], 'Stage']], alpha=config['alpha'])
            if config['generate']['stage2compare'] != '':
                rel_groups = str(np.min([self.stages[config['generate']['stage2compare']], self.stages[config['generate']['stage']]]))
                rel_groups += '-' + str(np.max([self.stages[config['generate']['stage2compare']], self.stages[config['generate']['stage']]]))
                rel_groups = [rel_groups]
            else:
                rel_groups = self._get_combinations(np.arange(0, 6), self.stages[config['generate']['stage']])
            test_results = test_results.loc[test_results['groups'].isin(rel_groups)]

            # Average values analysis
            mean_values = act_data.groupby('Stage')[config['generate']['column']].mean()
            rel_index = mean_values.index.get_loc(self.stages[config['generate']['stage']])
            if config['generate']['operation'] == 'sim':
                diffs = np.abs([mean_values.iloc[rel_index] - mean_values.iloc[i] for i in range(len(mean_values)) if i != rel_index])
                index = mean_values.index.get_loc(self.stages[config['generate']['stage2compare']])
                
                if 'threshold' in config.keys():
                    values_output = np.abs(mean_values.iloc[rel_index] - mean_values.iloc[index]) < config['threshold']
                else:
                    # To fit the index with the diffs array
                    index = index - 1 if index > rel_index else index
                    values_output = np.argmin(diffs) == index
                test_output = test_results['H0_Rejected'].iloc[0] == 'False'
            else:
                if config['generate']['operation'] == 'max':
                    index = np.argmax(mean_values)
                elif config['generate']['operation'] == 'min':
                    index = np.argmin(mean_values)
                values_output = index == rel_index
                test_output = np.all(test_results['H0_Rejected'] == 'True')

            self._save_results(condition_name,
                            test_output,
                            test_results['P-Value'].values,
                            values_output)
        """)

        # Compile the function code
        exec(function_code)

        # Get the function object
        function_object = locals()[function_name]

        # Bind the function to the class instance
        bound_function = types.MethodType(function_object, self)
        setattr(self, function_name, bound_function)

    def _save_results(self, condition_name, statistics, p_values, values):
        """ Save the results of a condition

        Args:
            condition_name (str): Name of the condition
            statistics (boolean): Statistic fulfillment of the condition
            p_values (float): Obtained P-Value of the condition
            values (boolean): On-Average fulfillment of the condition
        """
        self.output_statistics[condition_name].append(statistics)
        self.output_pvalues[condition_name].append(p_values)
        self.output_values[condition_name].append(values)
