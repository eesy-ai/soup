from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import sys
import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import ListFlowable, ListItem
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, Image, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle, ListStyle

class Display:
    def __init__(self):
        """ Class to display the results of the validation.
        """
        # Styles for the report
        # Styles
        styles = getSampleStyleSheet()
        self.title_style = ParagraphStyle(name='TitleStyle', parent=styles['Title'], fontName='Helvetica-Bold', fontSize=24, alignment=1, textColor=colors.navy)
        self.subtitle_style = ParagraphStyle(name='HeadingStyle', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=18, textColor=colors.darkblue)
        self.subsubtitle_style = ParagraphStyle(name='SubHeadingStyle', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=14, textColor=colors.blue)
        self.body_style = ParagraphStyle(name='BodyStyle', parent=styles['BodyText'], fontName='Helvetica', fontSize=12)
        self.bullet_style = ParagraphStyle(name='BulletStyle',parent=styles['BodyText'],fontName='Helvetica',fontSize=12,leftIndent=0,bulletIndent=20,spaceBefore=6,)

    def plot_heatmap(self, data, conditions, title='', output_path=None, filename=''):
        """ Plot a heatmap with the data provided, showing the conditions fulfilled by each patient in green and the unfulfilled in red.

        Args:
            data (pd.DataFrame): Dataframe with the data to plot.
            conditions (list): List of the conditions to show. They must match with the columns of the data.
            title (str, optional): Title of the plot. Defaults to ''.
            output_path (pathlib.Path, optional): Path to save the output summary. If it is not specified, it will not be saved. Defaults to None.
            filename (str, optional): Name of the file to be saved. Defaults to ''.
        """
        _, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(data, cmap='RdYlGn', aspect='auto', interpolation='nearest',vmin=0,vmax=1)
        # Set the ticks and tick labels
        ax.set_xticks(np.arange(len(conditions)))
        ax.set_xticks(np.arange(len(conditions)) - 0.5, minor=True)
        ax.set_xticklabels(conditions, rotation=90,fontsize=17)

        # Set the labels and title
        plt.xlabel('Conditions', fontsize=17)
        plt.ylabel('Patients', fontsize=17)
        plt.title(title, fontsize=17)

        # Adjust font size for the major ticks
        for tick in ax.get_xticklabels():
            tick.set_fontsize(17)
            
        if output_path:
            plt.savefig(output_path / (filename + '_heatmap.png'), bbox_inches='tight')
        plt.show()


    def plot_bar(self, data, conditions, title='', output_path=None, filename=''):
        """ Plot a barplot with the data provided, showing the percentage of fulfilled conditions in each bar.

        Args:
            data (pd.DataFrame): Dataframe with the data to plot.
            conditions (list): List of the conditions to show. They must match with the columns of the data.
            title (str, optional): Title of the plot. Defaults to ''.
            output_path (pathlib.Path, optional): Path to save the output summary. If it is not specified, it will not be saved. Defaults to None.
            filename (str, optional): Name of the file to be saved. Defaults to ''.
        """
        percentage_fulfilled = data.sum(axis=0) / len(data) * 100
        
        plt.figure(figsize=(10, 8))
        plt.bar(conditions, percentage_fulfilled, color='skyblue')
        plt.xlabel('Conditions',fontsize=17)
        plt.ylabel('Percentage Fulfilled',fontsize=17)
        plt.title(title, fontsize=17)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=90,fontsize=17)
        plt.yticks(fontsize=17)
        # y limit to 100%
        plt.ylim(0, 100)
        if output_path:
            plt.savefig(output_path / (filename + '_barplot.png'), bbox_inches='tight')
        plt.show()



    def visualize_output(self, output_, title='', output_path=None, filename=''):
        """ Visualize the output of the conditions

        Args:
            output_ (pd.DataFrame): Dataframe with one of the outputs generated in validate method. 
            title (str, optional): String indicating the title of the displayed graphics. Defaults to ''.
            output_path (pathlib.Path, optional): If specified, the output will be saved in the specified path. Defaults to None.
            filename (str, optional): String indicating the name of the output files. Defaults to ''.
        """
        features = [col for col in output_.columns if col not in ['id', 'Summary'] and not output_[col].isna().all()]
        data = output_[features].values
        data = data.astype(str)
        data[data == 'False'] = 0
        data[data == 'True'] = 1
        data = data.astype(float)

        self.plot_heatmap(data, features, title=title, output_path=output_path, filename=filename)
        self.plot_bar(data, features, title=title,output_path=output_path, filename=filename)


    def write_summary(self, data, clinical=None, output_path=None, filename=''):
        """ Write a summary of the fulfillment of the conditions.

        Args:
            data (pd.DataFrame): Dataframe with the data to summarize.
            clinical (pd.DataFrame, optional): Contains clinical information for disemination. Defaults to pd.DataFrame([]).
            output_path (pathlib.Path, optional): If specified, the output will be saved in the specified path. Defaults to None.
            filename (str, optional): String indicating the name of the output files. Defaults to ''.
        """
        if output_path: orig_stdout,f = self._open_file(output_path , (filename + '_summary.txt'))
            
        n_samples = len(data)
        cols2check = [col for col in data.columns if col != 'id']
        output_text = ''
        for col in cols2check:
            output_text += '* '+col + ": Fulfilled by " + str(np.round(data[col].sum() / n_samples * 1e2,2)) + "% of patients.\n"
            if clinical is not None:
                output_text += self.summarize_by_clinical(data[[col]+['id']],clinical )
        print(output_text)
        if output_path: self._close_file(orig_stdout,f)
        return output_text 

    def _write_grouped_summary(self, data, title=''):
        """ Write a summary of the fulfillment of the conditions grouped by certain disemination variable. E.G: by sleep stage.

        Args:
            data (pd.DataFrame): Dataframe with the data to summarize.
            title (str, optional): String indicating the title of summary. Defaults to ''.
        """
        text = ''
        text += '\t* By ' + title + '\n'
        for group in data.index:
            text += f"\t\t* " + str(group) + ": " + str(data.loc[group][0]) + " % \n"
        print(text)
        return text


    def summarize_by_clinical(self, data, clinical):
        """ Summarize the fulfillment of the conditions grouped by clinical information.

        Args:
            data (pd.DataFrame): Dataframe with the data to summarize.
            clinical (pd.DataFrame, optional): Contains clinical information for disemination.

        Returns:
            dict: Dictionary with the summary of the conditions grouped by clinical information.
        """
        output = dict()
        output_text = ''
        df = data.merge(clinical, on='id')
        clinical_cols = [c for c in clinical.columns if c != 'id']
        data_cols = [c for c in data.columns if c != 'id']
        for col in clinical_cols:
            output[col] = np.round(df.groupby(col)[data_cols].sum() / df.groupby(col)[data_cols].count() * 1e2,2)
            output_text += self._write_grouped_summary(output[col], col)

        return output_text


    def summarize_by_stages_patient(self, data, output_path=None, filename=''):
        """ Summarize the fulfillment of the conditions grouped by sleep stages for a single patient.

        Args:
            data (pd.DataFrame): DataFrame with the data to summarize.
            output_path (pathlib.Path, optional): If specified, the output will be saved in the specified path. Defaults to None.
            filename (str, optional): String indicating the name of the output files. Defaults to ''.

        Returns:
            pd.DataFrame: DataFrame with the summary of the conditions grouped by sleep stages.
        """
        output = dict()
        for stage in ['n1','n2','n3','nrem','rem','awake']:
            related_cols = [col for col in data.columns if '_' + stage in col]
            if len(related_cols) == 0 :
                continue
            output[stage] = [data[related_cols].values.sum() / len(related_cols) * 1e2]
        output = pd.DataFrame(output,index=['Percentage of conditions fulfilled']).T
        output.dropna()
        output['Stage'] = output.index 
        output = output[['Stage','Percentage of conditions fulfilled']]
        output.reset_index(drop=True, inplace=True)
        if output_path: 
            self._create_dir_if_needed(output_path)
            output.to_csv(output_path / (filename + '_stages_summary.csv'), index=False)
        return output

    
    def summarize_by_stages_dataset(self, data, output_path=None, filename=''):
        """ Summarize the fulfillment of the conditions grouped by sleep stages for the whole dataset.

        Args:
            data (pd.DataFrame): DataFrame with the data to summarize.
            output_path (pathlib.Path, optional): If specified, the output will be saved in the specified path. Defaults to None.
            filename (str, optional): String indicating the name of the output files. Defaults to ''.

        Returns:
            pd.DataFrame: DataFrame with the summary of the conditions grouped by sleep stages.
        """
        if output_path: 
            orig_stdout,f = self._open_file(output_path , (filename + '_summary.txt'))
        output = dict()
        output_text = ''
        for stage in ['n1','n2','n3','nrem','rem','awake']:
            related_cols = [col for col in data.columns if '_' + stage in col]
            if len(related_cols) == 0 :
                continue
            output[stage] = np.round(data[related_cols].sum() / data[related_cols].count() * 1e2,2)
            output[stage] = pd.DataFrame(output[stage],columns=[stage])
            output_text+=self._write_grouped_summary(output[stage], stage)
        if output_path: 
            self._close_file(orig_stdout,f)
        return output_text


    def _create_dir_if_needed(self, path):
        """ Create a directory if it does not exist.

        Args:
            path (pathlib.Path): Path to create.
        """
        if not os.path.isdir(path):
            os.makedirs(path)
        
    def _open_file(self,path, filename):
        """ Open a file to write the output.

        Args:
            path (pathlib.Path): Path to save the file.
            filename (str): Name of the file.

        Returns:
            tuple: Original stdout and file object.
        """
        self._create_dir_if_needed(path)
        orig_stdout = sys.stdout
        f = open(path/filename , 'w')
        sys.stdout = f
        return orig_stdout,f

    def _close_file(self,orig_stdout,f):
        """ Close the file and restore the original stdout.

        Args:
            orig_stdout (io.TextIOWrapper): Original stdout 
            f (io.TextIOWrapper): File object
        """
        sys.stdout = orig_stdout
        f.close()
# 
    def _visualize_clusters(self,clusters, Z, output_path=None, filename=''):
        """ Visualize the clusters obtained from the hierarchical clustering. Source: https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/

        Args:
            clusters (np.ndarray): Array with the cluster of each sample.
            Z (np.ndarray): Array with the linkage matrix.
            output_path (pathlib.Path, optional): If specified, the output will be saved in the specified path. Defaults to None.
            filename (str, optional): String indicating the name of the output files. Defaults to ''.
        """
        # Count the number of samples in each cluster
        cluster_counts = np.bincount(clusters)[1:]
        # Plot dendrogram
        plt.figure(figsize=(10, 5))
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Cluster')
        plt.ylabel('Distance')
        dendrogram(Z, 
                   truncate_mode='lastp',
                   p = len(cluster_counts),
                   show_contracted=True,  # to get a distribution impression in truncated branches
                   leaf_rotation=90., 
                   leaf_font_size=12.)
        plt.tight_layout()  # Adjust layout to prevent overlap
        if output_path:
            plt.savefig(output_path / (filename + '_dendrogram.png'))
        plt.show()

    def cluster_patients(self, data_, ids, max_distance=.1, output_path=None, filename=''):
        """ Cluster the patients based on the data provided.

        Args:
            data_ (pd.DataFrame): Dataframe with the data to cluster.
            ids (list): List of the ids of the patients.
            max_distance (float, optional): Maximum distance to consider two samples in the same cluster. Defaults to .1, so in the same cluster all the patients fulfill the same conditions.
            output_path (pathlib.Path, optional): If specified, the output will be saved in the specified path. Defaults to None.
            filename (str, optional): String indicating the name of the output files. Defaults to ''.
        """
        data = data_.drop(columns=['id'])
        data = data.astype(int)

        # Perform agglomerative clustering
        Z = linkage(data, method='ward') 
        clusters = fcluster(Z, max_distance, criterion='distance')
        
        # Visualize clusters
        if len(np.unique(clusters)) > 1:
            self._visualize_clusters(clusters,Z, output_path=output_path, filename=filename)

        # Print samples in each cluster
        if output_path:
            orig_stdout,f = self._open_file(output_path, (filename + '_clusters.txt'))

        unique_clusters = np.unique(clusters)
        for cluster in unique_clusters:
            samples_in_cluster = np.where(clusters == cluster)[0]
            samples_in_cluster = ids[samples_in_cluster]
            print(f"\n* Cluster {cluster}. Samples: \n\t{samples_in_cluster}")

        if output_path: 
            self._close_file(orig_stdout,f)

    def write_score_patient(self, output_values, output_statistics, output_path=None, filename=''):
        """ Write the score of the patient.

        Args:
            output_values (pd.DataFrame): Dataframe with the output of the on-average fulfillment of the conditions.
            output_statistics (pd.DataFrame): Dataframe with the output of the statistically relevant fulfillment of the conditions.
            output_path (pathlib.Path, optional): If specified, the output will be saved in the specified path. Defaults to None.
            filename (str, optional): String indicating the name of the output files. Defaults to ''.
        """
        output = pd.concat([output_values,output_statistics]).T
        output.columns=['True in Average', 'Statistically Significant']
        score_row = pd.DataFrame(['SCORE'] + np.round( output.sum() / len(output), 2).to_list()).T
        output['Condition'] = output.index
        output = output[['Condition','True in Average', 'Statistically Significant']]
        score_row.columns = output.columns
        output.reset_index(drop=True, inplace=True)
        output = pd.concat([output, score_row],axis=0)

        if output_path:
            self._save_csv(output,output_path,filename)
        return output

    def write_score_dataset(self, output_values, output_statistics, ids, output_path=None, filename=''):
        """ Write the score of the dataset.

        Args:
            output_values (pd.DataFrame): Dataframe with the output of the on-average fulfillment of the conditions.
            output_statistics (pd.DataFrame): Dataframe with the output of the statistically relevant fulfillment of the conditions.
            ids (list): List of the ids of the patients.
            output_path (pathlib.Path, optional): If specified, the output will be saved in the specified path. Defaults to None.
            filename (str, optional): String indicating the name of the output files. Defaults to ''.
        """
        output_values = np.round(output_values.sum(axis=1)/output_values.shape[1] * 1e2,2)
        output_statistics = np.round(output_statistics.sum(axis=1)/output_statistics.shape[1] * 1e2,2)
        output = pd.concat([output_values,output_statistics],axis=1)
        output.columns = ['True in Average', 'Statistically Significant']
        output = pd.concat([output,
                            pd.DataFrame(np.round(output.sum(axis=0) / output.shape[0],2)).T])
        output.index = np.append(ids, np.array(['SCORE']))

        if output_path:
            self._save_csv(output,output_path,filename)
        return output

    def _save_csv(self,output, output_path, filename):
        """ Save output to a csv file.

        Args:
            output (pd.DataFrame): Dataframe to save.
            output_path (pathlib.Path, optional): If specified, the output will be saved in the specified path. Defaults to None.
            filename (str, optional): String indicating the name of the output files. Defaults to ''.
        """
        self._create_dir_if_needed(output_path)
        output.to_csv(output_path / (filename + '.csv'))


    
    def generate_report(self, results, clinical, output_path):
        """ Generate PDF report with the results of the validation.

        Args:
            results (list): List of pandas dataframes with the results. The first element must contain the output values and the second the output statistics.
            clinical (pd.DataFrame): Contains clinical information for disemination.
            output_path (pathlib.Path): Path to save the report.
        """
        conditions = [col for col in results[0] if col not in ['id','Summary']]
        is_single_patient = results[0].shape[0] == 1
        ids = results[0]['id'].values
        doc = SimpleDocTemplate(str(output_path/'report.pdf'), pagesize=letter,author='Eesy-Innovation GmbH', title='soup - Report')


        # Create a list to store the report content
        report = []
        
        # Title section
        title = "SOUP - Report"
        if is_single_patient:
            title += ' for patient ' + str(ids[0])
        else:
            title += ' for ' + str(len(ids)) + ' patients'
        report = self._add_text_to_pdf(report, title, self.title_style)
        
        
        # Section 1: Global Scores
        report.append(Paragraph("Introduction", self.subtitle_style))

        if is_single_patient : # Single patient 
            scores = self.write_score_patient(results[0][conditions].astype(int), 
                                              results[1][conditions].astype(int),
                                              output_path, 
                                              'scores')
        else:
            scores = self.write_score_dataset(results[0][conditions].astype(int), 
                                              results[1][conditions].astype(int),
                                              ids, 
                                              output_path, 
                                              'scores')
            scores = scores.iloc[[-1]]
        self._add_table_to_pdf(report, scores)
        
        # Section 2: Statistical Analysis disemination. 
        for i in range(2):
            filename = 'values' if i == 0 else 'statistics'
            title = "Absolute Values " if i == 0 else "Statistical " 
            report.append(Paragraph(title + "Analysis disemination. ", self.subtitle_style))

            if not is_single_patient:
                # Conditions fulfillment visualization
                self.visualize_output(results[i],filename.upper(), output_path, filename)
                report = self._add_image_to_pdf(report, output_path / (filename + '_heatmap.png'))
                report = self._add_image_to_pdf(report, output_path / (filename + '_barplot.png'))


            # Sleep Stages disemination
            report.append(Paragraph('Sleep Stages disemination', self.subsubtitle_style))
            if not is_single_patient:
                stages_summary = self.summarize_by_stages_dataset(results[i], output_path, filename+'_stages_summary')
                report = self._add_text_to_pdf(report, stages_summary, self.body_style)
            else: 
                stages_summary = self.summarize_by_stages_patient(results[i], output_path, filename+'_stages_summary')
                report = self._add_table_to_pdf(report, stages_summary)


            # Cluster patients
            if not is_single_patient:
                # clinical disemination
                clinical_summary = self.write_summary(results[i],clinical, output_path, filename=filename + '_clinical_summary')
                report.append(Paragraph('Clinical disemination', self.subsubtitle_style))
                report = self._add_text_to_pdf(report, clinical_summary, self.body_style)

                # Patients clustering 
                self.cluster_patients(results[i],ids=ids,output_path=output_path, filename=filename+'_clustering')
                report.append(Paragraph('Clustering', self.subsubtitle_style))
                # if image exists
                if os.path.exists(output_path / (filename + '_clustering_dendrogram.png')):
                    report = self._add_image_to_pdf(report, output_path / (filename + '_clustering_dendrogram.png'), height=200)
                else: 
                    report = self._add_text_to_pdf(report, 'All patients are in the same cluster.', self.body_style)

        # Section 
        # Build the report
        doc.build( report)
    
    def _add_table_to_pdf(self,report, data):
        """ Private method to add a table to the report.

        Args:
            report (list): List with the content of the report.
            data (pd.DataFrame): Table to add to the report.

        Returns:
            list: Content of the report with the table added.
        """
        data = [data.columns.values.tolist()] + data.values.tolist()
        table = Table(data, style=[
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (1, 1), (-1, -1), 'RIGHT')
        ])
        report.append(table)
        return report

    def _add_image_to_pdf(self,report, image_path, width=400, height=400):
        """ Private method to add an image to the report.

        Args:
            report (list): List with the content of the report.
            image_path (pathlib.Path): Path to the image to add.
            width (int, optional): Width to set to the image in the report. Defaults to 400.
            height (int, optional): Height to set to the image in the report. Defaults to 400.

        Returns:
            list: Content of the report with the image added.
        """
        chart_image = Image(image_path, width=width, height=height)
        # report.append(Paragraph('Heatmap', subsubtitle_style))
        report.append(chart_image)
        report.append(Spacer(1, 12))
        return report
    
    def _add_text_to_pdf(self,report, text, style):
        """ Private method to add text to the report

        Args:
            report (list): List with the content of the report.
            text (str): Text to add to the report.
            style (reportlab.lib.styles.ParagraphStyle): Style to apply to the text.

        Returns:
            list: Content of the report with the text added.
        """
        if '\t' not in text: 
            text = text.replace('\n','<br/>')
            report.append(Paragraph(text, style))
        else:  
            def create_bullet_list(items):
                list_items = []
                for item in items:
                    level = item.count('\t')
                    text = item.lstrip('\t').strip('*').strip()
                    list_items.append(
                        ListItem(
                            Paragraph(text, self.bullet_style),
                            leftIndent=(level+2) * 10  # Adjust the indentation as needed
                        )
                    )
                return ListFlowable(list_items, bulletType='bullet', start='circle',bulletFontSize=6)
            import re
            sections = re.split(r'\n', text)
            bullet_list = create_bullet_list(sections)
            report.append(bullet_list )
        
        return report
