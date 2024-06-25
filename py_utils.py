import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from typing import List
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import py_params
"""
The mortality data used here needs to have the following format:
* Ages (or age group) must be in a wide format.
* Columns can only contain the following:
    1. `year` for the year when the death is recorded in `int`.
    2. `country` for multi-population setting in `int` using the code given by WHO.
    3. `cause` for the underlying cause of death in `int`. The code for this part is intended to be obtained from grouping the various CoD into a smaller category, coded using an integer [1,...]. The code `0` is specifically used for 'all-cause' mortality.
    4. `sex` for the gender in `int`, where `1` is male, `2` is female, and `3` is total.
    5. The rest of the columns must indicate ages or age-groups, sorted from youngest to oldest. All the columns will be used in the modeling, thus filtered must be done in the data preprocessing steps. These columns will contain the **mortality rate, not the death counts**.
"""
class Scaler:
    """
    A class intended to be a collection of scalers fitted to the training set of mortality data.
    The class contains all scalers used in this project, such as:
    1. MinMaxScaler fitted to the mortality data according to sex can cause (based on Adachi Mamiya's Master Thesis)    
    """   
    
    def __init__(self, df: pd.DataFrame,
                 max_year_train: int,
                 mortality_features: List[str]):
        """
        Args:
            df (pd.DataFrame): the mortality data in the format as explained in the beginning of this file.
            max_year_train (int): an integer indicating the maximum year used as training set (assuming the minimum year of the data as the starting point of the training set)
            mortality_features: a list of column names for the log mortality rate, each column specifying the mortality rate of a particular age or age-group
        """
        self.df = df
        self.dict_mamiya_scaler = {}
        for sex in df.sex.unique():
            for cause in df['cause'].unique():
                key = f"{sex}-{cause}"
                scaler = MinMaxScaler()
                self.dict_mamiya_scaler[key] = scaler.fit(df.loc[(df.sex == sex) & 
                                                                 (df.cause == cause) & 
                                                                 (df.year <= max_year_train), 
                                                                 mortality_features].to_numpy().reshape(-1,1))


    def mamiya_transform(self, sex: int, cause:int, data):
        return self.dict_mamiya_scaler[f"{sex}-{cause}"].transform(data)
    def mamiya_inverse_transform(self, sex: int, cause:int, data):
        return self.dict_mamiya_scaler[f"{sex}-{cause}"].inverse_transform(data)
    
    @staticmethod
    def is_mamiya(transform:callable):
        if(transform == None):
            return False
        
        return ((transform.__name__ == Scaler.mamiya_transform.__name__) or 
                (transform.__name__ == Scaler.mamiya_inverse_transform.__name__))

def plot_year_value(df_final_long: pd.DataFrame,
                    value:str,
                    row_feature_name:str, row_feature_values:List[int], row_labels:dict,
                    col_feature_name:str, col_feature_values:List[int], col_labels:dict,
                    hue_feature_name:str, hue_feature_values:List[int], 
                    types:List[str], 
                    years: List[int],
                    year_separators:List[int],
                    title_fig: str,
                    col_palette = None,
                    is_true_dotted = False,
                    is_fig_saved = False,
                    y_limit:list = None):
    """
    A function used to make a plot between the mortality rate (y-axis) of all years (x-axis).
    For the moment, the given DataFrame must be in long format with only 6 columns used as follows:
    - 1 column is used to determine the number of subplots in the vertical axis. If only 1 plot is desired, specify the name of the column and the specific value of that column as a one-element list.
    - 1 column is used to determine the number of subplots in the horizontal axis. If only 1 plot is desired, specify the name of the column and the specific value of that column as a one-element list.
    - 1 column is used to determine the number of lines made in each subplot, each with different colour. If only 1 line is desired, specify the name of the column and the specific value of that column as a one-element list.
    - 1 column named `type` that explains how the `log_mortality` is obtained, through training, validation, testing, or real value. If more than one type are given, each will be plotted with different type of linestroke.
    - 1 column named `year` that contains the year of the `log_mortality`
    - 1 column specified by the parameter `value` that contains the value to be plotted.

    Args:
        df_final_long (pd.DataFrame): a DataFrame containing the mortality data in a long format. The DataFrame must have columns 'year' and 'type' that indicate the year and the type (predicted, true, train, etc.) of the log_mortality. See the description of the function to see applicable constraints on the DataFrame.
        value (str): the name of the column to be used as the y-axis.
        row_feature_name (str): the name of the column to be used to make subplots on the vertical axis
        row_feature_values (List[int]): the values to be considered when making subplots (there will be one row for each value listed)
        row_labels (dict): 
        col_feature_name (str): the name of the column to be used to make subplots on the horizontal axis
        col_feature_values (List[int]): the values to be considered when making subplots (there will be one column for each value listed)
        col_labels (dict): 
        hue_feature_name (str): the name of the column to be plotted with different color in each subplot
        hue_feature_values (List[int]): the values to be considered when plotting multiple line with differnt colour (there will be one line with different colour for each value listed)
        types (List[int]): 
        years (List[int]): all the years to be plotted        
        year_separators (List[int]): a list of years where a straight vertical line will be drawn to separate the years into several intervals
        title_fig (str): the title of the figure
        col_palette:
        is_true_dotted (bool):
        is_fig_saved (bool):
        y_limit (List[int]): a list of two values indicating the minimum and maximum value of the y-axis respectively.
    """
    # Set the base font size of texts in the figure. 
    plt.rcParams['font.size'] = '21'

    # Specify the color pattern 
    if(col_palette is None):
        col_palette = sns.color_palette("viridis", len(hue_feature_values))   

    # Filter the given df according to the given years and types
    df_final_long = df_final_long.loc[(df_final_long[hue_feature_name].isin(hue_feature_values)) & 
                                      (df_final_long[row_feature_name].isin(row_feature_values)) & 
                                      (df_final_long[col_feature_name].isin(col_feature_values)) & 
                                      (df_final_long['type'].isin(types)) & 
                                      (df_final_long['year'].isin(years))]

    # Get all prediction types
    predicted_types = [predicted_type for predicted_type in list(df_final_long['type'].unique()) if predicted_type != 'true']

    # Create subplots
    fig, axs2d = plt.subplots(nrows=len(row_feature_values), ncols=len(col_feature_values), figsize=(18, 20), sharex=True, sharey=True)    
    if(len(row_feature_values) > 1 or len(col_feature_values) > 1):
        axs = axs2d.flatten()

    # for row_idx, sex in enumerate(sexes):
    #     for col_idx, cause in enumerate(causes):             
    for row_idx, row_val in enumerate(row_feature_values):
        for col_idx, col_val in enumerate(col_feature_values):             
             # Determine the axis to plot
            if(len(row_feature_values) > 1 or len(col_feature_values) > 1):
                curr_ax = axs[row_idx * len(col_feature_values) + col_idx]
            else:
                curr_ax = axs

            # Plot different series based on its colour (all age groups)
            # for age_idx, age in enumerate(ages):
            for hue_idx, hue_val in enumerate(hue_feature_values):
                # Plot the true value
                df_curr = df_final_long.loc[(df_final_long[row_feature_name] == row_val) & 
                                            (df_final_long[col_feature_name] == col_val) & 
                                            (df_final_long[hue_feature_name] == hue_val) & 
                                            (df_final_long[py_params.COL_TYPE] == py_params.TYPE_TRUE)]            
                if(is_true_dotted):
                    curr_ax.scatter(df_curr['year'], df_curr[value], label=hue_val, color=col_palette[hue_idx], s=0.5)
                else:
                    curr_ax.plot(df_curr['year'], df_curr[value], label=hue_val, color=col_palette[hue_idx], 
                                            linewidth=1, linestyle='solid')
                
                # Plot the predicted value            
                for type_idx, prediction_type in enumerate(predicted_types):
                    df_curr = df_final_long.loc[(df_final_long[row_feature_name] == row_val) & 
                                                (df_final_long[col_feature_name] == col_val) & 
                                                (df_final_long[hue_feature_name] == hue_val) & 
                                                (df_final_long['type'] == prediction_type)]
                    if(is_true_dotted):
                        curr_ax.plot(df_curr['year'], df_curr[value], color=col_palette[hue_idx], 
                                 linewidth=1, linestyle="-")                           
                    else:
                        curr_ax.plot(df_curr['year'], df_curr[value], color=col_palette[hue_idx], 
                                 linewidth=1, linestyle=py_params.LINESTYLES[type_idx])   
                
                    
                                    
            # Specify the y-axis limits
            if y_limit != None:
                curr_ax.set_ylim(y_limit[0], y_limit[1])  # Set y-axis limits

            # Vertical line separating training and test/forecast periods    
            if(year_separators != None):
                for year_separator in year_separators:
                    curr_ax.axvline(
                        year_separator,
                        color='black', linestyle='dashed', linewidth=0.7
                    )
            
            # Set the title of each subplots
            curr_ax.set_title(f'{row_feature_name}: {row_labels[row_val]} \n{col_feature_name}: {col_labels[col_val]}', fontsize=12)

            # Setting the x-axis labels
            curr_ax.tick_params(axis='x', rotation=45)
            curr_ax.xaxis.set_major_locator(plt.MaxNLocator(6))                              

    # Get the legend from the axes (for different hue colors)    
    if(len(row_feature_values) > 1 or len(col_feature_values) > 1):
        legend_ax = axs[len(col_feature_values)-1]        
    else:
        legend_ax = axs
    handles, labels = legend_ax.get_legend_handles_labels()    
    
    # Set the colorbar if hue is based on age
    if(len(hue_feature_values) > 10):
        num_hue_values = hue_feature_values
        if(hue_feature_name == "age"):
            num_hue_values = list()
            for age in hue_feature_values:
                num = [each for each in age if str(each).isdigit()]
                num = int(''.join(num))
                num_hue_values.append(num)

        norm = plt.Normalize(min(num_hue_values), max(num_hue_values))
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])
        if(len(row_feature_values) > 1 and len(col_feature_values) > 1):
            cbar = legend_ax.figure.colorbar(sm, ax=axs2d[:,:], shrink=0.6, fraction=0.05)
        else:
            cbar = legend_ax.figure.colorbar(sm, ax=legend_ax, shrink=0.6)
        cbar.set_label(hue_feature_name, rotation=90)
        handles = []
        labels = []        

    # Add legends for each linestyle (type)
    legend_linestyle = []
    if(is_true_dotted):
        legend_linestyle.append(mlines.Line2D([], [], linestyle='dotted', label=py_params.TYPE_TRUE))
        for type_idx, prediction_type in enumerate(predicted_types):    
            legend_linestyle.append(mlines.Line2D([], [], linestyle="-", label=prediction_type))        
    else:
        legend_linestyle.append(mlines.Line2D([], [], linestyle='-', label=py_params.TYPE_TRUE))
        for type_idx, prediction_type in enumerate(predicted_types):    
            legend_linestyle.append(mlines.Line2D([], [], linestyle=py_params.LINESTYLES[type_idx], label=prediction_type))        
    label_linestyle = [handle.get_label() for handle in legend_linestyle]

   # Combine automatic legend handles and custom legend handles
    all_handles = handles + legend_linestyle
    all_labels = labels + label_linestyle
    
    # Add legend with both automatic and custom legend entries
    ncol_legend = len(hue_feature_values)/2 if len(hue_feature_values) > 1 else 1

    # Set the legend
    fig.legend(handles=all_handles, labels=all_labels, loc='upper right',
            fancybox=True, shadow=True)
    fig.suptitle(title_fig, fontsize=24)
    fig.subplots_adjust(top=0.9, right=0.8)
    if(is_fig_saved):
        plt.savefig(f"{title_fig}.png")
    plt.show()   

def plot_age_mortality_year(df_long: pd.DataFrame, 
                            row_feature_name:str, row_feature_values:List[int], row_labels:dict,
                            col_feature_name:str, col_feature_values:List[int], col_labels:dict,
                            hue_feature_name:str, hue_feature_values:List[int],
                            ages:List[str],
                            types:List[str], 
                            title_fig:str, 
                            linestyles:dict = None,
                            is_fig_saved = False,
                            y_limit:list = None):
    """
    A function used to make a plot between the mortality rate (y-axis) of all ages or age groups (x-axis).
    This function can be used to plot various mortality-rate vs age groups 
    in various colours representing different categorical feature (such as years) 
    by using the `hue_feature_name` and `hue_feature_values` parameters.
    
    For the moment, the given DataFrame must be in long format with only 6 columns used as follows:
    - 1 column is used to determine the number of subplots in the vertical axis. If only 1 plot is desired, specify the name of the column and the specific value of that column as a one-element list.
    - 1 column is used to determine the number of subplots in the horizontal axis. If only 1 plot is desired, specify the name of the column and the specific value of that column as a one-element list.
    - 1 column is used to determine the number of lines made in each subplot, each with different colour. If only 1 line is desired, specify the name of the column and the specific value of that column as a one-element list.
    - 1 column named `type` that explains how the `log_mortality` is obtained, through training, validation, testing, or real value. If more than one type are given, each will be plotted with different type of linestroke.
    - 1 column named `age` that contains the age of the population 
    - 1 column named `log_mortality` that contains the log mortality rates

    Args:
        df_final_long (pd.DataFrame): a DataFrame containing the mortality data in a long format. The DataFrame must have columns 'year' and 'type' that indicate the year and the type (predicted, true, train, etc.) of the log_mortality. See the description of the function to see applicable constraints on the DataFrame.
        row_feature_name (str): the name of the column to be used to make subplots on the vertical axis
        row_feature_values (List[int]): the values to be considered when making subplots (there will be one row for each value listed)
        row_labels (dict): the labels for each feature values used to make subplots on the vertical axis
        col_feature_name (str): the name of the column to be used to make subplots on the horizontal axis
        col_feature_values (List[int]): the values to be considered when making subplots (there will be one column for each value listed)
        col_labels (dict): the labels for each feature values used to make subplots on the horizontal axis
        hue_feature_name (str): the name of the column to be plotted with different color in each subplot
        hue_feature_values (List[int]): the values to be considered when plotting multiple line with differnt colour (there will be one line with different colour for each value listed)
        ages (List[int]): all the ages to be plotted
        types (List[str]): all the prediction types (train, test, etc.) to be plotted 
        title_fig (str): the title of the figure
        linestyles (dict): a dictionary of dashes (in seaborn format) specifying the dash style for each value in the types parameter
        is_fig_saved (bool): a boolean to indicate whether the resulting plot is saved or not
        y_limit (List[int]): a list of two values indicating the minimum and maximum value of the y-axis respectively.        
    """   
    # Set the base font size of texts in the figure. 
    plt.rcParams['font.size'] = '21'

    # Specify the color pattern 
    col_palette = sns.color_palette("viridis", len(hue_feature_values))   
    
    # Filter the given df according to the given years and types
    df_long = df_long.loc[(df_long[hue_feature_name].isin(hue_feature_values)) & 
                          (df_long[row_feature_name].isin(row_feature_values)) & 
                          (df_long[col_feature_name].isin(col_feature_values)) & 
                          (df_long['type'].isin(types)) & 
                          (df_long['age'].isin(ages))]
    df_long = df_long.loc[(df_long['type'].isin(types)) & (df_long[hue_feature_name].isin(hue_feature_values))]

    # Get all prediction types    
    predicted_types = [predicted_type for predicted_type in list(df_long['type'].unique()) if predicted_type != py_params.TYPE_TRUE]
    # Set the line style for each predicted type
    if(linestyles is None):
        linestyles = {}
        for idx_style, predicted_type in enumerate(predicted_types):
            linestyles.update({predicted_type: py_params.LINESTYLES[idx_style][1]})
        linestyles[py_params.TYPE_TRUE] = ''

    # Create subplots
    fig, axs2d = plt.subplots(len(row_feature_values), len(col_feature_values), figsize=(18, 18), sharex=True, sharey=True)
    if(len(row_feature_values) > 1 or len(col_feature_values) > 1):
        axs = axs2d.flatten()
    
    for row_idx, row_val in enumerate(row_feature_values):
        for col_idx, col_val in enumerate(col_feature_values):
            # Determine the axis to plot
            if(len(row_feature_values) > 1 or len(col_feature_values) > 1):
                curr_ax = axs[row_idx * len(col_feature_values) + col_idx]
            else:
                curr_ax = axs

            # Plot
            sns.lineplot(data=df_long.loc[(df_long[row_feature_name] == row_val) & 
                                          (df_long[col_feature_name] == col_val)], 
                         x='age', y='log_mortality', hue=hue_feature_name, style='type', 
                         dashes=linestyles,
                         ax=curr_ax, palette=col_palette)
            
            # Specify the y-axis limits
            if y_limit != None:
                curr_ax.set_ylim(y_limit[0], y_limit[1])  # Set y-axis limits

            # Set the title of each subplots
            curr_ax.set_title(f'{row_feature_name}: {row_val}| {col_feature_name}: {col_val}', fontsize=12)

            # Remove the legend to create a separate collective legend for all plots
            curr_ax.get_legend().remove()            

            # Set the title of each subplots
            curr_ax.set_title(f'{row_feature_name}: {row_labels[row_val]} \n{col_feature_name}: {col_labels[col_val]}', fontsize=12)

            # Setting the x-axis labels
            curr_ax.tick_params(axis='x', rotation=45)
            curr_ax.xaxis.set_major_locator(plt.MaxNLocator(6)) 
            
    # Create a single legend for all the subplots
    # Get the legend from the axes (for different hue colors)    
    if(len(row_feature_values) > 1 or len(col_feature_values) > 1):
        legend_ax = axs[len(col_feature_values)-1]        
    else:
        legend_ax = axs
    all_handles, all_labels = legend_ax.get_legend_handles_labels()    
    
    # Set a colorbar if there are too many hue values, and make the legend for linestyle manually
    if(len(hue_feature_values) > 10):
        num_hue_values = hue_feature_values
        if(hue_feature_name == "age"):
            num_hue_values = list()
            for age in hue_feature_values:
                num = [each for each in age if str(each).isdigit()]
                num = int(''.join(num))
                num_hue_values.append(num)
        norm = plt.Normalize(min(num_hue_values), max(num_hue_values))
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])
        if(len(row_feature_values) > 1 and len(col_feature_values) > 1):
            cbar = legend_ax.figure.colorbar(sm, ax=axs2d[:,:], shrink=0.6, fraction=0.1)
        else:
            cbar = legend_ax.figure.colorbar(sm, ax=legend_ax, shrink=0.6, fraction=0.1)
        cbar.set_label(hue_feature_name, rotation=90)
        handles = []
        labels = []        

        # Add legends for each linestyle (type)
        legend_linestyle = []
        if(py_params.TYPE_TRUE in types):
            legend_linestyle.append(mlines.Line2D([], [], linestyle='-', label=py_params.TYPE_TRUE))    
        for prediction_type in predicted_types:    
            legend_linestyle.append(mlines.Line2D([], [], linestyle=(0, linestyles[prediction_type]), label=prediction_type))
        label_linestyle = [handle.get_label() for handle in legend_linestyle]
        
        # Combine automatic legend handles and custom legend handles
        all_handles = handles + legend_linestyle
        all_labels = labels + label_linestyle
        
    # Add legend with both automatic and custom legend entries
    # ncol_legend = len(hue_feature_values)/2 if len(hue_feature_values) > 1 else 1

    # Set the legend
    # print(legend_linestyle)
    fig.legend(handles=all_handles, labels=all_labels, loc='upper right',
            fancybox=True, shadow=True)
    fig.suptitle(title_fig, fontsize=24)
    fig.subplots_adjust(top=0.9, right=0.8)
    if(is_fig_saved):
        plt.savefig(f"{title_fig}.png")
    plt.show()   

def plot_age_mortality_model(df_long: pd.DataFrame, 
                             row_feature_name:str, row_feature_values:List[int], row_labels:dict,
                             col_feature_name:str, col_feature_values:List[int], col_labels:dict,
                             years:List[int],                       
                             ages:List[str],
                             types:List[str], 
                             title_fig:str, 
                             linestyles:dict = None,
                             is_fig_saved = False,
                             y_limit:list = None):
    """
    A function used to make a plot between the mortality rate (y-axis) of all ages or age groups (x-axis).
    This function can be used to plot various mortality-rate vs age groups 
    in various colours representing different categorical feature (such as years) 
    by using the `hue_feature_name` and `hue_feature_values` parameters.
    
    For the moment, the given DataFrame must be in long format with only 6 columns used as follows:
    - 1 column is used to determine the number of subplots in the vertical axis. If only 1 plot is desired, specify the name of the column and the specific value of that column as a one-element list.
    - 1 column is used to determine the number of subplots in the horizontal axis. If only 1 plot is desired, specify the name of the column and the specific value of that column as a one-element list.
    - 1 column is used to determine the number of lines made in each subplot, each with different colour. If only 1 line is desired, specify the name of the column and the specific value of that column as a one-element list.
    - 1 column named `type` that explains how the `log_mortality` is obtained, through training, validation, testing, or real value. If more than one type are given, each will be plotted with different type of linestroke.
    - 1 column named `age` that contains the age of the population 
    - 1 column named `log_mortality` that contains the log mortality rates

    Args:
        df_final_long (pd.DataFrame): a DataFrame containing the mortality data in a long format. The DataFrame must have columns 'year' and 'type' that indicate the year and the type (predicted, true, train, etc.) of the log_mortality. See the description of the function to see applicable constraints on the DataFrame.
        row_feature_name (str): the name of the column to be used to make subplots on the vertical axis
        row_feature_values (List[int]): the values to be considered when making subplots (there will be one row for each value listed)
        row_labels (dict): the labels for each feature values used to make subplots on the vertical axis
        col_feature_name (str): the name of the column to be used to make subplots on the horizontal axis
        col_feature_values (List[int]): the values to be considered when making subplots (there will be one column for each value listed)
        col_labels (dict): the labels for each feature values used to make subplots on the horizontal axis
        years (Lits[int]): all the years to be plotted
        ages (List[int]): all the ages to be plotted
        types (List[str]): all the prediction types (train, test, etc.) to be plotted 
        title_fig (str): the title of the figure
        linestyles (dict): a dictionary of dashes (in seaborn format) specifying the dash style for each value in the types parameter
        is_fig_saved (bool): a boolean to indicate whether the resulting plot is saved or not
        y_limit (List[int]): a list of two values indicating the minimum and maximum value of the y-axis respectively.        
    """   
    # Set the base font size of texts in the figure. 
    plt.rcParams['font.size'] = '21'

    # Specify the color pattern 
    col_palette = sns.color_palette("nipy_spectral", len(types))   
    
    # Filter the given df according to the given years and types
    df_long = df_long.loc[(df_long[row_feature_name].isin(row_feature_values)) & 
                          (df_long[col_feature_name].isin(col_feature_values)) & 
                          (df_long['type'].isin(types)) & 
                          (df_long['year'].isin(years)) &
                          (df_long['age'].isin(ages))]
    # df_long = df_long.loc[(df_long['type'].isin(types)) & (df_long[hue_feature_name].isin(hue_feature_values))]

    # Get all prediction types    
    predicted_types = [predicted_type for predicted_type in list(df_long['type'].unique()) if predicted_type != py_params.TYPE_TRUE]
    # Set the line style for each predicted type
    if(linestyles is None):
        linestyles = {}
        for idx_style, predicted_type in enumerate(predicted_types):
            linestyles.update({predicted_type: py_params.LINESTYLES[idx_style][1]})
        linestyles[py_params.TYPE_TRUE] = ''

    # Create subplots
    fig, axs2d = plt.subplots(len(row_feature_values), len(col_feature_values), figsize=(18, 18), sharex=True, sharey=True)
    if(len(row_feature_values) > 1 or len(col_feature_values) > 1):
        axs = axs2d.flatten()
    
    for row_idx, row_val in enumerate(row_feature_values):
        for col_idx, col_val in enumerate(col_feature_values):
            # Determine the axis to plot
            if(len(row_feature_values) > 1 or len(col_feature_values) > 1):
                curr_ax = axs[row_idx * len(col_feature_values) + col_idx]
            else:
                curr_ax = axs

            # Plot
            sns.lineplot(data=df_long.loc[(df_long[row_feature_name] == row_val) & 
                                          (df_long[col_feature_name] == col_val)], 
                         x='age', y='log_mortality', hue='type', style='type', 
                         dashes=linestyles,
                         ax=curr_ax, palette=col_palette)
            
            # Specify the y-axis limits
            if y_limit != None:
                curr_ax.set_ylim(y_limit[0], y_limit[1])  # Set y-axis limits

            # Set the title of each subplots
            curr_ax.set_title(f'{row_feature_name}: {row_val}| {col_feature_name}: {col_val}', fontsize=12)

            # Remove the legend to create a separate collective legend for all plots
            curr_ax.get_legend().remove()            

            # Set the title of each subplots
            curr_ax.set_title(f'{row_feature_name}: {row_labels[row_val]} \n{col_feature_name}: {col_labels[col_val]}', fontsize=12)

            # Setting the x-axis labels
            curr_ax.tick_params(axis='x', rotation=45)
            curr_ax.xaxis.set_major_locator(plt.MaxNLocator(6)) 
            
    # Create a single legend for all the subplots
    # Get the legend from the axes (for different hue colors)    
    if(len(row_feature_values) > 1 or len(col_feature_values) > 1):
        legend_ax = axs[len(col_feature_values)-1]        
    else:
        legend_ax = axs
    all_handles, all_labels = legend_ax.get_legend_handles_labels()    
    
    # Set a colorbar if there are too many hue values 
    # if(len(types) > 10):
    #     num_hue_values = len(types)
    #     if(hue_feature_name == "age"):
    #         num_hue_values = list()
    #         for age in hue_feature_values:
    #             num = [each for each in age if str(each).isdigit()]
    #             num = int(''.join(num))
    #             num_hue_values.append(num)
    #     norm = plt.Normalize(min(num_hue_values), max(num_hue_values))
    #     sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    #     sm.set_array([])
    #     if(len(row_feature_values) > 1 and len(col_feature_values) > 1):
    #         cbar = legend_ax.figure.colorbar(sm, ax=axs2d[:,:], shrink=0.6, fraction=0.1)
    #     else:
    #         cbar = legend_ax.figure.colorbar(sm, ax=legend_ax, shrink=0.6, fraction=0.1)
    #     cbar.set_label(hue_feature_name, rotation=90)
    #     handles = []
    #     labels = []        

    #     # Add legends for each linestyle (type)
    #     legend_linestyle = []
    #     if(py_params.TYPE_TRUE in types):
    #         legend_linestyle.append(mlines.Line2D([], [], linestyle='-', label=py_params.TYPE_TRUE))    
    #     for prediction_type in predicted_types:    
    #         legend_linestyle.append(mlines.Line2D([], [], linestyle=(0, linestyles[prediction_type]), label=prediction_type))
    #     label_linestyle = [handle.get_label() for handle in legend_linestyle]
        
    #     # Combine automatic legend handles and custom legend handles
    #     all_handles = handles + legend_linestyle
    #     all_labels = labels + label_linestyle
        
    # Add legend with both automatic and custom legend entries
    # ncol_legend = len(hue_feature_values)/2 if len(hue_feature_values) > 1 else 1

    # Set the legend
    # print(legend_linestyle)
    fig.legend(handles=all_handles, labels=all_labels, loc='upper right',
            fancybox=True, shadow=True)
    fig.suptitle(title_fig, fontsize=24)
    fig.subplots_adjust(top=0.9, right=0.8)
    if(is_fig_saved):
        plt.savefig(f"{title_fig}.png")
    plt.show()   

def multi_channels_recursive_prediction_to_long_df(predictions, loader, 
                                                   features_cat, 
                                                   features_m,
                                                   label_inverse_transform, horizon,
                                                   df_dataset=None,                          
                                                   is_include_true_values=False):
    """
    A function to process multi-channels predictions (i.e. all age-groups simultaneously) into a long DataFrame complete with all its categorical inputs.

    Args:
        predictions (torch.Tensor): predictions made by an ANN in a list of 3D tensor (a list of size total_batches containing batch_size x age_group x horizon)
        loader (DataLoader): DataLoader to load the batch used to make predictions. The loader is used to take (categorical) inputs fed for making the predictions.
        features_cat (List[str]): a list of string specifying the column names for the categorical features.
        df_dataset (pd.DataFrame): a DataFrame containing the true values. If `is_include_true_values` equals True, then data in `df_dataset` will be added to the returned DataFrame as the true value (with a flag variable `type` = 'true')        
        horizon (int): the prediction horizon (how many years ahead are predicted)
        is_include_true_values (bool): a flag indicating whether records with true values should be added to the returned DataFrame
        label_inverse_transform (callable): 
    
    Return:
        A DataFrame object in a long format, where each record contains all categorical variables (including age_group), a `type` column to indicate true or predicted values, and the log_mortality
    """
    predictions_transformed = []
    df_cats = pd.DataFrame(columns=['year'] + features_cat)
    for batch_idx, batch in enumerate(loader):
        for i in range(0, batch['year'].shape[0]):
            # Take the categorical value of each record
            features_cat_values = []
            for cat in features_cat:
                features_cat_values.append(batch[cat][i])
            year =  batch['year'][i]

            # Construct a dictionary
            curr_dict = {'year': year}
            for cat_idx in range(len(features_cat)):
                curr_dict[features_cat[cat_idx]] = features_cat_values[cat_idx]

            # Construct a DataFrame for each record
            df_cats = pd.concat([df_cats, pd.DataFrame.from_dict(curr_dict)], ignore_index=True)
            
            # Inverse transform the predictions if scaled
            if(Scaler.is_mamiya(label_inverse_transform)):
                predictions_transformed.append(label_inverse_transform(sex = batch['sex'][i].numpy()[0],
                                                                       cause = batch['cause'][i].numpy()[0],
                                                                       data = predictions[batch_idx][i].float().numpy()))
            else:
                predictions_transformed.append(predictions[batch_idx][i].float().numpy())

    # Construct a DataFrame for the predictions, 
    # flattening the predictions from 2D of size (22, HORIZON) into 1D of size (22*HORIZON)
    predictions_transformed = np.array(predictions_transformed)
    df_preds = pd.DataFrame(predictions_transformed.reshape(-1, np.multiply(*predictions_transformed[0].shape)))

    # join the predictions and the categorical features
    df_join = df_cats.join(df_preds)

    # Reformat the constructed DataFrame into a long format
    df_preds_long = pd.DataFrame()
    # For each row
    for i in df_join.index:
        # Get the starting year
        predicted_year = df_join.iloc[i]['year']

        # Rename the prediction columns into AGE_GROUP_YEAR
        col_names = [age_group + "_" + str(predicted_year + i) for age_group in features_m for i in range(0, horizon)]
        df_curr_row = df_join.iloc[i].to_frame().transpose()
        df_curr_row.columns = list(df_curr_row.columns[0:3]) + col_names

        # Add the column type to mark this is a prediction made from `predicted_year`
        df_curr_row['type'] = 'predicted_' + str(predicted_year)

        # Melt the dataframe into long format
        df_curr_row = df_curr_row.melt(id_vars=list(df_curr_row.columns[0:3]) + ['type'],
                        value_name='log_mortality')
        
        # Replace the `year` column with the value of YEAR
        # Separate the AGE_GROUP_YEAR into AGE_GROUP and YEAR   
        df_curr_row.drop(columns=['year'], inplace=True)
        df_curr_row = df_curr_row.join(df_curr_row['variable'].str.split('_', n=1, expand=True)
                        .rename(columns={0: "age",1: "year"}))
        df_curr_row.drop(columns=['variable'], inplace=True)
        
        # Combine into one long DataFrame
        df_preds_long = pd.concat([df_preds_long, df_curr_row], ignore_index=True, axis=0)    

    df_preds_long['year'] = df_preds_long['year'].astype(int)

    # Add the original data to the long-formatted dataframe
    if (is_include_true_values) and (type(df_dataset) != None):
        df_true = df_dataset.copy().drop(columns=['country'])

        df_true = df_true.melt(id_vars=['year'] + features_cat,
                               value_name='log_mortality',
                               var_name='age')
        df_true['type'] = 'true'

        df_preds_long = pd.concat([df_true, df_preds_long], axis=0, ignore_index=True)
    
    return df_preds_long

def single_channel_prediction_to_long_df(predictions, loader, 
                                         features_cat, 
                                         label_inverse_transform, 
                                         prediction_type,
                                         df_dataset=None,                          
                                         is_include_true_values=False):
    """
    A function to process single-channel non-recursive predictions (i.e. single age-groups without historical data) into a long DataFrame complete with all its categorical inputs.

    Args:
        predictions (torch.Tensor): predictions made by an ANN to the entire loader in a 3D tensor (total_batches x batch_size x 1)
        loader (DataLoader): DataLoader to load the batch used to make predictions. The loader is used to take (categorical) inputs fed for making the predictions.
        features_cat (List[str]): a list of string specifying the column names for the categorical features.
        df_dataset (pd.DataFrame): a DataFrame containing the true values. If `is_include_true_values` equals True, then data in `df_dataset` will be added to the returned DataFrame as the true value (with a flag variable `type` = 'true')        
        is_include_true_values (bool): a flag indicating whether records with true values should be added to the returned DataFrame
        label_inverse_transform (callable): 
    
    Return:
        A DataFrame object in a long format, where each record contains all categorical variables (including age_group), a `type` column to indicate true or predicted values, and the log_mortality
    """
    predictions_transformed = []
    df_cats = pd.DataFrame(columns=['year'] + features_cat)
    for batch_idx, batch in enumerate(loader):
        for i in range(0, batch['year'].shape[0]):
            # Take the categorical value of each record
            features_cat_values = []
            for cat in features_cat:
                features_cat_values.append(batch[cat][i])
            year =  batch['year'][i]

            # Construct a dictionary
            curr_dict = {'year': year}
            for cat_idx in range(len(features_cat)):
                curr_dict[features_cat[cat_idx]] = features_cat_values[cat_idx]

            # Construct a DataFrame for each record
            if(df_cats.empty):
                df_cats = pd.DataFrame.from_dict(curr_dict)
            else:
                df_cats = pd.concat([df_cats, pd.DataFrame.from_dict(curr_dict)], ignore_index=True)
            
            # Inverse transform the predictions if scaled
            if(Scaler.is_mamiya(label_inverse_transform)):
                predictions_transformed.append(label_inverse_transform(sex = batch['sex'][i].numpy()[0],
                                                                       cause = batch['cause'][i].numpy()[0],
                                                                       data = predictions[batch_idx][i].reshape(-1, 1).float().numpy()))
            else:
                predictions_transformed.append(predictions[batch_idx][i].float().numpy())

    # Construct a DataFrame for the predictions, give it a column name 'log_mortality'
    df_preds = pd.DataFrame(np.array(predictions_transformed).squeeze(), columns=['log_mortality'])

    # Join the predictions and the categorical features (since there is only one prediction, the resulting DataFrame is already in the long format)
    df_preds_long = df_cats.join(df_preds)

    # Join the predictions and the categorical features (since there is only one prediction, the resulting DataFrame is already in the long format)
    df_preds_long[py_params.COL_TYPE] = prediction_type

    # Add the original data to the long-formatted dataframe
    if (is_include_true_values) and (type(df_dataset) != None):
        df_true = df_dataset.copy()

        df_true = df_true.melt(id_vars=['year'] + features_cat,
                               value_name='log_mortality',
                               var_name=py_params.COL_TYPE)
        df_true['type'] = 'true'

        df_preds_long = pd.concat([df_true, df_preds_long], axis=0, ignore_index=True)
    
    return df_preds_long