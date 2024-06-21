import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import matplotlib.pyplot as plt

from typing import List
from typing import Type

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR

import py_params

TYPE_DATA_DEATH = 'death'
TYPE_DATA_EXP = 'exposure'
TYPE_DATA_M = 'mortality'
TYPE_DATA_LM = 'log_mortality'

TYPE_RES_DF = 'df'
TYPE_RES_NP = 'np'

class HmdMortalityData():
    """
    A class used to apply various operations to the HMD COD dataset. The dataframe should have the following columns:
    - 'year': the column containing the year when the mortality rates are applied
    - 'sex': the column describing the gender (1 for male, 2 for female, 3 for both) of whom the mortality rates are applied
    - 'cause': the column describing the underlying cause for the mortality rates
    ' 'mXX': the column containing the mortality rates for age XX. The mortality columns in the DataFrame should be sorted from youngest to oldest.
    """
    def __init__(self, df_hmd_rates, df_hmd_exp):
        """
        Args:
            - df_hmd_rates (pd.DataFrame): a pandas DataFrame containing the cause-of-death mortality rate from hmd in wide format for ages.
            - df_hmd_exp (pd.DataFrame): a pandas DataFrame containing the exposure from hmd in wide format for ages. Note that the exposure does not have the 'cause' feature (column)
        """
        # Store the mortality dataframe in a sorted manner
        self.df_hmd_rates = df_hmd_rates.sort_values(by=py_params.COL_CO_YR_SX_CA)
                
        # Extract all unique years in the mortality dataset
        self.years = df_hmd_rates['year'].unique()
        
        # Extract all unique ages or age-groups in the mortality dataset
        self.col_m = [column for column in df_hmd_rates.columns if column.find('m') == 0]
        self.ages = [int(age[1:]) for age in self.col_m]
        self.col_e = [f"e{age}" for age in self.ages]                

        # Determine whether the data is subdivided into age-group or single-year of age
        self.is_age_group = False
        for i in range(1, len(self.ages)):
            if(self.ages[i] - self.ages[i-1] > 1):
                self.is_age_group = True

        # Filter, sort, and store the exposure dataset
        self.df_hmd_exp = df_hmd_exp.loc[df_hmd_exp['year'].isin(self.years)]
        self.df_hmd_exp.sort_values(by=py_params.COL_CO_YR_SX, inplace=True)

        # Extract all other columns (features) aside from the mortality rates (and exposures)
        self.col_others_m = [column for column in df_hmd_rates.columns if column.find('m') != 0]
        self.col_others_e = [column for column in df_hmd_exp.columns if column.find('e') != 0]
    
    def get_mxt(self, year_start, year_end, age_start, age_end, sex, cause, country):
        """
        A function to prepare the data from hmd format to a suitable format for traditional stochastic models in py_model_st module.
        The function will prepare a numpy array object based on the specified years, ages, sex, cause, and country
        Args:
            year_start: the first year of the extracted mortality rates
            year_end: the last year of the extracted mortality rates
            age_start: the first age of the extracted mortality rates
            age_end: the last age of the extracted mortality rates
            sex: the gender of the extracted mortality rates
            cause: the underlying cause of death of the extracted mortality rates
            country: the country of the extracted mortality rates
        Returns:
            a list of 3 elements:
                - a numpy array of the mortality rates with ages in rows and years in columns
                - a numpy array of all ages considered
                - a numpy array of all years considered
        """
        if year_start not in self.years:
            raise ValueError("First year is not in the dataset")
        if year_end not in self.years:
            raise ValueError("Last year is not in the dataset")
        if age_start not in self.ages:
            raise ValueError("First age is not in the dataset")
        if age_end not in self.ages:
            raise ValueError("Last age is not in the dataset")
        
        df_temp =  self.df_hmd_rates.loc[(self.df_hmd_rates['sex'] == sex) & 
                                         (self.df_hmd_rates['cause'] == cause) & 
                                         (self.df_hmd_rates['country'] == country) &
                                         (self.df_hmd_rates['year'] >= year_start) & 
                                         (self.df_hmd_rates['year'] <= year_end),].set_index('year')
        df_temp = df_temp.loc[:,[col for col in self.col_m if int(col[1:]) >= age_start and int(col[1:]) <= age_end]]        
        return [df_temp.to_numpy().T,
                df_temp.index.to_numpy(),
                df_temp.columns.to_numpy()]        
    
    def get_ext(self, year_start, year_end, age_start, age_end, sex, country):
        """
        A function to prepare the data from hmd format to a suitable format for traditional stochastic models in py_model_st module.
        The function will prepare a numpy array object based on the specified years, ages, sex, cause, and country
        Args:
            year_start: the first year of the extracted exposures
            year_end: the last year of the extracted exposures
            age_start: the first age of the extracted exposures
            age_end: the last age of the extracted exposures
            sex: the gender of the extracted exposures
            country: the country of the extracted exposures
        Returns:
            a list of 3 elements:
                - a numpy array of the mortality rates with ages in rows and years in columns
                - a numpy array of all ages considered
                - a numpy array of all years considered
        """
        if year_start not in self.years:
            raise ValueError("First year is not in the dataset")
        if year_end not in self.years:
            raise ValueError("Last year is not in the dataset")
        if age_start not in self.ages:
            raise ValueError("First age is not in the dataset")
        if age_end not in self.ages:
            raise ValueError("Last age is not in the dataset")
        
        df_temp = self.df_hmd_exp.loc[(self.df_hmd_exp['sex'] == sex) & 
                                      (self.df_hmd_exp['country'] == country) &
                                      (self.df_hmd_exp['year'] >= year_start) &
                                      (self.df_hmd_exp['year'] <= year_end)].set_index('year')
        df_temp = df_temp.loc[:,[col for col in df_temp.columns if col[0] == 'e' and int(col[1:]) >= age_start and int(col[1:]) <= age_end]]
        return [df_temp.to_numpy().T,
                df_temp.index.to_numpy(),
                df_temp.columns.to_numpy()]

    def to_wide(self, data_type:str, countries:List[str]=None, sexes:List[int]=None, causes:List[int]=None, 
                years:List[int]=None, ages:List[int]=None, 
                start_year:int=None, end_year:int=None,
                start_age:int=None, end_age:int=None):
        """
        Args:
            data_type (str): the type of data to be extracted ('death' for total number of death, 'exposure' for exposure, 'mortality' for mortality rates, 'log_mortality' for log mortality rates). Use provided constants in this module.
            country (List[str]): a list of countries 
            sexes (List[int]): a list of integer indicating the genders to be included (1 for male, 2 for female, 3 for both) 
            causes (List[int]): a list of integer indicating the cause of death to be included
            years (List[int]): a list of integer specifying the years to be included
            ages (List[int]): a list of integer specifying the ages to be included
            start_year (int): the first year to be included. Only applied when years = None
            end_year (int): the last year to be included. Only applied when years = None
            start_age (int): the first age to be included. Only applied when ages = None
            end_age (int): the last age to be included. Only applied when ages = None
        Return:
            pd.DataFrame: a dataframe filtered by the given lists in a wide format for the ages (there will be 1 column containing the 'type' data for each age)
        """
        # Check whether ranges or the first and last year (or age) are given
        years = self._check_ranges(years, start_year, end_year)
        ages = self._check_ranges(ages, start_age, end_age)                
       
        # If exposure is requested
        if(data_type == TYPE_DATA_EXP):            
            return self._filter(TYPE_DATA_EXP, countries, sexes, causes, years, ages).copy()
        
        # Filter the dataframe according to the given lists
        df_returned = self._filter(TYPE_DATA_M, countries, sexes, causes, years, ages).copy()

        # Filter the ages for further processing (converting to other forms of info)
        filtered_col_m = self.col_m        
        filtered_col_e = self.col_e        
        if(ages != None):
            filtered_col_m = [f"m{age}" for age in ages]
            filtered_col_e = [f"e{age}" for age in ages]
        else:
            ages = self.ages

        # If mortality rates is requested
        if(data_type == TYPE_DATA_M):
            return df_returned
        # If log mortality is requested
        elif(data_type == TYPE_DATA_LM):            
            df_returned.loc[:, filtered_col_m] = np.log(df_returned.loc[:, filtered_col_m])
            df_returned.rename(columns={old:new for (old, new) in zip(filtered_col_m, [f"lm{age}" for age in ages])}, inplace=True)
            return df_returned
        # If the number of deaths is requested
        elif(data_type == TYPE_DATA_DEATH):        
            # Need to ensure that the mortality rates and exposures correspond to each other (both are for the same countries, years, sexes, causes, and ages)    
            # For each country, sex, and cause
            countries = df_returned['country'].unique()
            sexes = df_returned['sex'].unique()
            causes = df_returned['cause'].unique()
            years = df_returned['year'].unique()
            for idx_country in countries:
                for idx_sex in sexes:
                    for idx_cause in causes:                                                
                        df_returned.loc[(df_returned['country'] == idx_country) & 
                                        (df_returned['sex'] == idx_sex) & 
                                        (df_returned['cause'] == idx_cause), filtered_col_m] = df_returned.loc[(df_returned['country'] == idx_country) & 
                                                                                                           (df_returned['sex'] == idx_sex) & 
                                                                                                           (df_returned['cause'] == idx_cause), filtered_col_m].values * self.df_hmd_exp.loc[(self.df_hmd_exp['country'] == idx_country) & 
                                                                                                                                                                                         (self.df_hmd_exp['sex'] == idx_sex), filtered_col_e].values                        
            # rename the columns to d to signifies death
            df_returned.rename(columns={old:new for (old, new) in zip(filtered_col_m, [f"d{age}" for age in ages])}, inplace=True)
            return df_returned
        else:
            return None

    def to_long(self, data_type:str, countries:List[str]=None, sexes:List[int]=None, causes:List[int]=None, 
                years:List[int]=None, ages:List[int]=None, 
                start_year:int=None, end_year:int=None,
                start_age:int=None, end_age:int=None):
        """
        Args:
            data_type (str): the type of data to be extracted ('death' for total number of death, 'exposure' for exposure, 'mortality' for mortality rates, 'log_mortality' for log mortality rates). Use provided constants in this module.
            country (List[str]): a list of countries 
            sexes (List[int]): a list of integer indicating the genders to be included (1 for male, 2 for female, 3 for both) 
            causes (List[int]): a list of integer indicating the cause of death to be included
            years (List[int]): a list of integer specifying the years to be included
            ages (List[int]): a list of integer specifying the ages to be included
            start_year (int): the first year to be included. Only applied when years = None
            end_year (int): the last year to be included. Only applied when years = None
            start_age (int): the first age to be included. Only applied when ages = None
            end_age (int): the last age to be included. Only applied when ages = None
        Return:
            pd.DataFrame: a dataframe filtered by the given lists in a long format
        """
        # Use to_wide to filter and get the dataframe correspond to each data type (exposure, mortality, log mortality, deaths)
        df_returned = self.to_wide(data_type=data_type, countries=countries, sexes=sexes, causes=causes, years=years, ages=ages,
                                   start_year=start_year, end_year=end_year, start_age=start_age, end_age=end_age)
        
        # Determine the name of the column depending on the data type requested
        if(data_type == TYPE_DATA_EXP):
            value_name = "exposure"
        elif(data_type == TYPE_DATA_M):
            value_name = "mortality"
        elif(data_type == TYPE_DATA_LM):
            value_name = "log_mortality"
        elif(data_type == TYPE_DATA_DEATH):
            value_name = "death"
            
        # Melt the dataframe into long format
        if df_returned is not None:
            return df_returned.melt(id_vars=['country', 'sex', 'cause', 'year'],var_name='age', value_name=value_name)
        else:
            return None
        
        # # Check whether ranges or the first and last year (or age) are given
        # years = self._check_ranges(years, start_year, end_year)
        # ages = self._check_ranges(ages, start_age, end_age)

        # # If the exposure is requested
        # if(data_type == TYPE_EXP):            
        #     return self._filter(TYPE_EXP, countries, sexes, causes, years, ages).melt(id_vars=['country', 'sex', 'cause', 'year'],var_name='age', value_name='exposure')
        
        # df_returned = self._filter(TYPE_M, countries, sexes, causes, years, ages).copy()
        
        # # If the mortality rate is requested
        # if(data_type == TYPE_M):
        #     return df_returned.melt(id_vars=['country', 'sex', 'cause', 'year'],var_name='age', value_name='mortality')
        # # If the log mortality is requested
        # elif(data_type == TYPE_LM):            
        #     df_returned.loc[:, self.col_m] = np.log(df_returned.loc[:, self.col_m])
        #     return df_returned.melt(id_vars=['country', 'sex', 'cause', 'year'],var_name='age', value_name='log_mortality')
        # # If the number of deaths is requested
        # elif(data_type == TYPE_DEATH):     
        #     # Need to ensure that the mortality rates and exposures correspond to each other (both are for the same countries, years, sexes, causes, and ages)    
                   
        #     df_returned.loc[:, self.col_m] = self.df_hmd_rates[self.col_m].values * self.df_hmd_exp[self.col_e].values
        #     df_returned.rename(columns={old:new for (old, new) in zip(self.col_m, [f"d{age}" for age in self.ages])}, inplace=True)
        #     return df_returned.melt(id_vars=['country', 'sex', 'cause', 'year'],var_name='age', value_name='death')
        # else:
        #     return None

    def _filter(self, data_type:str, countries:List[str] = None, 
                sexes:List[int] = None, causes:List[int] = None, 
                years:List[int] = None, ages:List[int] = None):
        """
        Function to filter the mortality dataframes based on the given values for each column. Set None if no filter should be applied.
        Args:
            data_type (str): either 'mortality' or 'exposure' to specify which dataframe to be filtered
            country (List[str]): a list of countries 
            sexes (List[int]): a list of integer indicating the genders to be included (1 for male, 2 for female, 3 for both) 
            causes (List[int]): a list of integer indicating the cause of death to be included
            years (List[int]): a list of integer specifying the years to be included
            ages (List[int]): a list of integer specifying the ages to be included
        Raises:
            ValueError: if one value in any of the five lists do not exist in the original data frame given

        Returns:
            pd.DataFrame: a dataframe filtered by the given lists
        """

        if(data_type==TYPE_DATA_EXP and causes is not None):
            raise ValueError("Exposure data does not have cause.")
        
        if(countries == None):
            countries = self.df_hmd_rates['country'].unique()
        if(sexes == None):
            sexes = self.df_hmd_rates['sex'].unique()            
        if(causes == None):
            causes = self.df_hmd_rates['cause'].unique()
        if(years == None):
            years = self.years
        if(ages == None):
            ages = self.ages
            
        # Check whether the given feature values are all contained in the dataframe
        if(not all(elem in self.ages for elem in ages)):
            raise ValueError("Some ages given are not in the mortality dataframe.")
        if(not all(elem in self.years for elem in years)):
            raise ValueError("Some years given are not in the mortality dataframe.")
        if(not all(elem in self.df_hmd_rates.country.unique() for elem in countries)):
            raise ValueError("Some countries given are not in the mortality dataframe.")
        if(not all(elem in self.df_hmd_rates.cause.unique() for elem in causes)):
            raise ValueError("Some causes given are not in the mortality dataframe.")
        
        # Make columns for the given ages
        if(data_type == TYPE_DATA_M):            
            col_m = [f'm{age}' for age in ages]
            return self.df_hmd_rates.loc[(self.df_hmd_rates['country'].isin(countries)) & 
                                        (self.df_hmd_rates['sex'].isin(sexes)) & 
                                        (self.df_hmd_rates['cause'].isin(causes)) &
                                        (self.df_hmd_rates['year'].isin(years)), self.col_others_m + col_m]
        elif(data_type == TYPE_DATA_EXP):
            col_e = [f'e{age}' for age in ages]
            return self.df_hmd_exp.loc[(self.df_hmd_exp['country'].isin(countries)) & 
                                        (self.df_hmd_exp['sex'].isin(sexes)) & 
                                        (self.df_hmd_exp['year'].isin(years)), self.col_others_e + col_e]
        else:
            raise ValueError("Type can only either 'mortality' or 'exposure'.")
            
    def diff(self, data_other: 'HmdMortalityData', data_type:str,
             countries:List[str]=None, sexes:List[int]=None, causes:List[int]=None, 
             years:List[int]=None, ages:List[int]=None, 
             start_year:int=None, end_year:int=None,
             start_age:int=None, end_age:int=None):
        """
        Find the difference of value (specified through the 'type' parameter) between this and other HmdMortalityData.
        The function include some optional parameter to filter which country, gender, cause, years, and ages to be included in the calculation.

        If the given HmdMortalityData contains different feature values (country, gender, cause, years, and ages), only the difference for features contained in both data is calculated.

        Args:
            data_other (HmdMortalityData): another object of HmdMortalityData
            data_type (str): the type of data to be calculated ('death' for total number of death, 'exposure' for exposure, 'mortality' for mortality rates, 'log_mortality' for log mortality rates). Use provided constants in this module.
            country (List[str]): a list of countries 
            sexes (List[int]): a list of integer indicating the genders to be included (1 for male, 2 for female, 3 for both) 
            causes (List[int]): a list of integer indicating the cause of death to be included
            years (List[int]): a list of integer specifying the years to be included
            ages (List[int]): a list of integer specifying the ages to be included            
            start_year (int): the first year to be included. Only applied when years = None
            end_year (int): the last year to be included. Only applied when years = None
            start_age (int): the first age to be included. Only applied when ages = None
            end_age (int): the last age to be included. Only applied when ages = None

        Returns:
        """        
        # Check whether ranges or the first and last year (or age) are given
        years = self._check_ranges(years, start_year, end_year)
        ages = self._check_ranges(ages, start_age, end_age)

        # Find the smallest set of unfiltered feature values (features with None arguments) between the two dataframes
        if(countries == None):
            this_values = self.df_hmd_rates['country'].unique()
            other_values = data_other.df_hmd_rates['country'].unique()
            countries = list(set(this_values) & set(other_values))
        if(sexes == None):
            this_values = self.df_hmd_rates['sex'].unique()
            other_values = data_other.df_hmd_rates['sex'].unique()
            sexes = list(set(this_values) & set(other_values))          
        if(causes == None):
            this_values = self.df_hmd_rates['cause'].unique()
            other_values = data_other.df_hmd_rates['cause'].unique()
            causes = list(set(this_values) & set(other_values))    
        if(years == None):
            this_values = self.years
            other_values = data_other.years
            years = list(set(this_values) & set(other_values))    
        if(ages == None):
            this_values = self.ages
            other_values = data_other.ages
            ages = list(set(this_values) & set(other_values))    
            
        
        # Filter this data and to ensure only feature values contained in this data is requested            
        df_returned = self._filter(data_type=data_type, countries=countries, sexes=sexes, 
                                causes=causes, years=years, ages=ages).copy()            

        # Filter so that both data include the same features values (country, sex, cause, year, and age)
        df_other = data_other._filter(data_type=data_type,
                                        countries=countries, sexes=sexes, 
                                        causes=causes, years=years, ages=ages).copy()   
                        
        # Sort both data so both have the same order (and thus they correspond to each other)
        df_returned.sort_values(by=py_params.COL_CO_YR_SX_CA, inplace=True)
        df_other.sort_values(by=py_params.COL_CO_YR_SX_CA, inplace=True)
        
        # Take columns that contains the mortality (or exposure) data only
        cols = [col for col in df_returned.columns if col not in self.col_others_m]
        
        # Substract df_returned (this dataframe) with the df_other (the given df)
        df_returned.loc[:, cols] = df_returned.loc[:, cols].values() -  df_other.loc[:, cols].values()
        return df_returned                    
        
    def _check_ranges(self, values:List[int], start_range:int, end_range:int):
        """
        A function to check whether a range or only the starting and ending values are given
        Args:
            arange (List[int]): a list of values in integer
            start_range (int): the starting value of a range
            end_range (int): the ending value of a range
        Returns:
            the argument range if given, a range of values from start_range to end_range if both are given, or None
        """
        # If ranges is not given, but the start and end of the range is given
        if(values is None and start_range is not None and end_range is not None):
            if(start_range > end_range):
                raise ValueError('The beginning of the range should be smalled than the end.')            
            return list(range(start_range, end_range+1))
        # If ranges is given
        elif(values is not None):
            return values
        # If nothing is given
        else:
            return None
        
class HmdResidualData():
    def __init__(self, df_true_long:pd.DataFrame, df_pred_long:pd.DataFrame,
                 year_train_end, year_val_end = None):
        self.df_true = df_true_long
        self.df_pred = df_pred_long

        # Calculate the residuals (by sorting first, then applying a matrix operation)
        self.df_true.sort_values(by=py_params.COL_CO_YR_SX_CA, inplace=True)
        self.df_pred.sort_values(by=py_params.COL_CO_YR_SX_CA, inplace=True)
        self.df_res = self.df_true.copy()
        self.df_res.loc[:,"log_mortality"] = self.df_pred.loc[:,"log_mortality"].values - self.df_true.loc[:,"log_mortality"].values            

        # Add info on which ones are train, valid (if specified) and test data
        self.df_res['type'] = py_params.TYPE_TRAIN
        if(year_val_end is not None):
            self.df_res.loc[(self.df_res.year > year_train_end) & (self.df_res.year <= year_val_end), 'type'] = py_params.TYPE_VAL
            self.df_res.loc[(self.df_res.year > year_val_end), 'type'] = py_params.TYPE_TEST
        else:
            self.df_res.loc[(self.df_res.year > year_train_end), 'type'] = py_params.TYPE_TEST

    def residual(self, res_type:str, by:List[str]):
        df_returned = self.df_res.copy()

        # if(res_type == (TYPE_RES_Ddf_returned = self.df_res.loc[df_returned = self.df_res.year > year_train_end + ) & (TYPE_RES_Ddf_returned = self.df_res.loc[df_returned = self.df_res.year > year_train_end <= year_val_end)
        #     return df_returned, 'type'] = py_params.TYPE_VAL
        #     return df_returned, 'type'] = py_params.TYPE_Vvalyear_val_end:py_params.TYPE_VAL
        # elif(TYPE_TEST

    def mse(self, res_type:str, by:List[str]):
        pass

    def heatmap(self, by:List[str]):
        pass

    def _summarize(self, by:List[str]):
        pass