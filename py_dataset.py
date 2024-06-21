import pandas as pd
import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.preprocessing import MinMaxScaler

import py_params
import py_utils

from typing import List

from PIL import __version__ as PILLOW_VERSION
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import RandomSampler


class MortalityUcodWindowedDataset(Dataset):
    """
    Dataset for mortality dataset with cause of death details. 
    The dataset will accept a mortality dataframe with the same structure as explained in the beginning of this notebook.
    The mortality data used in this notebook needs to have the following format:
    1. Age (or age-group) must be in wide format, where each column contains the mortality rate for each age (or age-group)
    2. `year` for the year when the death is recorded in `int`.
    3. `country` for multi-population setting in `int` using the code given by WHO.
    4. `cause` for the underlying cause of death in `int`. The code for this part is intended to be obtained from grouping the various CoD into a smaller category, coded using an integer [1,...].
    5. `sex` for the gender in `int`, where `1` is male, `2` is female, and `3` is total.
    
    The dataset will extract from the dataframe the following features
    1. Categorical features that can be specified through the constructor's argument.
    2. Mortality rates in a 2D matrix form where rows indicate all considered ages (or age-groups) and columns indicate years. seq_len will determine how many years will be included for each datum (window's length)
    3. Label in a similar form as mortality rates (rows indicate ages and columns indicate years). forecast_horizon will determine how many years will be included for each datum (window's length)
    Therefore, each datum will has a mortality rates of seq_len years in the past and some categorical features for a particular year. 
    """
    def __init__(self, df_mortality: pd.DataFrame, mortality_features: list, 
                 year_start:int, year_end:int, train: bool, label_transform: callable,
                 categorical_features = [], 
                 seq_len = py_params.SEQ_LEN, 
                 forecast_horizon = py_params.FORECAST_HORIZON):
        """
        Construct a dataset object from the given mortality dataframe. 
        Args:
            df_mortality (pd.DataFrame): the dataframe containing the mortality rate and additional categorifal features
            mortality_features (list): list of columns' name specifying which column contains the mortality rate of a particular age-group            
            year_start (int): the first year to be considered in the dataset. This is the first year to be included in the input, thus the first label will be at year_start + seq_len.
            year_end (int): the last year to be considered in the dataset. This year_end will be included as the last label in the dataset
            train (bool): a boolean value indicating whether the dataset will be used in training or validation (or testing)
            categorical_features (list): a list of columns' name specifying which column will be used as categorical features
            seq_len (int, optional): the total years of historical data to be considered for each datum. Defaults to 10.
            forecast_horizon (int): the total years of future data will be predicted (thus included as true value) for each datum.
            label_transform (callable): a function used to transform the label (mortality rates to be predicted)
        """        
        self.year_start = year_start
        self.year_end = year_end
        self.df = df_mortality.loc[(df_mortality['year'] >= year_start) & (df_mortality['year'] <= year_end),
                                   categorical_features + mortality_features + ['year']].reset_index()
        self.mortality_features = mortality_features
        self.categorical_features = categorical_features
        self.train = train
        self.seq_len = seq_len
        self.forecast_horizon = forecast_horizon
        self.label_transform = label_transform

    def __len__(self):        
        # If building training set, the prediction is made only one step ahead
        if(self.train):
            return self.df.loc[(self.df.year >= self.year_start + self.seq_len) & 
                               (self.df.year <= self.year_end), :].shape[0]
        
        # If building validation or testing set, the prediction depends on the specified horizon 
        # (predictions can be made a few years ahead)
        else:
            return self.df.loc[(self.df.year >= self.year_start + self.seq_len) & 
                               (self.df.year <= self.year_end - (self.forecast_horizon - 1)), :].shape[0]

    def __getitem__(self, index):
        # Get the data at the given index
        if(self.train):
            datum = self.df.loc[(self.df['year'] >= self.year_start + self.seq_len)].iloc[index,:]
        else:
            datum = self.df.loc[(self.df['year'] >= self.year_start + self.seq_len) & 
                                (self.df['year'] <= self.year_end - (self.forecast_horizon - 1)) ].iloc[index,:]
        
        # Get the first and last years of historical data for a particular index
        year_start_feat = datum['year'] - self.seq_len
        year_end_feat = datum['year'] - 1  
        
        # Get all the necessary information to filter
        categorical_values = [value for value in datum[self.categorical_features]]
        
        # Filter the dataframe to only include the mortality rates between the first and last years with the current categorical features
        df_current_train = self.df.loc[(self.df['year'] >= year_start_feat) & (self.df['year'] <= year_end_feat)]
        for i in range(0, len(self.categorical_features)):
            df_current_train = df_current_train.loc[df_current_train[self.categorical_features[i]] == categorical_values[i], :]
        
        # Construct the mortality features and the labels
        X_m = torch.tensor(np.transpose(df_current_train.loc[:, self.mortality_features].to_numpy()), dtype=torch.float32)
        X_year = torch.tensor([datum['year']], dtype=torch.int32)
    
        # Constructing label for training dataset
        # Get the MinMaxScaler
        if(self.train):            
            if(py_utils.Scaler.is_mamiya(self.label_transform)):
                y = torch.tensor(self.label_transform(sex = int(datum['sex']), 
                                                      cause = int(datum['cause']),
                                                      data = datum[self.mortality_features].to_numpy().reshape(-1,1)), 
                                 dtype=torch.float32)
            else:                
                y = torch.tensor(np.expand_dims(datum[self.mortality_features].to_numpy(), axis=-1), 
                                 dtype=torch.float32)
        # Constructing label for validation or testing dataset
        else:
            # Get the first and last years of future data to be forecasted for a particular index
            year_start_label_test = datum['year']
            year_end_label_test = datum['year'] + (self.forecast_horizon - 1)
            
            # Filter the data to get necessary the mortality rate
            df_current_test = self.df.loc[(self.df['year'] >= year_start_label_test) & (self.df['year'] <= year_end_label_test)]
            for i in range(0, len(self.categorical_features)):
                df_current_test = df_current_test.loc[df_current_test[self.categorical_features[i]] == categorical_values[i], :]

            # Construct the label into a matrix of AGE_GROUP x YEAR
            if(py_utils.Scaler.is_mamiya(self.label_transform)):
                y = torch.tensor(self.label_transform(sex = int(datum['sex']), 
                                                      cause = int(datum['cause']),
                                                      data = np.transpose(df_current_test.loc[:, self.mortality_features].to_numpy()).reshape(-1,1)) 
                                 .reshape(len(self.mortality_features),self.forecast_horizon), 
                                 dtype=torch.float32)                
            else:
                y = torch.tensor(np.transpose(df_current_test.loc[:, self.mortality_features].to_numpy()), 
                   dtype=torch.float32)
        
        # Combining all features and labels into a dictionary
        dict_returned = {self.categorical_features[i]: torch.tensor([categorical_values[i]], dtype=torch.int32) for i in range(0, len(categorical_values))}
        dict_returned

        dict_returned[py_params.FEATURES_M_NAME] = X_m
        dict_returned[py_params.FEATURES_LABEL_NAME] = y
        dict_returned['year'] = X_year
        return dict_returned

class MortalityUcodWindowedDataModule(pl.LightningDataModule):
    """
    DataLoader to be used with PyTorch Lightning. This class will use the MortalityUcodWindowedDataset class as its dataset,
    and construct training, validation, and testing batches according to a given batch_size.

    Training batches is constructed with one-year forecasting horizon, which means each datum will have only one year of data to be forecasted.
    There is an additional setting called `is_bootstrap`. 
    When True, each training batch will be built by sampling with replacement from the given training set.
    When False, each training batch will be built normally with shuffling.

    Validation and testing batches are constructed with multi-year forecasting horizon (set by the forecast_horizon parameter).
    Both batches will not be shuffled.
    """

    def __init__(self, df_mortality: pd.DataFrame, batch_size: int, steps: int, num_workers: int,
                 is_bootstrap: bool,
                 ratios: list,
                 seq_len, categorical_features, mortality_features, 
                 forecast_horizon = py_params.FORECAST_HORIZON, is_label_scaled = py_params.IS_LABEL_SCALED,
                 seed = py_params.SEED_TORCH):
        """
        Args:
            df_mortality: the dataframe containing the log mortality rates at a particular 'year' complete with all categorical features describing each particular record
            batch_size: the number of sample in each batch
            steps: the number of batches when bootstraping is True
            num_workers:
            is_bootstrap: a flag to indicate whether bootstraping should be done when constructing training batch
            ratios: a list of float containing the percentages for training, validation and testing data respectively, or a list of years specifying the last year to be considered as training, validation, and testing data respectively.
            seq_len: the number of years to look back when constructing the mortality features
            categorical_features: a list of column names considered as categorical features
            mortality_features: a list of column names for the log mortality rate, each column specifying the mortality rate of a particular age or age-group
        """

        super().__init__()
        self.df_mortality = df_mortality        
        self.batch_size = batch_size
        self.steps = steps
        self.num_workers = num_workers
        self.is_bootstrap = is_bootstrap        
        self.seq_len = seq_len
        self.forecast_horizon = forecast_horizon
        self.is_label_scaled = is_label_scaled
        self.categorical_features = categorical_features
        self.mortality_features = mortality_features
        self.seed = seed
        
        # Ensure there are 3 inputs
        assert (len(ratios) == 3)
        
        # Determining the included years for training, validation, and test dataset
        # If ratio is given
        if(sum(ratios) == 1):
            ttl_years = (self.df_mortality['year'].max() - self.df_mortality['year'].min()) + 1        
            self.first_year_train = self.df_mortality['year'].min()
            self.last_year_train = self.first_year_train + int(ttl_years * ratios[0])
            self.last_year_valid = (self.last_year_train) + int(ttl_years * ratios[1])
            self.last_year_test = self.df_mortality['year'].max()        
        # If not a ratio
        else:
            min_year = self.df_mortality['year'].min()
            max_year = self.df_mortality['year'].max()
            # Ensure all dataset are used, also the boundaries of training, validation, and testing set are valid
            assert (ratios[2] == max_year) & (ratios[0] < ratios[1]) & (ratios[1] < ratios[2]) & (ratios[0] in range(min_year, max_year + 1))
            self.first_year_train = min_year
            self.last_year_train = ratios[0]
            self.last_year_valid = ratios[1]
            self.last_year_test = ratios[2]
    
        # Scaler 
        self.scaler = py_utils.Scaler(self.df_mortality, self.last_year_train, mortality_features)
        
    def setup(self, stage):
        if(self.is_label_scaled):
            label_transform = self.scaler.mamiya_transform
        else:
            label_transform = None
        self.train_ds = MortalityUcodWindowedDataset(df_mortality=self.df_mortality,
                                                      year_start=self.first_year_train,
                                                      year_end=self.last_year_train,
                                                      train=True,
                                                      forecast_horizon=self.forecast_horizon,
                                                      label_transform=label_transform,
                                                      categorical_features=self.categorical_features,
                                                      mortality_features=self.mortality_features,
                                                      seq_len=self.seq_len)
        self.val_ds = MortalityUcodWindowedDataset(df_mortality=self.df_mortality,
                                                      year_start=self.last_year_train+1-py_params.SEQ_LEN,
                                                      year_end=self.last_year_valid,
                                                      train=False,
                                                      forecast_horizon=self.forecast_horizon,
                                                      label_transform=label_transform,
                                                      categorical_features=self.categorical_features,
                                                      mortality_features=self.mortality_features,
                                                      seq_len=self.seq_len)
        self.test_ds = MortalityUcodWindowedDataset(df_mortality=self.df_mortality,
                                                      year_start=self.last_year_valid+1-py_params.SEQ_LEN,
                                                      year_end=self.last_year_test,
                                                      train=False,
                                                      forecast_horizon=self.forecast_horizon,
                                                      label_transform=label_transform,
                                                      categorical_features=self.categorical_features,
                                                      mortality_features=self.mortality_features,
                                                      seq_len=self.seq_len)
        
        torch.manual_seed(self.seed)
        self.sampler = RandomSampler(data_source=self.train_ds, replacement=True,
                                     num_samples=self.batch_size * self.steps)        
    
    def train_dataloader(self):
        if(self.is_bootstrap):
            return DataLoader(
                self.train_ds,
                batch_size=self.batch_size,
                sampler=self.sampler,
                num_workers = self.num_workers
            )
        else:
            return DataLoader(
                self.train_ds,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers = self.num_workers
            )
        
    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers = self.num_workers)
        
    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers = self.num_workers)

class MortalityUcodLongDataset(Dataset):
    """
    Dataset for mortality dataset with cause of death details.
    The dataset will accept a mortality dataframe with the same structure as explained in the beginning of this notebook.
    The mortality data used in this notebook needs to have the following format:
    1. Age (or age-group) must be in wide format, where each column contains the mortality rate for each age (or age-group)
    2. `year` for the year when the death is recorded in `int`.
    3. `country` for multi-population setting in `int` using the code given by WHO.
    4. `cause` for the underlying cause of death in `int`. The code for this part is intended to be obtained from grouping the various CoD into a smaller category, coded using an integer [1,...].
    5. `sex` for the gender in `int`, where `1` is male, `2` is female, and `3` is total.
    
    The dataset will extract from the dataframe the following features
    1. Categorical features that can be specified through the constructor's argument.
    2. Label that is the mortality rate to be predicted for a set of categorical features
    Therefore, each datum will have some categorical features (including age or age group) for a particular year, each of which is used to predict the mortality rate of each 'population'
    """
    def __init__(self, df_mortality: pd.DataFrame, mortality_features: str, 
                year_start:int, year_end:int, label_transform: callable,
                categorical_features):
        """
        Args:
            df_mortality (pd.DataFrame): the dataframe containing the mortality rates (in a wide format) and additional categorifal features
            mortality_features (list): list of columns' name specifying which column contains the mortality rate of a particular age-group
            year_start (int): the first year to be considered in the dataset. 
            year_end (int): the last year to be considered in the dataset. 
            categorical_features (list): a list of columns' name specifying which column will be used as categorical featuress
            label_transform (callable): a function used to transform the label (mortality rates to be predicted)
        """
        # Filter the dataset to contain only the necessary features in the specified years
        self.df = df_mortality.loc[(df_mortality['year'] >= year_start) &
                                   (df_mortality['year'] <= year_end), categorical_features + ['year'] + mortality_features]
        
        # Change the name of the mortality column so that age-group name can be converted into LongTensor
        self.df.rename(columns=py_params.COL_MORTALITY_HMD_MAP, 
                       inplace=True)
        
        # Convert the dataframe into long format
        self.df = pd.melt(self.df,
                          id_vars = categorical_features + ['year'], 
                          value_vars=py_params.COL_MORTALITY_HMD_MAP.values(), var_name=py_params.FEATURES_AGE_NAME,
                          value_name=py_params.FEATURES_LABEL_NAME)
        self.df[py_params.FEATURES_AGE_NAME] = self.df[py_params.FEATURES_AGE_NAME].astype(int)
        
        # Store all the given parameters
        self.categorical_features = categorical_features + ['year'] + [py_params.FEATURES_AGE_NAME]
        self.label_transform = label_transform
        self.year_start = year_start
        self.year_end = year_end
    
    def __len__(self):        
        return self.df.shape[0]

    def __getitem__(self, index):
        # Create a dictionary to store the data
        dict_returned = {}

        # Get the data at the given index
        datum = self.df.iloc[index,:]
        
        # Get all the necessary information to filter
        categorical_values = [value for value in datum[self.categorical_features]]
                
        # Constructing label for training dataset            
        if(py_utils.Scaler.is_mamiya(self.label_transform)):
            y = torch.tensor(self.label_transform(sex = int(datum['sex']), 
                                                  cause = int(datum['cause']),
                                                  data = datum[py_params.FEATURES_LABEL_NAME]), 
                             dtype=torch.float32)
        else:                
            y = torch.tensor(np.expand_dims(datum[py_params.FEATURES_LABEL_NAME], axis=-1), 
                            dtype=torch.float32)
        
        # Combining all features and labels into a dictionary
        dict_returned = {self.categorical_features[i]: torch.tensor([categorical_values[i]], dtype=torch.int32) for i in range(0, len(categorical_values))}
        dict_returned[py_params.FEATURES_LABEL_NAME] = y
        dict_returned['year'] = torch.tensor(np.expand_dims(datum['year'], axis=-1), 
                            dtype=torch.float32)
        return dict_returned

class MortalityUcodLongDataModule(pl.LightningDataModule):
    """
    DataLoader to be used with PyTorch Lightning. This class will use the MortalityUcodLongDataset class as its dataset,
    and construct training, validation, and testing batches according to a given batch_size.
    """

    def __init__(self, df_mortality: pd.DataFrame, batch_size: int, steps: int, num_workers: int,
                 is_bootstrap: bool,
                 ratios: list,
                 categorical_features: List[str], mortality_features: List[str], 
                 is_label_scaled = py_params.IS_LABEL_SCALED,
                 seed = py_params.SEED_TORCH):
        """
        Args:
            df_mortality: the dataframe containing the log mortality rates at a particular 'year' complete with all categorical features describing each particular record
            batch_size: the number of sample in each batch
            steps: the number of batches when bootstraping is True
            num_workers:
            is_bootstrap: a flag to indicate whether bootstraping should be done when constructing training batch
            ratios: a list of float containing the percentages for training, validation and testing data respectively, or a list of years specifying the last year to be considered as training, validation, and testing data respectively.
            categorical_features: a list of column names considered as categorical features
            mortality_features: a list of column names for the log mortality rate, each column specifying the mortality rate of a particular age or age-group
        """

        super().__init__()
        self.df_mortality = df_mortality        
        self.batch_size = batch_size
        self.steps = steps
        self.num_workers = num_workers
        self.is_bootstrap = is_bootstrap        
        self.is_label_scaled = is_label_scaled
        self.categorical_features = categorical_features
        self.mortality_features = mortality_features
        self.seed = seed
        
        # Ensure there are 3 inputs
        assert (len(ratios) == 3)
        
        # Determining the included years for training, validation, and test dataset
        # If ratio is given
        if(sum(ratios) == 1):
            ttl_years = (self.df_mortality['year'].max() - self.df_mortality['year'].min()) + 1        
            self.first_year_train = self.df_mortality['year'].min()
            self.last_year_train = self.first_year_train + int(ttl_years * ratios[0])
            self.last_year_valid = (self.last_year_train) + int(ttl_years * ratios[1])
            self.last_year_test = self.df_mortality['year'].max()        
        # If not a ratio
        else:
            min_year = self.df_mortality['year'].min()
            max_year = self.df_mortality['year'].max()
            # Ensure all dataset are used, also the boundaries of training, validation, and testing set are valid
            assert (ratios[2] == max_year) & (ratios[0] < ratios[1]) & (ratios[1] < ratios[2]) & (ratios[0] in range(min_year, max_year + 1))
            self.first_year_train = min_year
            self.last_year_train = ratios[0]
            self.last_year_valid = ratios[1]
            self.last_year_test = ratios[2]
    
        # Scaler 
        self.scaler = py_utils.Scaler(self.df_mortality, self.last_year_train)
        
    def setup(self, stage):
        if(self.is_label_scaled):
            label_transform = self.scaler.mamiya_transform
        else:
            label_transform = None
        self.train_ds = MortalityUcodLongDataset(df_mortality=self.df_mortality,
                                                 year_start=self.first_year_train,
                                                 year_end=self.last_year_train,
                                                 label_transform=label_transform,
                                                 categorical_features=self.categorical_features,
                                                 mortality_features=self.mortality_features)
        self.val_ds = MortalityUcodLongDataset(df_mortality=self.df_mortality,
                                               year_start=self.last_year_train+1,
                                               year_end=self.last_year_valid,
                                               label_transform=label_transform,
                                               categorical_features=self.categorical_features,
                                               mortality_features=self.mortality_features)
        self.test_ds = MortalityUcodLongDataset(df_mortality=self.df_mortality,
                                               year_start=self.last_year_valid+1,
                                               year_end=self.last_year_test,
                                               label_transform=label_transform,
                                               categorical_features=self.categorical_features,
                                               mortality_features=self.mortality_features)
        torch.manual_seed(self.seed)
        if(self.is_bootstrap):
            self.sampler = RandomSampler(data_source=self.train_ds, replacement=True,
                                         num_samples=self.batch_size * self.steps)        
        else:
            self.sampler = None
    
    def train_dataloader(self):
        if(self.is_bootstrap):
            return DataLoader(
                self.train_ds,
                batch_size=self.batch_size,
                sampler=self.sampler,
                num_workers = self.num_workers
            )
        else:
            return DataLoader(
                self.train_ds,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers = self.num_workers
            )
        
    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers = self.num_workers)
        
    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers = self.num_workers)

