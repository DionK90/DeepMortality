import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from typing import List

import torch
import torch.nn.functional as F
import torch.optim
from PIL import __version__ as PILLOW_VERSION
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import RandomSampler
from torch import optim
from torch import nn

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything

import py_params
import py_utils

class CausalConv1d(torch.nn.Conv1d):
    """
    Note that in PyTorch, Conv1D input is defined as (batch_size, embedding_dimension, seq_length).     
    Moreover, PyTorch Conv1D does not have `causal` padding. 
    Therefore, a new class of Conv1D is defined based on SoheilZabihi post in (https://github.com/pytorch/pytorch/issues/1333).
    """
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride=1, dilation=1,
                 groups=1, bias=True):

        super(CausalConv1d, self).__init__( in_channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=0,
            dilation=dilation, groups=groups, bias=bias)

        self.__padding = (kernel_size - 1) * dilation

    def forward(self, input):
        return super(CausalConv1d, self).forward(F.pad(input, (self.__padding, 0)))

def get_output_shape(model, input_dim):
    return model(torch.rand(*(input_dim))).data.shape

class CnnEmbedding(pl.LightningModule):
    """_summary_

    Args:
        pl (_type_): _description_
    """
    def __init__(self, config,
                 cat_names, cat_embedding_input_dims, cat_embedding_output_dims,
                 label_inverse_transform:callable,
                 out_of_sample_horizon, is_multi_year_validation = True):
        """_summary_

        Args:
            config (_type_): a dictionary of hyperparameters for the network structure. The list of keys to be provided are:
                num_m_features: the number of channels (similar to feature dimension in multivariate time-series) for the 1D CNN
                kernel_number: the number of kernels (filters) for the 1D CNN
                kernel_size: the size of each kernel (filter) for the 1D CNN
                pool_size: the size of the pooling kernel for the convolution branch
                conv_drop_rate: the rate of the dropout layer for the convolution branch
                batch_size: the batch size, necessary to construct the concatenate layer
                seq_len: the total years of historical data considered, necessary to construct the concatenate layer
                combin_drop_rate: the rate of the dropout layer for the final dense layer
                last_activation: the activation function for the final dense layer
                lr: the learning rate
            cat_names: 
            cat_embedding_input_dims: 
            cat_embedding_output_dims:
            label_inverse_transform:  
            out_of_sample_horizon:
            is_multi_year_validation:
        """
        super(CnnEmbedding, self).__init__()
        self.save_hyperparameters()
        self.loss_fn = F.mse_loss
        self.config = config
        
        # Variables used in forecast        
        self.label_inverse_transform = label_inverse_transform
        self.out_of_sample_horizon = out_of_sample_horizon
        self.is_multi_year_validation = is_multi_year_validation

        # Create some activation functions supported
        activations = nn.ModuleDict([
            ['lrelu', nn.LeakyReLU()],
            ['rrelu', nn.RReLU()],
            ['relu', nn.ReLU()],
            ['softmax', nn.Softmax()],
            ['tanh', nn.Tanh()],
            ['gelu', nn.GELU()],
            ['linear', nn.Identity()]
        ])

        # Create the convolution branch
        self.conv_branch = nn.Sequential(
            CausalConv1d(in_channels=config['num_m_features'], 
                         out_channels=config['kernel_number'],
                         kernel_size=config['kernel_size']),
            # Normalized each channels in one mini-batch (for matrix of (N, C, L), computed over the channel dimension at each (N, L) slice)
            # Calculate mean and variance for each channel
            nn.BatchNorm1d(config['kernel_number']),            
            # Max pooling on 1 dimension (the last dimension, the sequence length / time steps)
            nn.MaxPool1d(kernel_size=config['pool_size']),            
            # Dropout layer
            nn.Dropout(config['conv_drop_rate']),
            # Flatten layer
            nn.Flatten()
        )
        # Calculate the output dimension for the last linear layout
        expected_input_shape = (config['batch_size'], config['num_m_features'], config['seq_len'])
        conv_branch_out_dim = get_output_shape(self.conv_branch, expected_input_shape)        
        
        # Create embedding layer for each given categorical variables
        self.cat_names = cat_names
        self.embeddings = nn.ModuleList([nn.Sequential(nn.Embedding(cat_embedding_input_dims[i]*5, 
                                                               cat_embedding_output_dims[i]),
                                                       nn.Flatten()) for i in range(0, len(cat_names))])
        ttl_embedding_output_dims = sum(cat_embedding_output_dims)
        
        # The last dense layer
        self.fc_combined = nn.Sequential(
            nn.Dropout(config['combin_drop_rate']),
            nn.Linear(conv_branch_out_dim[1] + ttl_embedding_output_dims, config['num_m_features']),
            activations[config['last_activation']]
        )
        
    def forward(self, mortalityUcodWindowedDict):
        # Convolution branch
        conv_branch_out = self.conv_branch(mortalityUcodWindowedDict[py_params.FEATURES_M_NAME])

        # Categorical variables (embedding) branch
        embeddings_out = []
        for i, cat_name in enumerate(self.cat_names):
            embeddings_out.append(self.embeddings[i](mortalityUcodWindowedDict[cat_name]))
        
        # Combined the two inputs
        combined_info = torch.cat((conv_branch_out, *embeddings_out), dim=1)
        
        # Final layer
        final_out = self.fc_combined(combined_info)

        return final_out
    
    def training_step(self, batch, batch_idx):
        # Behind the scene of training_step        
        # for batch_idx, batch in enumerate(train_loader):
        #     loss = autoencoder.training_step(batch, batch_idx)
        y_hat = self._one_year_step(batch, batch_idx)        
        loss = self.loss_fn(y_hat, batch[py_params.FEATURES_LABEL_NAME][:,:,0])
        self.log('train_mse_loss', loss)
        return {'loss': loss, 'scores': y_hat}
    
    def validation_step(self, batch, batch_idx):
        if(self.is_multi_year_validation):
            y_hat = self._multi_year_step(batch, batch_idx)
            loss = self.loss_fn(y_hat, batch[py_params.FEATURES_LABEL_NAME])
            self.log('val_mse_loss', loss)
            return {'loss': loss, 'scores': y_hat}
        else:
            y_hat = self._one_year_step(batch, batch_idx)        
            loss = self.loss_fn(y_hat, batch[py_params.FEATURES_LABEL_NAME][:,:,0])
            self.log('val_mse_loss', loss)
            return {'loss': loss, 'scores': y_hat}
    
    def test_step(self, batch, batch_idx):        
        if(self.is_multi_year_validation):
            y_hat = self._multi_year_step(batch, batch_idx)
            loss = self.loss_fn(y_hat, batch[py_params.FEATURES_LABEL_NAME])
            self.log('test_mse_loss', loss)
            return {'loss': loss, 'scores': y_hat}
        else:
            y_hat = self._one_year_step(batch, batch_idx)        
            loss = self.loss_fn(y_hat, batch[py_params.FEATURES_LABEL_NAME][:,:,0])
            self.log('test_mse_loss', loss)
            return {'loss': loss, 'scores': y_hat} 
               
    def predict_step(self, batch, batch_idx):
        if(self.is_multi_year_validation):
            y_hat = self._multi_year_step(batch, batch_idx)
            return y_hat
        else:
            y_hat = self._one_year_step(batch, batch_idx)
            return y_hat
        
    def _multi_year_step(self, batch, batch_idx):        
        """
        A function used to do multi-year predict (forecast) for each datum.
        The datum need to be formatted in a batch of size 1.
        This function will return the multi-year predictions according to the forecast horizon given to the model
        """                        
        self.eval()
        forecasts = []
        with torch.no_grad():
            for i in range(self.out_of_sample_horizon):
                # Make batch prediction
                curr_predictions = self(batch)
                
                # Store the original (may be scaled) predictions
                forecasts.append(curr_predictions.clone().detach())
                
                # Transform back using scaler different scaler for each data in the batch
                if(py_utils.Scaler.is_mamiya(self.label_inverse_transform)):
                    for i in range(batch['sex'].shape[0]):              
                        curr_predictions[i] = torch.tensor(self.label_inverse_transform(sex = batch['sex'][i].item(), 
                                                                                   cause = batch['cause'][i].item(),
                                                                                   data = curr_predictions[None, i].float().numpy()))            
                
                # Update the mortality features of each data in the batch
                batch[py_params.FEATURES_M_NAME] = torch.cat((batch[py_params.FEATURES_M_NAME][:,:,1:], curr_predictions[:,:,None]), dim=-1)
            
            # Check whether the constructed final predictions are in the right format
            forecasts = torch.stack(forecasts, dim=-1)            
        
        return forecasts

    def _one_year_step(self, batch, batch_idx):        
        y_hat = self.forward(batch)    
        return y_hat
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.config['lr'])

class RwBlock(nn.Module):
    def __init__(self, config,
                 cat_names, cat_embedding_input_dims, cat_embedding_output_dims,
                 is_skip):
        super(Fcnn, self).__init__()
        self.save_hyperparameters()
        self.loss_fn = F.mse_loss
        self.config = config        

        # Create some activation functions supported
        activations = nn.ModuleDict([
            ['lrelu', nn.LeakyReLU()],
            ['rrelu', nn.RReLU()],
            ['relu', nn.ReLU()],
            ['softmax', nn.Softmax()],
            ['tanh', nn.Tanh()],
            ['gelu', nn.GELU()],
            ['linear', nn.Identity()]
        ])
        
        # Create embedding layer for each given categorical variables in order
        self.cat_names = cat_names
        self.embeddings = nn.ModuleList([nn.Sequential(nn.Embedding(cat_embedding_input_dims[i], 
                                                               cat_embedding_output_dims[i]),
                                                       nn.Flatten()) for i in range(0, len(cat_names))])
        ttl_embedding_output_dims = sum(cat_embedding_output_dims)
                        
        # The first dense layer that receives input from all embeddings and the numerical variable `year`
        self.feature_dimension = ttl_embedding_output_dims + 1
        self.feature_layer_1 = nn.Sequential(nn.Linear(self.feature_dimension, self.config['dense_neuron_0']),
                                     activations[self.config['dense_activation_0']],
                                     nn.BatchNorm1d(self.config['dense_neuron_0']),
                                     nn.Dropout(self.config['dense_drop_rate_0']))
        
        # The rest of the dense layer
        self.feature_layers = nn.ModuleList()
        for i in range(1, self.config['num_layers']):
            self.feature_layers.append(nn.Sequential(nn.Linear(config[f'dense_neuron_{i-1}'], config[f'dense_neuron_{i}']),
                                                     activations[config[f'dense_activation_{i}']],
                                                     nn.BatchNorm1d(config[f'dense_neuron_{i}']),
                                                     nn.Dropout(config[f'dense_drop_rate_{i}'])))
        
        # Final dense layer (with skip connections and concatenation if specified)
        self.is_skip = is_skip
        last_feature_neuron = config['num_layers']-1
        if(is_skip):
            final_dimension = self.feature_dimension + config[f'dense_neuron_{last_feature_neuron}']
        else:
            final_dimension = config[f'dense_neuron_{last_feature_neuron}']
        self.layer_final = nn.Sequential(nn.Linear(final_dimension, config['final_neuron']),
                                         activations[config[f'final_activation']],
                                         nn.BatchNorm1d(config[f'final_neuron']),
                                         nn.Dropout(config[f'final_drop_rate']),
                                         nn.Linear(config['final_neuron'], 1),
                                         activations[config[f'output_activation']]
                                         )
                
    def forward(self, mortalityUcodLongDict:dict):
        # Categorical variables (embedding) branch (age, gender, cause, etc.) in the order given in the initialization
        embeddings_out = []
        for i, cat_name in enumerate(self.cat_names):
            embeddings_out.append(self.embeddings[i](mortalityUcodLongDict[cat_name]))

        features = torch.cat((mortalityUcodLongDict['year'], *embeddings_out), dim=1)
        
        # Input all features to the first dense layer
        x = self.feature_layer_1(features)
        
        # Then to all other dense layers
        for feature_layer in self.feature_layers:
            x = feature_layer(x)
        
        # Combine processed feature with the original featuers (skip connection)
        if(self.is_skip):
            combined_info = torch.cat((x, features), dim=1)
        else:
            combined_info = x
        
        return self.layer_final(combined_info)
      
class Fcnn(pl.LightningModule):
    """
    A fully connected neural network based on the paper by Richman and Wüthrich (A neural network extension of the Lee-Carter model to
    multiple populations). The model is called DEEP6 is Listing 2 of the paper.
    
    The model consists of: 
    1. embedding layer for each categorical variable
    2. fully-connected layer, with inputs concatenated from all the embeddings and the year as a numerical input
    3. a skip-connection to the last dense layer (the original features are concatenated with the processed features from the dense layer)
    """
    def __init__(self, config,
                 cat_names, cat_embedding_input_dims, cat_embedding_output_dims):
        """_summary_

        Args:
            config (_type_): a dictionary of hyperparameters for the network structure. The list of keys to be provided are:
                dense_neuron_1: the total number of neuron on the first layer (the first layer to process the features from embeddings and the numerical feature 'year')
                dense_activation_1: the activation layer of the first dense layer
                dense_drop_rate_1: the dropout rate for the first dense layer
                num_layers: the number of hidden layers to process the features (including the first layer)
                dense_neuron_i: the total number of neuron on the i-th dense layer
                dense_activation_i: the activation layer of the i-th dense layer
                dense_drop_rate_i: the dropout rate for the i-th dense layer
                final_neuron: the total number of neuron on the concatenating layer who accepts processed features and the original features (through a skip connection)
                final_activation: the activation layer of the concatenating layer
                final_drop_rate:  the dropout rate for the concatenating layer
                output_activation: the activation function for the concatenating layer
                lr: the learning rate
            cat_names: 
            cat_embedding_input_dims: 
            cat_embedding_output_dims:
            label_inverse_transform:  
        """
        super(Fcnn, self).__init__()
        self.save_hyperparameters()
        self.loss_fn = F.mse_loss
        self.config = config        

        # Create some activation functions supported
        activations = nn.ModuleDict([
            ['lrelu', nn.LeakyReLU()],
            ['rrelu', nn.RReLU()],
            ['relu', nn.ReLU()],
            ['softmax', nn.Softmax()],
            ['tanh', nn.Tanh()],
            ['gelu', nn.GELU()],
            ['linear', nn.Identity()]
        ])
        
        # Create embedding layer for each given categorical variables in order
        self.cat_names = cat_names
        self.embeddings = nn.ModuleList([nn.Sequential(nn.Embedding(cat_embedding_input_dims[i], 
                                                               cat_embedding_output_dims[i]),
                                                       nn.Flatten()) for i in range(0, len(cat_names))])
        ttl_embedding_output_dims = sum(cat_embedding_output_dims)
                        
        # The first dense layer that receives input from all embeddings and the numerical variable `year`
        self.feature_dimension = ttl_embedding_output_dims + 1
        self.feature_layer_1 = nn.Sequential(nn.Linear(self.feature_dimension, self.config['dense_neuron_0']),
                                     activations[self.config['dense_activation_0']],
                                     nn.BatchNorm1d(self.config['dense_neuron_0']),
                                     nn.Dropout(self.config['dense_drop_rate_0']))
        
        # The rest of the dense layer
        self.feature_layers = nn.ModuleList()
        for i in range(1, self.config['num_layers']):
            self.feature_layers.append(nn.Sequential(nn.Linear(config[f'dense_neuron_{i-1}'], config[f'dense_neuron_{i}']),
                                                     activations[config[f'dense_activation_{i}']],
                                                     nn.BatchNorm1d(config[f'dense_neuron_{i}']),
                                                     nn.Dropout(config[f'dense_drop_rate_{i}'])))
        
        # Final dense layer with skip connections and concatenation
        last_feature_neuron = config['num_layers']-1
        final_skip_dimension = self.feature_dimension + config[f'dense_neuron_{last_feature_neuron}']
        self.layer_final = nn.Sequential(nn.Linear(final_skip_dimension, config['final_neuron']),
                                         activations[config[f'final_activation']],
                                         nn.BatchNorm1d(config[f'final_neuron']),
                                         nn.Dropout(config[f'final_drop_rate']),
                                         nn.Linear(config['final_neuron'], 1),
                                         activations[config[f'output_activation']]
                                         )
        
        
    def forward(self, mortalityUcodLongDict:dict):
        # Categorical variables (embedding) branch (age, gender, cause, etc.) in the order given in the initialization
        embeddings_out = []
        for i, cat_name in enumerate(self.cat_names):
            embeddings_out.append(self.embeddings[i](mortalityUcodLongDict[cat_name]))

        features = torch.cat((mortalityUcodLongDict['year'], *embeddings_out), dim=1)
        
        # Input all features to the first dense layer
        x = self.feature_layer_1(features)
        
        # Then to all other dense layers
        for feature_layer in self.feature_layers:
            x = feature_layer(x)
        
        # Combine processed feature with the original featuers (skip connection)
        combined_info = torch.cat((x, features), dim=1)
        return self.layer_final(combined_info)
    
    def training_step(self, batch, batch_idx):
        step = self._common_step(batch, batch_idx)        
        self.log('train_mse_loss', step['loss'])
        return {'loss': step['loss'], 'scores': step['scores']}
    
    def validation_step(self, batch, batch_idx):        
        step = self._common_step(batch, batch_idx)        
        self.log('val_mse_loss', step['loss'])
        return {'loss': step['loss'], 'scores': step['scores']}
    
    def test_step(self, batch, batch_idx):        
        step = self._common_step(batch, batch_idx)        
        self.log('test_mse_loss', step['loss'])
        return {'loss': step['loss'], 'scores': step['scores']}
    
    def _common_step(self, batch, batch_idx):
        y_hat = self.forward(batch)        
        loss = self.loss_fn(y_hat, batch[py_params.FEATURES_LABEL_NAME])        
        return {'loss': loss, 'scores': y_hat}   

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config['lr'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_mse_loss"}
        # return 

class Lcnn(pl.LightningModule):
    """
    Another Neural Network architecture by Richman and Wüthrich (A neural network extension of the Lee-Carter model to
    multiple populations). This architecture is built to mimic the original LC model. The model can be seen in Listing 1 of the paper.
    This model can only be used to get an estimation for the three parameters in the LC model (ax, bx, and kt), and cannot be used to forecast mortality.
    Forecasting is done using the classical LC model (forecasting using Time Series method)   
    
    The model consists of: 
    1. One embedding layer for year (kt) and Two embedding layers for age (ax and bx)
    2. Multiply kt with bx, then add the resulting vector to ax
    3. One untrainable dense layer to c
    
    Note that the only trainable parameters are the weights in the three embedding layers.

    Args:
        pl (_type_): _description_
    """
    def __init__(self, nb_years:int, nb_ages:int):
        """
        Args:
            nb_years (int): the total number of years considered by the model
            nb_ages (int): the total number of ages considered by the model
        """
        super(Lcnn, self).__init__()
        
        # The total number of years and ages used
        self.nb_years = nb_years
        self.nb_ages = nb_ages
        
        # 
        self.year_embed = nn.Embedding(nb_years, 1)
        self.age_embed_1 = nn.Embedding(nb_ages, 1)
        self.age_embed_2 = nn.Embedding(nb_ages, 1)
        
        self.dense = nn.Linear(1, 1, bias=False)
        nn.init.ones_(self.dense.weight)
        self.dense.weight.requires_grad = False 
        
    def forward(self, x):
        year = self.year_embed(x['year']).squeeze(1)
        age_1 = self.age_embed_1(x['age']).squeeze(1)
        age_2 = self.age_embed_2(x['age']).squeeze(1)
        
        year_effect = year * age_2
        combined = age_1 + year_effect
        
        output = self.dense(combined)
        output = torch.exp(output)
        
        return output
    
    def training_step(self, batch, batch_idx):
        y_hat = self.forward(batch)
        loss = F.mse_loss(y_hat, batch['Label'])
        self.log('mse_train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
