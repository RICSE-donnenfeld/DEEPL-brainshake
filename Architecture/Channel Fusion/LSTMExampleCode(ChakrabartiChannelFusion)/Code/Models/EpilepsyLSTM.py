#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 13 16:12:08 2022

@author: Guillermo Torres
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.ModelWeightsInit import *


class EpilepsyLSTM(nn.Module):
    """
    Implementation:
        A channel independent generalized seizure detection method for pediatric epileptic seizures
        batch_size 600
        epochs 1000
        lr = 1e-4
        optmizer Adam
    """
    def __init__(self, inputmodule_params,net_params,outmodule_params):
        super().__init__()

        print('Running class: ', self.__class__.__name__)
        
        ### NETWORK PARAMETERS
        n_nodes=inputmodule_params['n_nodes']
    
        Lstacks=net_params['Lstacks']
        dropout=net_params['dropout'] 
        hidden_size=net_params['hidden_size']
       
        n_classes=outmodule_params['n_classes']
        hd=outmodule_params['hd']
        
        self.inputmodule_params=inputmodule_params
        self.net_params=net_params
        self.outmodule_params=outmodule_params
        
        ### NETWORK ARCHITECTURE
        # IF batch_first THEN (batch, timesteps, features), ELSE (timesteps, batch, features)
        self.lstm = nn.LSTM(input_size=n_nodes, # the number of expected features (out of convs)
                                       hidden_size= hidden_size, # the number of features in the hidden state h
                                       num_layers= Lstacks, # number of stacked lstms 
                                       batch_first = True,
                                       bidirectional = False,
                                       dropout=dropout)

        self.fc = nn.Sequential(nn.Linear(hidden_size, hd),
                                nn.ReLU(),
                                nn.Linear(hd, n_classes)
                                ) 

    
    def init_weights(self):
         init_weights_xavier_normal(self)
        
    def forward(self, x):
        
        ## Reshape input
        # input [batch, features (=n_nodes), sequence_length (T)] ([N, 21, 640])
        x = x.permute(0, 2, 1) # lstm  [batch, sequence_length, features]
        
        ## LSTM Processing
        out, (hn, cn) = self.lstm(x)
        # out is [batch, sequence_length, hidden_size] for last stack output
        # hn and cn are [1, batch, hidden_size]
        out = out[:, -1, :] # hT state of lenght hidden_size

        ## Output Classification (Class Probabilities)
        x = self.fc(out)

        return x