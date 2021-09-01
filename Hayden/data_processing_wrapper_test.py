#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 10:46:44 2021

@author: haydenlw4
"""

import pandas as pd
import helper

housing = pd.read_csv('Ames_Housing_Price_Data.csv', index_col=0,low_memory = False)

# initial run.
train, test = helper.data_processing_wrapper(housing)

# you can specify the num_to_cat_list. the default list is :
#['GarageCars','MSSubClass','KitchenAbvGr','BedroomAbvGr','MoSold','YrSold']
# following example runs the same split but 
# doesn't convert 'GarageCars' or 'BedroomAbvGr' to categories
train1, test1 = helper.data_processing_wrapper(housing, 
                            num_to_cat_list = [
                                'MSSubClass',
                                'KitchenAbvGr',
                                'MoSold',
                                'YrSold']
                            )

# you can choose to remove PID or not. Default is to remove_PID = True.
# following example runs the same split but keeps index columns
train2, test2 = helper.data_processing_wrapper(housing, 
                                               remove_PID = False,
                                               )

# you can choose to change the stratified split inputs with a dictionary. 
# defaults are:{'stratified_col':'Neighborhood',
#                 'test_size':0.25, 
#                 'random_state':42}  
# following example runs the same split but but with a different random state 
# just changing 42 to 402                
train3, test3 = helper.data_processing_wrapper(housing, 
                            stratified_split_inputs = {
                                'stratified_col':'Neighborhood',
                                'test_size':0.25, 
                                'random_state':402
                                }                                               
                            )