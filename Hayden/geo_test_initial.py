#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 14:15:40 2021

@author: Hayden, Julie, Alex, Niki
"""

# Loading packages
import pandas as pd
import helper

# loading and splitting data
housing = pd.read_csv('ames_housing_latlong.csv', index_col = 0,
                      low_memory=False)

housing.drop_duplicates(inplace=True)
housing.reset_index(drop=True, inplace=True)

housing = housing[(housing['SaleCondition'] == 'Normal') | 
                   (housing['SaleCondition'] == 'Partial')].reset_index(drop=True)

train, test = helper.stratified_split(housing,'Neighborhood')

train, mapping_dict = helper.geo_cords_imputing(train)
test = helper.geo_cords_imputing(test,mapping_dict)

train = helper.data_processing_wrapper(train)
test = helper.data_processing_wrapper(test)

train.to_csv('train_geo.csv')
test.to_csv('test_geo.csv')
