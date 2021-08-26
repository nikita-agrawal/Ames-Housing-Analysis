#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 15:10:24 2021

@author: Julie Hemily, Hayden Warren
"""
###################FUTURE WORK##################################
# Fix the drop_now_but_look_at_later features.
################################################################

# Loading packages
import pandas as pd
import numpy as np
import helper # my function that i added

# loading and splitting data
housing = pd.read_csv('Ames_Housing_Price_Data.csv', index_col=0)

train, test = helper.stratified_split(housing,'Neighborhood')

# converting all similar mappings together
# most popular mapping
cat_ordinal_features = [
    'GarageQual','GarageCond',
    'FireplaceQu',
    'KitchenQual',
    'ExterQual','ExterCond',
    'BsmtQual','BsmtCond',
    'HeatingQC'
    ]
cat_ordinal_dict = {'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}
train = helper.convert_cat_ordinal_vars_to_num(train,
                                               cat_ordinal_features,
                                               cat_ordinal_dict)
# now just unique mappings
# BsmtExposure
cat_ordinal_features = [
    'BsmtExposure'
]
cat_ordinal_dict = {'No':1,'Mn':2,'Av':3,'Gd':4}
train = helper.convert_cat_ordinal_vars_to_num(train,
                                               cat_ordinal_features,
                                               cat_ordinal_dict)
# Functional
cat_ordinal_features = [
    'Functional'
]
cat_ordinal_dict = {'Sal':1,'Sev':2,'Maj2':3,'Maj1':4,
                    'Mod':5,'Min2':6,'Min1':7,'Typ':8}
train = helper.convert_cat_ordinal_vars_to_num(train,
                                               cat_ordinal_features,
                                               cat_ordinal_dict)
# PoolQC
cat_ordinal_features = [
    'PoolQC'
]
cat_ordinal_dict = {'Fa':1,'TA':2,'Gd':3,'Ex':4}
train = helper.convert_cat_ordinal_vars_to_num(train,
                                               cat_ordinal_features,
                                               cat_ordinal_dict)
# (JH) Garage Finish
cat_ordinal_features = [
    'GarageFinish'
]
cat_ordinal_dict = {'Unf':1,'RFn':2,'Fin':3}
train = helper.convert_cat_ordinal_vars_to_num(train,
                                               cat_ordinal_features,
                                               cat_ordinal_dict)

# (JH) Alley
cat_ordinal_features = [
    'Alley'
]
cat_ordinal_dict = {'Grvl':1,'Pave':2}
train = helper.convert_cat_ordinal_vars_to_num(train,
                                               cat_ordinal_features,
                                               cat_ordinal_dict)

# (JH) PavedDrive
cat_ordinal_features = [
    'Alley'
]
cat_ordinal_dict = {'P':1,'Y':2}
train = helper.convert_cat_ordinal_vars_to_num(train,
                                               cat_ordinal_features,
                                               cat_ordinal_dict)

############################################################
# weirdest nas. lot frontage. probably worth removing
# not dealing with them out of expediance. 
drop_now_but_look_at_later = ['LotFrontage','MasVnrArea','GarageYrBlt']
train.drop(drop_now_but_look_at_later, axis=1,inplace = True)

# NAs that have meaning based on data dicitonary.
# nas are "None" categorical value
na_none_features = ['MiscFeature','BsmtFinType1','BsmtFinType2',
                   'GarageType', 'MasVnrType']
for na_none_feature in na_none_features:
    train[na_none_feature] = train[na_none_feature].fillna(value = 'None')
# nas are 0 numerical value
na_zero_features = ['BsmtFullBath','BsmtHalfBath','GarageFinish','Alley']
for na_none_feature in na_zero_features:
    train[na_none_feature] = train[na_none_feature].fillna(value = 0)
    
############################################################
# second group of features that have problems that I need help solving.
cols_na = train.loc[:,train.isna().any(axis=0)].columns.to_list()
cols_na

#############################################################
# (JH) Julie's specific additions
# (HW) Consider making this more generalizable
# 1. Replacing two values of GarageType to None

train.loc[train['PID'] == 903426160,'GarageType'] = 'None'
train.loc[train['PID'] == 910201180,'GarageType'] = 'None'