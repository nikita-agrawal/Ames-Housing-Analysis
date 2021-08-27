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
# (AA) Modified to exclude SaleCondition other than Normal and Partial.
# (Niki) Typo hous_df - Changed to housing. 
housing = pd.read_csv('Ames_Housing_Price_Data.csv')
housing.drop('Unnamed: 0', axis=1, inplace=True)
housing = housing[(housing['SaleCondition'] == 'Normal') | (housing['SaleCondition'] == 'Partial')].reset_index(drop=True)

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
#(Niki) Updated values in mapping dictionary
# Functional
cat_ordinal_features = [
    'Functional'
]
cat_ordinal_dict = {'Sal':1,'Sev':2,'Maj2':3,'Maj1':4,
                    'Mod':6,'Min2':8,'Min1':9,'Typ':10}
train = helper.convert_cat_ordinal_vars_to_num(train,
                                               cat_ordinal_features,
                                               cat_ordinal_dict)
# PoolQC ((Niki) Suggest Dropping in future and replace by _has_Pool binary feature)
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
# (AA) LotFrontage imputed as (coefficient from dict) * sqrt(LotArea)
LotFrontage_dict = {'1Fam':0.7139, 'TwnhsE':0.5849, 'Twnhs':0.5227, 'Duplex':0.7725, '2fmCon':0.6922}
train.loc[train['LotFrontage'].isna(), 'LotFrontage'] = train.loc[train['LotFrontage'].isna(), :].apply(
    lambda x: LotFrontage_dict[x['BldgType']]*np.sqrt(x['LotArea']), axis=1
)

# (Niki) Utilities
cat_ordinal_features = [
    'Utilities'
]
cat_ordinal_dict = {'ELO':1,'NoSeWa':2,'NoSewr':3,'AllPub':4}
train = helper.convert_cat_ordinal_vars_to_num(train,
                                               cat_ordinal_features,
                                               cat_ordinal_dict)


############################################################
# weirdest nas. lot frontage. probably worth removing
# not dealing with them out of expediance.
# (AA) Removing LotFrontage here because imputed above.
drop_now_but_look_at_later = ['MasVnrArea','GarageYrBlt']
train.drop(drop_now_but_look_at_later, axis=1,inplace = True)

# NAs that have meaning based on data dicitonary.
# nas are "None" categorical value
na_none_features = ['MiscFeature','BsmtFinType1','BsmtFinType2',
                   'GarageType', 'MasVnrType']
for na_none_feature in na_none_features:
    train[na_none_feature] = train[na_none_feature].fillna(value = 'None')

# nas are 0 numerical value
# (AA) Changed na_none_feature to na_zero_feature for readability. 
# (AA) Added 'BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF' to the list.
na_zero_features = ['BsmtFullBath','BsmtHalfBath','GarageFinish','Alley', 
                   'BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF']
for na_zero_feature in na_zero_features:
    train[na_zero_feature] = train[na_zero_feature].fillna(value = 0)
    
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

#####
# (Niki) Replace 'Pool Area' feature with binary '_has_Pool'. Drop PoolQC
#train['_has_Pool'] = [1 if row != 0 else 0 for row in train['PoolArea']]

# (Niki) Replace null value for Electrical with most common value 'SBrkr'
train['Electrical'] = train['Electrical'].fillna(value = 'SBrkr')
