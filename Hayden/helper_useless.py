#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 10:34:02 2021

@author: haydenlw4
"""

import numpy as np

def convert_cat_ordinal_vars_to_num(df, columns, mapping_dict):
    for column in columns:
        df[column] = df[column].map(mapping_dict).fillna(value=0)
    return df

def convert_nas_to_none(df, columns):
    for column in columns:
        df[column] = df[column].fillna(value = 'None')
    return df

def convert_all_cat_ordinals(df):
    cat_ordinal_features = [
        'GarageQual','GarageCond',
        'FireplaceQu',
        'KitchenQual',
        'ExterQual','ExterCond',
        'BsmtQual','BsmtCond',
        'HeatingQC'
        ]
    cat_ordinal_dict = {'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}
    df = convert_cat_ordinal_vars_to_num(df,
                                                cat_ordinal_features,
                                                cat_ordinal_dict)
    # BsmtExposure
    cat_ordinal_features = [
        'BsmtExposure'
    ]
    cat_ordinal_dict = {'No':1,'Mn':2,'Av':3,'Gd':4}
    df = convert_cat_ordinal_vars_to_num(df,
                                                cat_ordinal_features,
                                                cat_ordinal_dict)
    #(Niki) Updated values in mapping dictionary
    #(HW) Why are the mapping numbers spaced now? Is that in documentation?
    # Functional
    cat_ordinal_features = [
        'Functional'
    ]
    cat_ordinal_dict = {'Sal':1,'Sev':2,'Maj2':3,'Maj1':4,
                        'Mod':6,'Min2':8,'Min1':9,'Typ':10}
    df = convert_cat_ordinal_vars_to_num(df,
                                                cat_ordinal_features,
                                                cat_ordinal_dict)
    # PoolQC
    cat_ordinal_features = [
        'PoolQC'
    ]
    cat_ordinal_dict = {'Fa':1,'TA':2,'Gd':3,'Ex':4}
    df = convert_cat_ordinal_vars_to_num(df,
                                                    cat_ordinal_features,
                                                    cat_ordinal_dict)
    # Fence
    cat_ordinal_features = [
        'Fence'
    ]
    cat_ordinal_dict = {'MnWw':1,'GdWo':2,'MnPrv':3,'GdPrv':4}
    df = convert_cat_ordinal_vars_to_num(df,
                                                    cat_ordinal_features,
                                                    cat_ordinal_dict)
    # (JH) Garage Finish
    cat_ordinal_features = [
        'GarageFinish'
    ]
    cat_ordinal_dict = {'Unf':1,'RFn':2,'Fin':3}
    df = convert_cat_ordinal_vars_to_num(df,
                                                    cat_ordinal_features,
                                                    cat_ordinal_dict)

    # (JH) Alley
    cat_ordinal_features = [
        'Alley'
    ]
    cat_ordinal_dict = {'Grvl':1,'Pave':2}
    df = convert_cat_ordinal_vars_to_num(df,
                                                    cat_ordinal_features,
                                                    cat_ordinal_dict)
    
    # (JH) PavedDrive
    cat_ordinal_features = [
        'PavedDrive'
    ]
    cat_ordinal_dict = {'P':1,'Y':2}
    df = convert_cat_ordinal_vars_to_num(df,
                                                    cat_ordinal_features,
                                                    cat_ordinal_dict)
    # (Niki) Utilities
    cat_ordinal_features = [
        'Utilities'
    ]
    cat_ordinal_dict = {'ELO':1,'NoSeWa':2,'NoSewr':3,'AllPub':4}
    df = convert_cat_ordinal_vars_to_num(df,
                                                    cat_ordinal_features,
                                                    cat_ordinal_dict)
    return df

def convert_all_well_defined_nas(df):
    # NAs that have meaning based on data dicitonary.
    # nas are "None" categorical value
    na_none_features = ['MiscFeature','BsmtFinType1','BsmtFinType2',
                        'GarageType', 'MasVnrType']
    for na_none_feature in na_none_features:
        df[na_none_feature] = df[na_none_feature].fillna(value = 'None')
        
    # nas are 0 numerical value
    # (AA) Changed na_none_feature to na_zero_feature for readability. 
    # (AA) Added 'BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF' to the list.
    na_zero_features = ['BsmtFullBath','BsmtHalfBath','GarageFinish','Alley', 
                        'BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF']
    for na_zero_feature in na_zero_features:
        df[na_zero_feature] = df[na_zero_feature].fillna(value = 0)
    # (Niki) Replace null value for Electrical with most common value 'SBrkr'
    df['Electrical'] = df['Electrical'].fillna(value = 'SBrkr')
    return df

def data_processing_wrapper(df):
    df = convert_all_cat_ordinals(df)
    df = convert_all_well_defined_nas(df)
    # dropping columns that no one tried to salvage or impute with.
    drop_now_but_look_at_later = ['MasVnrArea','GarageYrBlt']
    df.drop(drop_now_but_look_at_later, axis=1,inplace = True)
    for column in drop_now_but_look_at_later:
        print(f'Dropping {column} from our data set.')
    # alex's imputing of lot frontage.
    # (AA) LotFrontage imputed as (coefficient from dict) * sqrt(LotArea)
    LotFrontage_dict = {'1Fam':0.7139, 
                        'TwnhsE':0.5849, 
                        'Twnhs':0.5227, 
                        'Duplex':0.7725, 
                        '2fmCon':0.6922}
    df.loc[df['LotFrontage'].isna(), 
            'LotFrontage'] = df.loc[df['LotFrontage'].isna(), :].apply(
        lambda x: LotFrontage_dict[x['BldgType']]*np.sqrt(x['LotArea']), 
        axis=1)
    # special garagetypes that should be None.
    df.loc[df['PID'] == 903426160,'GarageType'] = 'None'
    df.loc[df['PID'] == 910201180,'GarageType'] = 'None'
    return df

