#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 08:55:00 2021

@author: Alex Austin, Hayden Warren
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

# Stratified split authored by Alex A, generalized by Hayden W.
def stratified_split(df,stratified_col,
                     n_splits = 1, 
                     test_size = 0.25, 
                     random_state = 42):
    '''
    Completes a stratified split on unique rows of a dataframe 
    on specified column.
    
    Parameters
    ----------
    df : pandas Dataframe
        Dataframe to be split.
    stratified_col : string
        Column that stratification will be done on.
    n_splits : integer, optional
        number of splits. The default is 1.
    test_size : float, optional
        proportion in df_test. The default is 0.25.
    random_state : integer, optional
        The default is 42.

    Returns
    -------
    df_train : pandas Dataframe
        Dataframe after split. 
        Will have 1 - test_size proportion of rows.
    df_test : pandas Dataframe
        Dataframe after split. 
        Will have test_size proportion of rows.

    '''
    # Ensure clean index and remove duplicates
    df.reset_index(inplace=True,drop=True)
    df.drop_duplicates(inplace=True)
    # Check for stratifications with single element. 
    value_count = df[stratified_col].value_counts(ascending=True).reset_index()
    single_strat = list(value_count[value_count[stratified_col]==1]['index'])
    # Remove single strats from split. They will be added later.
    df_non_single_strat = df[~df[stratified_col].isin(single_strat)
                             ].reset_index(drop=True)
    # Perform stratified split.
    split_specs = StratifiedShuffleSplit(n_splits = n_splits, 
                                      test_size = test_size, 
                                      random_state = random_state)
    split = split_specs.split(df_non_single_strat, 
                              df_non_single_strat[stratified_col])
    for train_index, test_index in split:
        df_train = df_non_single_strat.loc[train_index].reset_index(drop=True)
        df_test = df_non_single_strat.loc[test_index].reset_index(drop=True)
    # Put single stratified splits into train.
    df_train = pd.concat([df_train, df[df[stratified_col].isin(single_strat)]
                          ]).reset_index(drop=True)
    return df_train, df_test

def convert_cat_ordinal_vars_to_num(df, columns, mapping_dict):
    for column in columns:
        df[column] = df[column].map(mapping_dict).fillna(value=0)
    return df

def convert_nas_to_none(df, columns):
    for column in columns:
        df[column] = df[column].fillna(value = 'None')
    return df

def geo_cords_imputing(df, imputing = 0):
    # if imputing dictionary is input then it will use that dictionary
    if imputing:
        geo_cords = ['latitude','longitude']
        for geo_cord in geo_cords:
            df.loc[df[geo_cord].isna(), 
                      geo_cord] = df.loc[df[geo_cord].isna(), :].apply(
                lambda x: imputing[geo_cord][x['Neighborhood']], axis=1
            )
        return df
    # if no imputing dictionary is input then it will make the imputing dict.
    else:
        nbhd_geo_dict = df.groupby('Neighborhood').agg({'longitude':'median',
                                                          'latitude':'median',}
                                                         ).to_dict()
        geo_cords = ['latitude','longitude']
        for geo_cord in geo_cords:
            df.loc[df[geo_cord].isna(), 
                      geo_cord] = df.loc[df[geo_cord].isna(), :].apply(
                lambda x: nbhd_geo_dict[geo_cord][x['Neighborhood']], axis=1
            )
        return df, nbhd_geo_dict


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