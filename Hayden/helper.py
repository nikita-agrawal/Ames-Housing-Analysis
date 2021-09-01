#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 08:55:00 2021

@author: Alex Austin, Hayden Warren, Niki Agrawal
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

# written by Niki
def impute_missing_vals(df):
    #1. Impute LotFrontage missing values with linear regression coefficients 
    # AA: LotFrontage imputed as (coefficient from dict) * sqrt(LotArea)
    LotFrontage_dict = {'1Fam':0.7139, 'TwnhsE':0.5849, 'Twnhs':0.5227, 'Duplex':0.7725, '2fmCon':0.6922}
    df.loc[df['LotFrontage'].isna(), 'LotFrontage'] = df.loc[df['LotFrontage'].isna(), :].apply(
        lambda x: LotFrontage_dict[x['BldgType']]*np.sqrt(x['LotArea']), axis=1)

    #2. All rows with MasVnrArea null values also have MasVnrType as null.\
    idx = df['MasVnrArea'].isna() & (df['MasVnrType'].isna())
    #Assume these properties do not have a veneer, so set MasVnrType as "None" and MasVnrArea as 0 
    df.loc[idx,'MasVnrArea'] = 0 
    df.loc[idx,'MasVnrType'] = "None" #motivated by the null value in test, is this data leakage?

    #3 & 4. BsmtFullBath & BsmtHalfBath nulls corresponds with No basement. Can impute with 0. 
    df.loc[df['BsmtFullBath'].isna() & (df['TotalBsmtSF']==0),'BsmtFullBath'] = 0
    df.loc[df['BsmtHalfBath'].isna() & (df['TotalBsmtSF']==0),'BsmtHalfBath'] = 0 

    #5. GarageYrBuilt - repalce missing year with YearBuilt for properties with non-zero garage area values  
    idx = df['GarageYrBlt'].isna() & (df['GarageArea']!=0.0)
    df.loc[idx,'GarageYrBlt'] = df.loc[idx,'YearBuilt']
    #The rest do not have garages so fill with 0, later convert to None 
    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(value=0)
    
    #6. Impute 'Electrical' null values with the most common type 'SBrKr' -> motivated by the null value in test, is this data leakage?
    ### trainX['Electrical'].mode() = SBrkr
    df.loc[df['Electrical'].isna(),'Electrical'] = 'SBrkr'
    
    #7. JH:Specific additions: Replacing two values of GarageType to None
    df.loc[df['PID'] == 903426160,'GarageType'] = 'None'
    df.loc[df['PID'] == 910201180,'GarageType'] = 'None'
    
    return df
# written by Niki
def convert_num_to_categorical(df,num_to_nominal_cat_feats=['GarageCars','MSSubClass','KitchenAbvGr','BedroomAbvGr','MoSold','YrSold']):
    #Features that were originally numeric but should be treated as nominal categories since there is no clear 
    #advantage from applying a rank:
    for feat in num_to_nominal_cat_feats:
        df[feat] = df[feat].astype(str)
    
    return df

# written by Niki
#According to data dictionary, NA translates to 'None' (No access, No basement etc.) for the following categories:
def replace_na_with_none(df):
    na_means_none_cols = ['Alley','BsmtQual','BsmtCond','BsmtFinType1','BsmtFinType2',
                 'BsmtExposure','FireplaceQu','GarageType','GarageFinish',
                 'GarageQual','GarageCond','PoolQC','Fence','MiscFeature']
    for col in na_means_none_cols:
        df[col] = df[col].fillna(value = 'None')
    return df

# written by Niki
def map_ordinal_cat(df):
    #Maps
    common_ranks_dict = {'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}
    replace_map = {
        'ExterQual': common_ranks_dict,
        'ExterCond': common_ranks_dict,
        'BsmtQual': common_ranks_dict,
        'BsmtCond': common_ranks_dict,
        'BsmtExposure': {'None':0,'No':1,'Mn':2,'Av':3,'Gd':4}, 
        'HeatingQC': common_ranks_dict,
        'KitchenQual': common_ranks_dict,
        'FireplaceQu': common_ranks_dict,
        'GarageFinish': {'None':0,'Unf':1,'RFn':2,'Fin':3},
        'GarageQual': common_ranks_dict,
        'GarageCond': common_ranks_dict,
        'PavedDrive': {'N':0,'P':1,'Y':2},
        'PoolQC': {'None':0,'Fa':1,'TA':2,'Gd':3,'Ex':4},
        'Alley': {'None':0,'Grvl':1,'Pave':2}
    }              
    #Replace strings with numbers 
    df.replace(replace_map, inplace=True)
    return df 

def data_processing_wrapper(df,
                            # should be able to change these 
                            # inputs for possible changes
                            # stratified_split_dict = 
                            ):
    df = df[(df['SaleCondition'] == 'Normal') | 
            (df['SaleCondition'] == 'Partial')].reset_index(drop=True)
    
    train_raw, test_raw = stratified_split(df,'Neighborhood')
    # train cleaning
    train_clean = impute_missing_vals(train_raw)
    num_to_cat_list =['GarageCars','MSSubClass','KitchenAbvGr',
                      'BedroomAbvGr','MoSold','YrSold']
    train_clean = convert_num_to_categorical(train_clean,num_to_cat_list) 
    train_clean = replace_na_with_none(train_clean)
    # this could be an option when calling the function
    train_clean = train_clean.drop(['PID'],axis='columns')
    train_clean = map_ordinal_cat(train_clean)
    # test cleaning
    test_clean = impute_missing_vals(test_raw)
    num_to_cat_list =['GarageCars','MSSubClass','KitchenAbvGr',
                      'BedroomAbvGr','MoSold','YrSold']
    test_clean = convert_num_to_categorical(test_clean,num_to_cat_list) 
    test_clean = replace_na_with_none(test_clean)
    # this could be an option when calling the function
    test_clean = test_clean.drop(['PID'],axis='columns')
    test_clean = map_ordinal_cat(test_clean)
    return train_clean, test_clean






    


    