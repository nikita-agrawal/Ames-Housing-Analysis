#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 08:55:00 2021

@author: Alex Austin, Hayden Warren, Niki Agrawal, Julie Hemily
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import seaborn as sns


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

#(HW)
def data_processing_wrapper(df, 
                            num_to_cat_list = [
                                'GarageCars',
                                'MSSubClass',
                                'KitchenAbvGr',
                                'BedroomAbvGr',
                                'MoSold',
                                'YrSold'
                                ],
                            remove_PID = True,
                            stratified_split_inputs = {
                                'stratified_col':'Neighborhood',
                                'test_size':0.25, 
                                'random_state':42
                                }
                            ):
    '''
    

    Parameters
    ----------
    df : pandas DataFrame
        Built to take the initial Ames_Housing_Price_Data.csv file without
        the index_col. Example line necessary to run code:
            df = pd.read_csv('Ames_Housing_Price_Data.csv', 
                             index_col=0,low_memory = False)

    num_to_cat_list : list, optional
        list of numerical features turned into categorical features. 
        The default is ['GarageCars','MSSubClass',KitchenAbvGr',
                        'BedroomAbvGr','MoSold','YrSold'].
    remove_PID : boolean, optional
        True if you want to remove PID col. False if you want to keep PID col.
        The default is True.
    stratified_split_inputs : dictionary, optional
        To change the inputes of the stratefied_split function. 
        The default is {                                
            'stratified_col':'Neighborhood',                                
            'test_size':0.25,                                
            'random_state':42                                
            }.

    Returns
    -------
    train_clean : pandas DataFrame
        train_clean.
    test_clean : pandas DataFrame
        test_clean.

    '''
    df = df[(df['SaleCondition'] == 'Normal') | 
            (df['SaleCondition'] == 'Partial')].reset_index(drop=True)
    
    # we can only do 1 split with this function.
    train_raw, test_raw = stratified_split(
        df,
        stratified_col = stratified_split_inputs['stratified_col'],
        n_splits = 1,
        test_size = stratified_split_inputs['test_size'],
        random_state = stratified_split_inputs['random_state'],
                                           )
    # train cleaning
    train_clean = impute_missing_vals(train_raw)
    train_clean = convert_num_to_categorical(train_clean,num_to_cat_list) 
    train_clean = replace_na_with_none(train_clean)
    if remove_PID:
        train_clean = train_clean.drop(['PID'],axis='columns')
    train_clean = map_ordinal_cat(train_clean)
    # test cleaning
    test_clean = impute_missing_vals(test_raw)
    test_clean = convert_num_to_categorical(test_clean,num_to_cat_list) 
    test_clean = replace_na_with_none(test_clean)
    if remove_PID:
        test_clean = test_clean.drop(['PID'],axis='columns')
    test_clean = map_ordinal_cat(test_clean)
    return train_clean, test_clean




######for graphing

def lasso_model_score(alpha, train_, test_, target, 
                             categorical_features,
                             drop_cols = ['SalePrice', 'TotalBsmtSF']):
    scaler = StandardScaler(with_mean=False)
    lasso = Lasso()
    
    
    X = train_.drop(drop_cols,axis=1)
    transformer = ColumnTransformer([("Cat", 
                                      OneHotEncoder(handle_unknown = 'ignore'), 
                                      categorical_features)], remainder='passthrough')
    X = transformer.fit_transform(X)
    X = scaler.fit_transform(X)
    y = np.log(train_[target])
    lasso = Lasso(alpha=alpha)
    selector = SelectFromModel(estimator=lasso)
    X = selector.fit_transform(X, y)
    
    lasso.fit(X,y)
    train_score = lasso.score(X,y)

    X_tst = test_.drop(drop_cols,axis=1)
    X_tst = transformer.transform(X_tst)
    X_tst = scaler.transform(X_tst)
    y_tst = np.log(test_[target])
    X_tst = selector.transform(X_tst)
    test_score = lasso.score(X_tst,y_tst)
    
    
    feat_names = transformer.get_feature_names()
    mask = selector.get_support()
    lasso_feats = [a for a, b in zip(feat_names, mask) if b]
    
    return train_score, test_score,lasso_feats
    

def plot_lasso_grid_search(
    alpha_start_, alpha_stop_, alpha_num_,
    train_, 
    test_,
    target_, 
    cat_feats_,
    drop_cols = ['SalePrice', 'TotalBsmtSF'],
    n_folds = 3
               ):
    colors = ["#FF0B04", "#F1BE48",
           "#B9975B", "#8B5B29",
           "#524727",
         ]
    # lasso regression model set up.
    scaler = StandardScaler(with_mean=False)
    lasso = Lasso()

    X = train_.drop(drop_cols,axis=1)
    transformer = ColumnTransformer([("Cat", 
                                      OneHotEncoder(handle_unknown = 'ignore'), 
                                      cat_feats_)], remainder='passthrough')
    X = transformer.fit_transform(X)
    X = scaler.fit_transform(X)
    y = np.log(train_[target_])

    # Grid Search set up.

    alphas = np.linspace(alpha_start_, alpha_stop_, alpha_num_)
    tuned_parameters = [{'alpha': alphas}]
    clf = GridSearchCV(lasso, tuned_parameters, cv=n_folds, refit=False)
    clf.fit(X, y)
    
    # graphing data set up
    grid_search_df = pd.DataFrame({
        'split0_test_score':clf.cv_results_['split0_test_score'],
        'split1_test_score':clf.cv_results_['split1_test_score'],
        'split2_test_score':clf.cv_results_['split2_test_score'],
        'mean_test_score':clf.cv_results_['mean_test_score'],
        'param_alpha':clf.cv_results_['param_alpha']
    })

    graph_df = pd.melt(
        grid_search_df,
        id_vars=['param_alpha'], 
        value_vars=['split0_test_score','split1_test_score','split2_test_score']
    )
    graph_df.columns = ['alphas','split_test','model_score']
    
    # graph
    sns.lineplot(data = graph_df,
                x = 'alphas', y = 'model_score',
    #             ax=axs[0]
                 color=colors[0]
                )
    plt.show()
    
    
def lasso_train_test_graph(alpha_start, alpha_stop, alpha_num,
               train_, 
               test_,
               target_, 
               cat_feats_,
               drop_cols = ['SalePrice', 'TotalBsmtSF']
               ):
    colors = ["#FF0B04", "#F1BE48",
           "#B9975B", "#8B5B29",
           "#524727",
         ]
    lasso_scores_train = []

    lasso_scores_test  = []
    lasso_feat_len = []

    alphas = np.linspace(alpha_start, alpha_stop, alpha_num)

    for alpha in alphas:
        try:
            train_score, test_score,lasso_feats = lasso_model_score(
                               alpha,

                train_, 
               test_,
               target_, 
               cat_feats_,
               drop_cols
            )
            lasso_scores_train.append(train_score)
            lasso_scores_test.append(test_score)
            lasso_feat_len.append(len(lasso_feats))
        except ValueError:
            print(f'Alpha of {alpha} fails')
            lasso_scores_train.append(0)
            lasso_scores_test.append(0)
            lasso_feat_len.append(0)

    lasso_scores_train = np.array(lasso_scores_train) 
    lasso_scores_test  = np.array(lasso_scores_test)
    lasso_feat_len = np.array(lasso_feat_len)
    #graph
    # construct df.
    lasso_alpha_scores = pd.DataFrame({"alphas":alphas,
                'train':lasso_scores_train,
                'test':lasso_scores_test,
                'feature_len':lasso_feat_len,                                       
                                      })
    # change df to graphable structure.
    graph_df = pd.melt(
        lasso_alpha_scores,
        id_vars=['alphas'], 
        value_vars=['train','test']
    )
    graph_df.columns = ['alphas','data_type','model_score']
    
    fig, axs = plt.subplots(2,1,figsize=(10,12))
    sns.lineplot(data = graph_df,
                x = 'alphas', y = 'model_score',hue = 'data_type',
                ax=axs[0]
                )
    # define variable for test train model that is the closest.
    lasso_alpha_scores['train_test_dist'] = abs(lasso_alpha_scores['train'] -
                                                lasso_alpha_scores['test'])
    shortest_dist = lasso_alpha_scores.sort_values('train_test_dist'
                                               ).reset_index(drop = True).loc[0,:]
    best_lasso_alpha = shortest_dist['alphas']
    best_lasso_train = shortest_dist['train']
    best_lasso_test = shortest_dist['test']
    best_lasso_dist = shortest_dist['train_test_dist']

    # construct clostest alpha line
    axs[0].plot([best_lasso_alpha,best_lasso_alpha], 
             [best_lasso_train,best_lasso_test],color=colors[2])
    axs[0].plot(best_lasso_alpha, best_lasso_train,
             marker='o', markersize=8,
             color=colors[2])
    axs[0].plot(best_lasso_alpha, best_lasso_test,
             marker='o', markersize=8,
             color=colors[2])
    # label closest alpha point
    axs[0].text(best_lasso_alpha +.0005, ((best_lasso_test+best_lasso_train)/2), 
             "Alpha = {:.5f}\nDistance = {:.5f}".format(best_lasso_alpha,best_lasso_dist))
    
    # define variable for the test model that has the best score
    
    test_max = lasso_alpha_scores.sort_values('test',ascending=False
                                             ).reset_index(drop = True).loc[0,:]
    best_lasso_alpha = test_max['alphas']
    best_lasso_test = test_max['test']

    axs[0].plot(best_lasso_alpha, best_lasso_test,
             marker='o', markersize=8,
             color=colors[3])
    axs[0].text(best_lasso_alpha +.0005, best_lasso_test, 
         "Alpha = {:.5f}\nScore = {:.3f}".format(best_lasso_alpha,best_lasso_test))
    
    
    

    axs[0].set_title(r'Lasso Train-Test $R^2$ Comparison')
    axs[0].set_xlabel(r'hyperparameter $\alpha$')
    axs[0].set_ylabel(r'$R^2$')
    
    axs[1].set_xlabel(r'hyperparameter $\alpha$')
    axs[1].set_ylabel(r'Number of Features')




    
    sns.lineplot(data = lasso_alpha_scores,
            x = 'alphas', y = 'feature_len',
                 color = colors[2],
            ax=axs[1]
            )
    plt.show()


    