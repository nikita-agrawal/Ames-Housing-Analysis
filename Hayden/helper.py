#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 08:55:00 2021

@author: haydenlw4
"""

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

def stratified_split(df,stratified_col,
                     n_splits = 1, 
                     test_size = 0.25, 
                     random_state = 42):
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