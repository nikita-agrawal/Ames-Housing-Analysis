{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c4115edf",
   "metadata": {},
   "outputs": [],
   "source": [
    "#!/usr/bin/env python3\n",
    "# -*- coding: utf-8 -*-\n",
    "\"\"\"\n",
    "Created on Wed Aug 25 15:10:24 2021\n",
    "\n",
    "@author: Hayden Warren, \n",
    "\"\"\"\n",
    "###################FUTURE WORK##################################\n",
    "# Fix the drop_now_but_look_at_later features.\n",
    "################################################################\n",
    "\n",
    "# Loading packages\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import helper # my function that i added\n",
    "\n",
    "# loading and splitting data\n",
    "housing = pd.read_csv('Ames_Housing_Price_Data.csv', index_col=0)\n",
    "\n",
    "train, test = helper.stratified_split(housing,'Neighborhood')\n",
    "\n",
    "# converting all similar mappings together\n",
    "# most popular mapping\n",
    "cat_ordinal_features = [\n",
    "    'GarageQual','GarageCond',\n",
    "    'FireplaceQu',\n",
    "    'KitchenQual',\n",
    "    'ExterQual','ExterCond',\n",
    "    'BsmtQual','BsmtCond',\n",
    "    'HeatingQC'\n",
    "    ]\n",
    "cat_ordinal_dict = {'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}\n",
    "train = helper.convert_cat_ordinal_vars_to_num(train,\n",
    "                                               cat_ordinal_features,\n",
    "                                               cat_ordinal_dict)\n",
    "# now just unique mappings\n",
    "# BsmtExposure\n",
    "cat_ordinal_features = [\n",
    "    'BsmtExposure'\n",
    "]\n",
    "cat_ordinal_dict = {'No':1,'Mn':2,'Av':3,'Gd':4}\n",
    "train = helper.convert_cat_ordinal_vars_to_num(train,\n",
    "                                               cat_ordinal_features,\n",
    "                                               cat_ordinal_dict)\n",
    "# Functional\n",
    "cat_ordinal_features = [\n",
    "    'Functional'\n",
    "]\n",
    "cat_ordinal_dict = {'Sal':1,'Sev':2,'Maj2':3,'Maj1':4,\n",
    "                    'Mod':5,'Min2':6,'Min1':7,'Typ':8}\n",
    "train = helper.convert_cat_ordinal_vars_to_num(train,\n",
    "                                               cat_ordinal_features,\n",
    "                                               cat_ordinal_dict)\n",
    "# PoolQC\n",
    "cat_ordinal_features = [\n",
    "    'PoolQC'\n",
    "]\n",
    "cat_ordinal_dict = {'Fa':1,'TA':2,'Gd':3,'Ex':4}\n",
    "train = helper.convert_cat_ordinal_vars_to_num(train,\n",
    "                                               cat_ordinal_features,\n",
    "                                               cat_ordinal_dict)\n",
    "# (JH) Garage Finish\n",
    "cat_ordinal_features = [\n",
    "    'GarageFinish'\n",
    "]\n",
    "cat_ordinal_dict = {'Unf':1,'RFn':2,'Fin':3}\n",
    "train = helper.convert_cat_ordinal_vars_to_num(train,\n",
    "                                               cat_ordinal_features,\n",
    "                                               cat_ordinal_dict)\n",
    "\n",
    "# (JH) Alley\n",
    "cat_ordinal_features = [\n",
    "    'Alley'\n",
    "]\n",
    "cat_ordinal_dict = {'Grvl':1,'Pave':2}\n",
    "train = helper.convert_cat_ordinal_vars_to_num(train,\n",
    "                                               cat_ordinal_features,\n",
    "                                               cat_ordinal_dict)\n",
    "\n",
    "# (JH) PavedDrive\n",
    "cat_ordinal_features = [\n",
    "    'Alley'\n",
    "]\n",
    "cat_ordinal_dict = {'P':1,'Y':2}\n",
    "train = helper.convert_cat_ordinal_vars_to_num(train,\n",
    "                                               cat_ordinal_features,\n",
    "                                               cat_ordinal_dict)\n",
    "\n",
    "# Fence\n",
    "# (JH) Changing Fence, will feature engineer it to ordinal\n",
    "# cat_ordinal_features = [\n",
    "#     'Fence'\n",
    "# ]\n",
    "# cat_ordinal_dict = {'MnWw':1,'GdWo':2,'MnPrv':1,'GdPrv':2}\n",
    "# train = helper.convert_cat_ordinal_vars_to_num(train,\n",
    "#                                                cat_ordinal_features,\n",
    "#                                                cat_ordinal_dict)\n",
    "\n",
    "############################################################\n",
    "# weirdest nas. lot frontage. probably worth removing\n",
    "# not dealing with them out of expediance. \n",
    "# (JH) moved MasVnrType to categorical because the boxplot shows it is it's own cat.\n",
    "drop_now_but_look_at_later = ['LotFrontage','MasVnrArea','GarageYrBlt']\n",
    "train.drop(drop_now_but_look_at_later, axis=1,inplace = True)\n",
    "\n",
    "# NAs that have meaning based on data dicitonary.\n",
    "# nas are \"None\" categorical value\n",
    "# (JH) Moved 'GarageFinish' and 'Alley' to ordinal\n",
    "na_none_features = ['MiscFeature','BsmtFinType1','BsmtFinType2',\n",
    "                   'GarageType', 'MasVnrType']\n",
    "for na_none_feature in na_none_features:\n",
    "    train[na_none_feature] = train[na_none_feature].fillna(value = 'None')\n",
    "# nas are 0 numerical value\n",
    "na_zero_features = ['BsmtFullBath','BsmtHalfBath','GarageFinish','Alley']\n",
    "for na_none_feature in na_zero_features:\n",
    "    train[na_none_feature] = train[na_none_feature].fillna(value = 0)\n",
    "    \n",
    "############################################################\n",
    "# second group of features that have problems that I need help solving.\n",
    "cols_na = train.loc[:,train.isna().any(axis=0)].columns.to_list()\n",
    "cols_na\n",
    "\n",
    "#############################################################\n",
    "# (JH) Julie's specific additions\n",
    "# 1. Replacing two values of GarageType to None\n",
    "\n",
    "train.loc[trainSet['PID'] == 903426160,'GarageType'] = 'None'\n",
    "train.loc[trainSet['PID'] == 910201180,'GarageType'] = 'None'\n",
    "\n",
    "# "
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}