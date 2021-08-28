#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 19:14:24 2021

@author: haydenlw4
"""

import pandas as pd
import helper

housing = pd.read_csv('ames_housing_latlong.csv', index_col=0)

train, test = helper.stratified_split(housing,'Neighborhood')

train, mapping_dict = helper.geo_cords_imputing(train)
test = helper.geo_cords_imputing(test,mapping_dict)
# note there are three houses that didn't have address values and were not
# in neighborhoods to have their geo-coordinates computed. 
# They are all in train.
# their PID are: [916252170, 916253320, 907230240]
train.to_csv('train_geo.csv')
test.to_csv('test_geo.csv')
