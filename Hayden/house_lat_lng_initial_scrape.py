#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 12:53:38 2021

@author: haydenlw4
"""

import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

housing = pd.read_csv('Ames_Housing_Price_Data.csv', index_col=0,low_memory = False)
location = pd.read_csv('Ames_Real_Estate_Data.csv', low_memory=False)

location = location[['MapRefNo', 'Prop_Addr', 'MA_City', 'MA_State', 'MA_Zip1']]
PID_list = list(housing.PID.unique())
location = location[location['MapRefNo'].isin(PID_list)]
location.drop_duplicates(inplace=True)
location = location.reset_index(drop=True)
location.loc[:,'latitude'] = -1
location.loc[:,'longitude'] = -1

locator = Nominatim(user_agent="ames_location")
geocode = RateLimiter(locator.geocode, min_delay_seconds=1)
problem_map_ref = []
location.loc[:,'MA_Zip1'] = location.loc[:,'MA_Zip1'].fillna(value=0)
for house in location.index:
    address = (str(location.loc[house,'Prop_Addr']) + ', ' + 
        str(location.loc[house,'MA_City']) + ', ' + 
        str(location.loc[house,'MA_State']) + ' ' +
        str(int(location.loc[house,'MA_Zip1'])))
    geolocator = Nominatim(user_agent="ames_location")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    loc = geolocator.geocode(address)
    try:
        location.loc[house,'latitude'] = loc.latitude
        location.loc[house,'longitude'] = loc.longitude
        print(location.loc[house,'Prop_Addr'])
    except:
        print(f"{location.loc[house,'Prop_Addr']} is a problem.")
        problem_map_ref.append(location.loc[house,'MapRefNo'])
        
location.to_csv('location_lat_long.csv')

