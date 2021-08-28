#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 16:43:27 2021

@author: haydenlw4
"""

import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

housing = pd.read_csv('Ames_Housing_Price_Data.csv', index_col=0,low_memory = False)
location = pd.read_csv('Ames_Real_Estate_Data.csv', low_memory=False)

latlng = pd.read_csv('location_lat_long.csv', 
                       index_col=0,
                      # low_memory = False
                      )
# these lat/longs were found by going on google maps and looking for the next
# landmark that was outside of Ames, Iowa.
north_limit = 42.1066576
south_limit = 41.9116165
east_limit = -93.471096
west_limit = -93.8985648

for house in latlng.index:
    # only looking at houses where lat is outside Ames city limits
    lat = latlng.loc[house,'latitude']
    if not ((lat > south_limit) & (lat < north_limit)):
        long = latlng.loc[house,'longitude']
        # only looking at houses where long is outside Ames city limits
        if  not ((long > west_limit) & (long < east_limit)):
            # cleaning street address information
            address = (''.join(''.join(''.join(
                latlng.loc[house,'Prop_Addr'
            ].partition(' ST')[0:2] # removing values after "ST"
            ).partition(' AVE')[0:2] # removing values after "AVE"
            ).partition(' CT')[0:2] # removing values after "CT"
            ).replace('O NEIL',"O'NEIL" 
            ) + ', ' + 
                'Ames, Iowa')
            geolocator = Nominatim(user_agent="ames_location")
            geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
            loc = geolocator.geocode(address)
            try:
                latlng.loc[house,'latitude'] = loc.latitude
                latlng.loc[house,'longitude'] = loc.longitude
                print(latlng.loc[house,'Prop_Addr'])
            except:
                print(f"{latlng.loc[house,'Prop_Addr']} is a problem.")
            
# Looking at addresses that are still outside of Ames.
failed_lat_long_list = []
for house in latlng.index:
    # only looking at houses where lat is outside Ames city limits
    lat = latlng.loc[house,'latitude']
    if not ((lat > south_limit) & (lat < north_limit)):
        long = latlng.loc[house,'longitude']
        # only looking at houses where long is outside Ames city limits
        if  not ((long > west_limit) & (long < east_limit)):
            failed_lat_long_list.append(house)
failed_lat_long_df = latlng.loc[failed_lat_long_list,:]

# merging lat long onto original information
latlng = latlng[['MapRefNo', 'Prop_Addr', 'latitude','longitude']]
latlng.rename(columns={'MapRefNo':'PID'},inplace=True)
housing = pd.merge(housing,latlng,on='PID',how='left')

# saving values
housing.to_csv('ames_housing_latlong.csv')
