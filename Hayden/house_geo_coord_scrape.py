#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 17:38:57 2021

@author: haydenlw4
"""


import pandas as pd
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

# addresses that need to be manually found.
manual_address = {
        '3629 CHILTON AVE':(42.057103,-93.6536014),
        '3623 CHILTON AVE':(42.0569526,-93.6536067),
        '200 TRAIL RIDGE RD':(42.0238493,-93.6761195),
        '3702 CHILTON AVE':(42.057098,-93.6530511),
        '418 E 6TH ST':(42.0268297,-93.6069337),
        '4302 COCHRANE PKWY':(42.0164154,-93.6788928),
        '1505 LITTLE BLUESTEM CT UNIT 116':(42.0095386,-93.6483339),
        '3703 CHILTON AVE':(42.0572375,-93.6536212,),
        '212 TRAIL RIDGE RD':(42.0242917,-93.6761789),
        '2520 E LINCOLN WAY':(42.0223441,-93.5816784),
        '1505 LITTLE BLUESTEM CT UNIT 119':(42.0099384,-93.6482335),
        '1214 N 3RD ST':(42.024953,-93.6297674),
        '1505 LITTLE BLUESTEM CT UNIT 103':(42.0098529,-93.6491565),
        '1505 LITTLE BLUESTEM CT UNIT 107':(42.0098982,-93.6489898),
        '236 TRAIL RIDGE RD':(42.0252163,-93.6758661),
        '1506 LITTLE BLUESTEM CT':(42.009138,-93.6484637),
        '1510 LITTLE BLUESTEM CT':(42.009175,-93.6487457),
        '2133 CONEFLOWER CT':(42.0091729,-93.6461174),
        '1505 LITTLE BLUESTEM CT UNIT 106':(42.0095731,-93.6487992),
        '206 N RIVERSIDE DR':(42.0242525,-93.6317663),
        '1504 LITTLE BLUESTEM CT':(42.0091279,-93.6483207),
        '2027 INDIANGRASS CT':(42.009929,-93.6440617),
        '4107 TRAIL RIDGE CIR':(42.0246355,-93.6771447),
        '1505 LITTLE BLUESTEM CT UNIT 114':(42.0095306,-93.6484221),
        '102 N RIVERSIDE DR':(42.0229961,-93.6317861),
        '1020 10TH ST':(42.0310747,-93.6267592),
        '112 N RIVERSIDE DR':(42.0233439,-93.6317861),
        '1505 LITTLE BLUESTEM CT UNIT 112':(42.0095563,-93.6485228),
        '3712 CHILTON AVE':(42.0574537,-93.6530256),
        '326 N RIVERSIDE DR':(42.0259101,-93.6312715),
        '4205 COCHRANE PKWY':(42.0173822,-93.6777835),
        '207 TRAIL RIDGE RD':(42.0240609,-93.6767534),
                      }

for house in location.index:
    try:
        lat_long = manual_address[location.loc[house,'Prop_Addr']]
        location.loc[house,'latitude'] = lat_long[0]
        location.loc[house,'longitude'] = lat_long[1]
    except KeyError:
        # cleaning street address information
        address = (''.join(''.join(''.join(
            location.loc[house,'Prop_Addr'
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
            location.loc[house,'latitude'] = loc.latitude
            location.loc[house,'longitude'] = loc.longitude
            print(location.loc[house,'Prop_Addr'])
        except:
            print(f"{location.loc[house,'Prop_Addr']} is a problem.")
                    
            

# merging lat long onto original information
location = location[['MapRefNo', 'Prop_Addr', 'latitude','longitude']]
location.rename(columns={'MapRefNo':'PID'},inplace=True)
housing = pd.merge(housing,location,on='PID',how='left')

housing.to_csv('ames_housing_latlong.csv')
