# Name: 	Denis Stukal
# Date: 	March 23, 2018
# Summary: 	Use twitter_bots module to compute features for bot detection using a JSON file as input
#			Output: features.txt

import codecs, re, os, datetime, time, math, sys
import numpy as np
import pandas as pd
from pysmap import SmappDataset
import pickle
sys.path.insert(0, '~/twitter_bots')
from twitter_bots import Twitter_accounts

print(sys.stdout.encoding)
print(sys.version)

dataset = SmappDataset(['json', 'file_pattern', '~/YOUR_DATA.json'])
output_name = '~/bot_applications/data/mydata'

#---- Extract necessary data from a JSON file
mycol = Twitter_accounts(dates = [('2014-02-06', '2014-10-01'), ('2015-01-30', '2015-12-31')])
mycol.loop(collections = [dataset], functions = ['features'])
print('Looping done')



#---- Write out collected data
print('Get and dump FINAL FEATURES')
mycol.get_final_feature_dict()

output = pd.DataFrame.from_dict(mycol.final_feature_dict, orient = 'index')
output.to_csv('~/bot_applications/data/features.txt', sep='\t')


print('JOB DONE')


