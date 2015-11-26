#!/usr/bin/env python

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint
from collections import Counter
from collections import defaultdict


if __name__ == '__main__':
    path = 'data/usagov_bitly_data2012-03-16-1331923249.txt'
    
    with open(path, 'rb') as f:
        records = [json.loads(line.decode('utf-8')) for line in f]
    
    # get top 10 time zones
    print('Top Time Zones:')
    tz = [rec['tz'] for rec in records if 'tz' in rec]
    tz = ['Unknown' if x == '' else x for x in tz]
    tz_counts = Counter(tz)
    pprint(tz_counts.most_common(10))
    
    # now do it with pandas DataFrame
    print('\nTop Time Zones Using DataFrame:')
    frame = pd.DataFrame(records)
    clean_tz = frame['tz'].fillna('Missing')
    clean_tz[clean_tz == ''] = 'Unknown'
    tz_counts = clean_tz.value_counts()
    print(tz_counts[:10])
    
    # plot it
    #tz_counts[:10].plot(kind='barh', rot=0)
    #plt.show()

    # get top agents
    print('\nTop Agents:')
    results = pd.Series([x.split()[0] for x in frame.a.dropna()])
    print(results.value_counts()[:8])
    
    # get top time zones with Windows and non-Windows users
    print('\nUsers and Time Zones:')
    
    # get the frames where the agent is not null
    cframe = frame[frame.a.notnull()]
    
    operating_system = np.where(cframe['a'].str.contains('Windows'),
                                'Windows', 'Not Windows')
    
    # group the time zone to the operating system
    by_tz_os = cframe.groupby(['tz', operating_system])
    
    # count the os by time zones, unstack to a table of counts, replace N/A w/0
    agg_counts = by_tz_os.size().unstack().fillna(0)
    print(agg_counts[:5])
    
    # sort the Windows count column in ascending order
    indexer = agg_counts.sum(1).argsort()
    
    # get the last 10 in that order
    count_subset = agg_counts.take(indexer)[-10:]
    print(count_subset)
    
    # normalize it to the relative percentage
    normed_subset = count_subset.div(count_subset.sum(1), axis=0)
    
    # plot it
    normed_subset.plot(kind='barh', stacked=True)
    plt.show()