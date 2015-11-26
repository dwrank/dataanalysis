#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def add_prop(group):
    births = group.births.astype(float)
    group['prop'] = births / births.sum()
    return group


def get_quantile_count(group, q=0.5):
    group = group.sort_values('prop', ascending=False)
    return group.prop.cumsum().values.searchsorted(q) + 1

    
def get_top(group, count):
    return group.sort_values(by='births', ascending=False)[:count]


if __name__ == '__main__':
    columns = ['name', 'sex', 'births']
    names1880 = pd.read_csv('data/names/yob1880.txt', names=columns)
    print(names1880.groupby('sex')['births'].sum())
    
    # assemble all of the data
    years = range(1880, 2011)
    pieces = []
    
    for year in years:
        path = 'data/names/yob%d.txt' % year
        frame = pd.read_csv(path, names=columns)
        frame['year'] = year
        pieces.append(frame)
    
    # concat into a single data Frame
    names = pd.concat(pieces, ignore_index=True)
    
    # aggregate the data at the year and sex level
    total_births = names.pivot_table('births', index='year', columns='sex', aggfunc=sum)
    print(total_births.tail())
    
    total_births.plot(title='Total births by sex and year')
    #plt.show()
    
    names = names.groupby(['year', 'sex']).apply(add_prop)
    print(names[:6])
    
    # verify that proportion adds up to approx 1
    print(np.allclose(names.groupby(['year', 'sex']).prop.sum(), 1))
    
    # get the top names for each sex-year combo
    grouped = names.groupby(['year', 'sex'])
    top = grouped.apply(get_top, 1000)
    top.index = np.arange(len(top))
    print(top)
    
    # get the top girl and boy names
    boys = top[top.sex == 'M']
    girls = top[top.sex == 'F']
    
    # get the total births by year and name
    total_births = top.pivot_table('births', index='year', columns='name', aggfunc=sum)
    total_births.info()
    
    # plot the top names
    subset = total_births[['John', 'Harry', 'Mary', 'Marilyn']]
    #subset.plot(subplots=True, figsize=(12, 10), grid=False, title="Number of births per year")
    #plt.show()
    
    # plot the proportion of births of the top names by year
    table = top.pivot_table('prop', index='year', columns='sex', aggfunc=sum)
    #table.plot(title='Sum of table1000.prop by year and sex',
    #           yticks=np.linspace(0, 1.2, 13), xticks=range(1880, 2020, 10))
    #plt.show()
    
    # get the boy names for 2010
    df = boys[boys.year == 2010]
    
    # sort the proportions in descending order, and get the cumulative sum
    prop_cumsum = df.sort_values(by='prop', ascending=False).prop.cumsum()
    
    # find how many of the smallest proportioned names it takes to reach 50%, add 1 to offset the 0 index
    print(prop_cumsum.values.searchsorted(0.5) + 1)
    
    # check what it is for 1900
    df = boys[boys.year == 1900]
    prop_cumsum = df.sort_values(by='prop', ascending=False).prop.cumsum()
    print(prop_cumsum.values.searchsorted(0.5) + 1)
    
    # plot the diversity (the # of least proportioned names to make 50%)
    diversity = top.groupby(['year', 'sex']).apply(get_quantile_count)
    diversity = diversity.unstack('sex')
    print(diversity.head())
    
    #diversity.plot(title="Number of popular names in top 50%")
    #plt.show()
    
    # investigate the distribution of names by the last letter
    # get the last letters and name it
    get_last_letter = lambda x: x[-1]
    last_letters = names.name.map(get_last_letter)
    last_letters.name = 'last_letter'
    
    # get the # of births by the last letters for each sex and year
    table = names.pivot_table('births', index=last_letters, columns=['sex', 'year'], aggfunc=sum)
    
    # select 3 years
    subtable = table.reindex(columns=[1910, 1960, 2010], level='year')
    print(subtable.head())
    
    # get the proportion of last letters
    letter_prop = subtable / subtable.sum().astype(float)
    
    # plot it
    fig, axes = plt.subplots(2, 1, figsize=(10,8))
    letter_prop['M'].plot(kind='bar', rot=0, ax=axes[0], title='Male')
    letter_prop['F'].plot(kind='bar', rot=0, ax=axes[1], title='Female', legend=False)
    
    # select a subset of letters for a time series
    letter_prop = table / table.sum().astype(float)
    dny_ts = letter_prop.ix[['d', 'n', 'y'], 'M'].T
    #dny_ts.plot()
    #plt.show()
    
    # boy names that became girl names using Leslie
    all_names = names.name.unique()
    mask = np.array(['lesl' in x.lower() for x in all_names])  # true or false values
    leslie_like = all_names[mask]
    
    # get the top names that are lesley like
    filtered = top[top.name.isin(leslie_like)]
    
    # sum them
    print(filtered.groupby('name').births.sum())
    
    # get Leslie like names births by sex and year
    table = filtered.pivot_table('births', index='year', columns='sex', aggfunc=sum)
    
    # normalize by year
    table = table.div(table.sum(1), axis=0)
    print(table.tail())
    