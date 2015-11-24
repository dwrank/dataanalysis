#!/usr/bin/env python

from __future__ import division

import numpy as np
import pandas as pd


if __name__ == '__main__':
    unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
    users = pd.read_table('data/movielens/users.dat', sep='::', header=None,
                          names=unames, engine='python')
    
    rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_table('data/movielens/ratings.dat', sep='::', header=None,
                            names=rnames, engine='python')
    
    mnames = ['movie_id', 'title', 'genres']
    movies = pd.read_table('data/movielens/movies.dat', sep='::', header=None,
                            names=mnames, engine='python')
    
    # merge the data
    data = pd.merge(pd.merge(ratings, users), movies)
    print(data[:8])
    
    # get mean ratings for each title by gender
    mean_ratings = data.pivot_table('rating', index='title', columns='gender', aggfunc='mean')
    
    # filter to movies w/ at least 250 ratings
    ratings_by_title = data.groupby('title').size()
    active_titles = ratings_by_title.index[ratings_by_title >= 250]
    
    # select those titles from the mean ratings
    mean_ratings = mean_ratings.ix[active_titles]
    
    # get the top films among female viewers
    top_female_ratings = mean_ratings.sort_values(by='F', ascending=False)
    print(top_female_ratings[:5])
    
    # measure ratings disagreements
    mean_ratings['diff'] = mean_ratings['M'] - mean_ratings['F']
    sorted_by_diff = mean_ratings.sort_values(by='diff')
    print('\nRatings Diff')
    print(sorted_by_diff[:5])
    
    # reverse the order
    sorted_by_diff[::-1][:5]
    
    # ratings w/ the most disagreements (measured by variation or stdev)
    rating_std_by_title = data.groupby('title')['rating'].std()
    
    # filter down to active titles
    rating_std_by_title = rating_std_by_title.ix[active_titles]
    
    # order series by value in descending order
    print('\nHighest Rating Difference')
    print(rating_std_by_title.sort_values(ascending=False)[:5])