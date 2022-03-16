import os
import argparse
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import csv
from typing import Dict

import nltk
from nltk import word_tokenize
from nltk.stem import SnowballStemmer

# Useful if you want to perform stemming.
nltk.download('punkt')
# stemmer = nltk.stem.PorterStemmer()
stemmer = SnowballStemmer("english") 


# utils
def transform_query(
    query: str, lower: bool, remove_punc: bool, stem: bool, remove_line_break: bool=False) -> str:
    """
    Args:
        query: Query string (ex. ipad 13)
    Returns:
        normalized query string, ex. lowercasing and removing punctuation
    """
    normalized = None
    if lower: normalized = query.lower()
    if remove_punc: 
        #normalized = ''.join([c for c in normalized if c.isalnum() or c == ' '])
       translator = str.maketrans('', '', string.punctuation)
       normalized = normalized .translate(translator)
    if stem: normalized = ' '.join([stemmer.stem(token) for token in word_tokenize(normalized)])
    return normalized

def normalize_column(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    df[col_name] = df[col_name].apply(lambda x: transform_query(x, lower=True, remove_punc=True, stem=True))
    return df 

def shuffle_rows(df: pd.DataFrame, sample_rows: int=5000) -> pd.DataFrame:
    return df.sample(n=sample_rows)

# RUN:  python week4/create_labeled_queries.py --min_queries 1000 --read_normalized True

categories_file_name = r'/workspace/datasets/product_data/categories/categories_0001_abcat0010000_to_pcmcat99300050000.xml'

queries_file_name = r'/workspace/datasets/train.csv'
normalized_queries_file_name = r'/workspace/datasets/train_normalized.csv'
output_file_name = r'/workspace/datasets/labeled_query_data.txt'

parser = argparse.ArgumentParser(description='Process arguments.')
general = parser.add_argument_group("general")
general.add_argument("--min_queries", default=1,  help="The minimum number of queries per category label (default is 1)")
general.add_argument("--output", default=output_file_name, help="the file to output to")
general.add_argument("--read_normalized", default=False, help="Read from normalized query file (intermediate data)")

args = parser.parse_args()
output_file_name = args.output
read_normalized = args.read_normalized or None
min_queries = int(args.min_queries) if args.min_queries else None
if min_queries:
    output_file_name = f'/workspace/datasets/labeled_query_data_{min_queries}.txt'

print(f'Config: min queries {min_queries}')

# The root category, named Best Buy with id cat00000, doesn't have a parent.
root_category_id = 'cat00000'

tree = ET.parse(categories_file_name)
root = tree.getroot()

# Parse the category XML file to map each category id to its parent category id in a dataframe.
categories = []
parents = []
depths = []
for child in root:
    id = child.find('id').text
    cat_path = child.find('path')
    cat_path_ids = [cat.find('id').text for cat in cat_path]
    leaf_id = cat_path_ids[-1]
    if leaf_id != root_category_id:
        categories.append(leaf_id)
        parents.append(cat_path_ids[-2])
        depths.append(len(cat_path_ids))
parents_df = pd.DataFrame(list(zip(categories, parents, depths)), columns =['category', 'parent', 'depths'])
# import ipdb; ipdb.set_trace()

# Read the training data into pandas, only keeping queries with non-root categories in our category tree.
print('Read training data: category and query')
df = None
if read_normalized:
    print(f'Read from normalized file: {normalized_queries_file_name}')
    df = pd.read_csv(normalized_queries_file_name)
    df = df[df['query'].notna()]
else: 
    df = pd.read_csv(queries_file_name)[['category', 'query']]
    df = df[df['category'].isin(categories)]

    # [DONE]: Convert queries to lowercase, and optionally implement other normalization, like stemming.
    print('Normalize queries... (lowercase, remove puncutation, etc.)')

    df = normalize_column(df=df, col_name='query')
    print(f'Normalized {len(df)} rows')

    print(f'Save normalized file to {normalized_queries_file_name}')
    df.to_csv(normalized_queries_file_name, index=False)

# [DONE]: Roll up categories to ancestors to satisfy the minimum number of queries per category.
print(f'Rolling up categories to ancestors.')
num_categories = df['category'].nunique()
print(f'Current unique categories: {num_categories}')

def group_by_and_count(df: pd.DataFrame, groupby_col: str, count_col: str) -> pd.DataFrame:
    return df.groupby(groupby_col).size().reset_index(name=count_col)


def roll_up_categories(df_queries: pd.DataFrame, df_parents: pd.DataFrame, min_queries: int) -> pd.DataFrame:
    print('[roll up] Add category count column')
    category_count = group_by_and_count(df=df_queries, groupby_col='category', count_col='count')

    print('[roll up] Merge dataframes: queries, category count and parents.')
    print('')
    df = (
        df_queries
        .merge(category_count, how='left', on='category') # add category count column
        .merge(df_parents, how='left', on='category') # add category's parent
    )

    # should_roll_up = df['count'] < min_queries
    category_to_parent = dict(zip(df_parents['category'], df_parents['parent'])) 
    category_to_parent[root_category_id] = root_category_id # add root just in case

    roll_up_round = 0

    while (category_count['count'] < min_queries).any():
        # import ipdb; ipdb.set_trace()
        should_roll_up = df['count'] < min_queries
        print(f'[roll up] rows: {should_roll_up.sum()}')
        print(f'[roll up] Current roll up round: {roll_up_round}')

        print('[roll up] Replace category')
        df.loc[should_roll_up, 'category'] = df.loc[should_roll_up]['category'].apply(
            lambda cat: category_to_parent[cat])
        
        num_categories = df['category'].nunique()
        print(f'[roll up] Unique categories after round ({roll_up_round}): {num_categories}')

        # print('[roll up] Replace parent')
        # df.loc[should_roll_up, 'parent'] = df.loc[should_roll_up]['parent'].apply(lambda x: category_to_parent.get(x))

        # recalculte category count 
        category_count = group_by_and_count(df=df, groupby_col='category', count_col='count')
        new_count = dict(zip(category_count['category'], category_count['count']))
        df.loc[should_roll_up, 'count'] = df.loc[should_roll_up]['category'].apply(lambda x: new_count.get(x))

        # reset
        roll_up_round += 1
        # should_roll_up = df['count'] < min_queries
    return df


df = roll_up_categories(df_queries=df, df_parents=parents_df, min_queries=min_queries or 1)

print('Shuffle and sample 50,000 rows')
df = shuffle_rows(df=df, sample_rows=50000)


# Create labels in fastText format.
df['label'] = '__label__' + df['category']

# Output labeled query data as a space-separated file, making sure that every category is in the taxonomy.
df = df[df['category'].isin(categories)]
df = df[df['query'].notna() & df['category'].notna() & df['label'].notna()] # not sure why this was ncessary for me
df['output'] = df['label'] + ' ' + df['query']
df[['output']].to_csv(output_file_name, header=False, sep='|', escapechar='\\', quoting=csv.QUOTE_NONE, index=False)
