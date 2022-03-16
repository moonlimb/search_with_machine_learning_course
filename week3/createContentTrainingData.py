import argparse
import csv
import string
import logging
import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pandas as pd
import nltk
from nltk.stem import SnowballStemmer
from nltk import word_tokenize


nltk.download('punkt')
stemmer = SnowballStemmer("english") 


def transform_name(product_name: str, stemmer: Any) -> str:
    """
    Args:
        product_name: best buy product name
    Returns:
        Transformed product_name: 1) lowercase 2) remove puncutation 3) stem
    """
    # 1) lowercase 
    name = product_name.lower()

    # 2) remove punc
    translator = str.maketrans('', '', string.punctuation)
    name = name.translate(translator)

    # 3) stem
    name = ' '.join([stemmer.stem(token) for token in word_tokenize(name)])
    return name

# Directory for product data
directory = r'/workspace/search_with_machine_learning_course/data/pruned_products/'

parser = argparse.ArgumentParser(description='Process some integers.')
general = parser.add_argument_group("general")
general.add_argument("--input", default=directory,  help="The directory containing product data")
general.add_argument("--output", default="/workspace/datasets/fasttext/output.fasttext", help="the file to output to")

# Consuming all of the product data will take over an hour! But we still want to be able to obtain a representative sample.
general.add_argument("--sample_rate", default=1.0, type=float, help="The rate at which to sample input (default is 1.0)")

# IMPLEMENT: Setting min_products removes infrequent categories and makes the classifier's task easier.
general.add_argument("--min_products", default=0, type=int, help="The minimum number of products per category (default is 0).")

args = parser.parse_args()
output_file = args.output
path = Path(output_file)
output_dir = path.parent
if os.path.isdir(output_dir) == False:
        os.mkdir(output_dir)

if args.input:
    directory = args.input
# IMPLEMENT:  Track the number of items in each category and only output if above the min
min_products = args.min_products
sample_rate = args.sample_rate

logging.info("Writing results to %s" % output_file)

def check_valid_category_name(child: Any):
    """Returns true if category name is valid, false otherwise."""
    return (child.find('name') is not None and child.find('name').text is not None and
            child.find('categoryPath') is not None and len(child.find('categoryPath')) > 0 and
            child.find('categoryPath')[len(child.find('categoryPath')) - 1][0].text is not None)


def get_products_from_files(directory: str, writer: Any):

    for filename in os.listdir(directory):
        if filename.endswith(".xml"):
            print(f"Processing {filename}")

            f = os.path.join(directory, filename)
            tree = ET.parse(f)
            root = tree.getroot()

            for child in root:
                if random.random() > sample_rate:
                    continue
                # Check to make sure category name is valid
                if check_valid_category_name(child):
                    # Choose last element in categoryPath as the leaf categoryId

                    # PROMPT: try different category nodes => adjust granularity of category
                    leaf_index = len(child.find('categoryPath')) - 1
                    ancestor_depth_2_index = 2  
                    ancestor_depth_3_index = 3 
                    # category = child.find('categoryPath')[min(ancestor_depth_2_index, leaf_index)][0].text
                    category = child.find('categoryPath')[min(ancestor_depth_3_index, leaf_index)][0].text

                    # Replace newline chars with spaces so fastText doesn't complain
                    name = child.find('name').text.replace('\n', ' ')

                    transformed_name = transform_name(name, stemmer)
                    writer.writerow([category, transformed_name])
                    # output.write(f"__label__{category} {transformed_name}")

def remove_infrequent_category_products(df: pd.DataFrame, min_products: int) -> pd.DataFrame:
    """Remove categories with products less than min_products from dataframe"""
    # Reference: https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.DataFrameGroupBy.filter.html

    df = pd.read_csv(tempfile, names=['label', 'product_name'])
    return df.groupby('label').filter(lambda x: len(x) > min_products)

tempfile = 'tempfile.csv'
with open(tempfile, 'w') as f:
    writer = csv.writer(f)
    get_products_from_files(directory, writer)
    f.close()

with open(output_file, 'w') as output:
    if min_products > 0:
        dff = remove_infrequent_category_products(tempfile, min_products)
        for _, row in dff.iterrows():
            category = row[0]
            product_name = row[1]
            output.write(f"__label__{category} {product_name}\n")

    else:
        for product in products:
            category = product[0]
            product_name = product[1]
            output.write(f"__label__{category} {product_name}\n")
    f.close()

