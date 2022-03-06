import os
import random
import xml.etree.ElementTree as ET
import argparse
import string
from pathlib import Path
from typing import Any

# import pandas as pd
import nltk
from nltk.stem import SnowballStemmer
from nltk import word_tokenize


directory = r'/workspace/search_with_machine_learning_course/data/pruned_products'
parser = argparse.ArgumentParser(description='Process some integers.')

general = parser.add_argument_group("general")
general.add_argument("--input", default=directory,  help="The directory containing the products")
general.add_argument("--output", default="/workspace/datasets/fasttext/titles.txt", help="the file to output to")

# Consuming all of the product data takes a while. But we still want to be able to obtain a representative sample.
general.add_argument("--sample_rate", default=0.1, type=float, help="The rate at which to sample input (default is 0.1)")

args = parser.parse_args()
output_file = args.output
path = Path(output_file)
output_dir = path.parent
if os.path.isdir(output_dir) == False:
        os.mkdir(output_dir)

if args.input:
    directory = args.input

sample_rate = args.sample_rate

nltk.download('punkt')
stemmer = SnowballStemmer("english") 


def transform_training_data(name: str, stemmer: SnowballStemmer) -> str:
    """
    Args:
        name: best buy product category name (ex. Printer Ink, Best Buy)
    Returns:
        Transformed name: 1) lowercase 2) remove puncutation 3) stem
    """
    print(name)
    # 1) lowercase 
    name = name.lower()

    # 2) remove punc
    translator = str.maketrans('', '', string.punctuation)
    name = name.translate(translator)

    # 3) stem
    name = ' '.join([stemmer.stem(token) for token in word_tokenize(name)])
    return name.replace('\n', ' ')

def check_valid_category_name(child: Any) -> bool:
    return child.find('name') is not None and child.find('name').text is not None
# Directory for product data

print("Writing results to %s" % output_file)
with open(output_file, 'w') as output:
    for filename in os.listdir(directory):
        if filename.endswith(".xml"):
            f = os.path.join(directory, filename)
            tree = ET.parse(f)
            root = tree.getroot()
            for child in root:
                if random.random() > sample_rate:
                    continue
                if check_valid_category_name(child):
                    name = transform_training_data(child.find('name').text, stemmer)
                    output.write(name + "\n")
