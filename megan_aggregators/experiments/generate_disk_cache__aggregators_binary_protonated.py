import os
import pathlib
import typing as t
import random
from collections import Counter

import numpy as np
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path

PATH = pathlib.Path(__file__).parent.absolute()
ASSETS_PATH = os.path.join(PATH, 'assets')
CACHE_PATH = os.path.join(PATH, 'cache')

# == SOURCE PARAMETERS ==
# These parameters configure the source dataset which should be converted.

# :param CSV_PATH:
#       This is the path to the source dataset, which has to be given in the CSV format.
#       This CSV file has to contain the SMILES representation of the file in question 
#       and the information about the corresponding ground truth binary classification
#       labels.
CSV_PATH: str = os.path.join(ASSETS_PATH, 'aggregators_binary_protonated.csv')
# :param SMILES_COLUMN:
#       This is the string name of the CSV column that contains the SMILES representations 
#       of the various molecules that make up the dataset.
SMILES_COLUMN: str = 'smiles'
# :param TARGET_COLUMNS:
#       In this list, define the names of the columns that define the target label annotations 
#       of the dataset.
TARGET_COLUMNS: t.List[str] = ['nonaggregator', 'aggregator']
# :param NUM_TEST:
#       This is the number of elements to be used for the test set.
NUM_TEST: int = 2000
# :param TARGET_OVERSAMPLING_FACTORS:
#       This dictionary defines the oversampling to be applied to the dataset. The keys are the 
#       integer indices of the possible target classes and the values are the integer factors of 
#       oversampling to be applied to all elements of that corresponding class. For example the 
#       entry "{0: 5}" means that all the elements with the target class 0 are copied a total 
#       of 5 times into the final dataset.
TARGET_OVERSAMPLING_FACTORS: t.Dict[int, float] = {
    0: 1,
    1: 30,
}
# :param SHUFFLE_DATASET:
#       A boolean flag of whether the dataset should be additionally shuffled before it is 
#       converted into the disk files. Note that the shuffling is applied after the oversampling
SHUFFLE_DATASET: bool = True

# == DESTINATION PARAMETERS ==
# These are the parameters that determine where and how the dataset will be saved to.

# :param DESTINATION_PATH:
#       This is the the folder path at which the cache folder should be created. Make sure that 
#       this path does NOT already exist. This folder will be created and populated during 
#       the experiment
DESTINATION_PATH: str = os.path.join(CACHE_PATH, 'aggregators_binary_protonated')
# :param CHUNK_SIZE:
#       This is the number of elements that should comprise a single training chunk. This is 
#       therefore the number of elements that should be able to comfortably fit into the 
#       system memory at the same time.
CHUNK_SIZE: int = 500_000


experiment = Experiment.extend(
    'generate_disk_cache.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)


@experiment.hook('choose_test_set')
def choose_test_set(e: Experiment,
                    element_dict: t.List[dict]
                    ):
    """
    For the aggregators_binary_protonated dataset we need a special method to choose the test set 
    due to the following property of the dataset.
    
    The dataset consists of ~2.5M elements but was originally derived from a dataset of ~300k elements 
    through data augmentation. Essentially, in this dataset a lot of the values are essentially duplicates 
    of each other but with a small difference. Now for the test set we actually want to sample elements 
    such that:
    - They are equally distributed w.r.t. the target values
    - There are none of these duplicates, but instead all the elements are actually unique molecules
    """
    
    # First of all we choose the indices...    
    num_targets = len(e.TARGET_COLUMNS)
    num_elements = int(e.NUM_TEST / num_targets)
    
    unique_keys = {i: set() for i in range(num_targets)}
    test_dict = {}
    
    index = 0
    while not all([len(keys) >= num_elements for keys in unique_keys.values()]) and index < len(element_dict):
        data = element_dict[index]
        label = np.argmax(data['targets'])
        
        if data['unique'] not in unique_keys[label] and len(unique_keys[label]) < num_elements:
            unique_keys[label].add(data['unique'])
            test_dict[index] = data
            del element_dict[index]
            
        index += 1
            
    # This actually fixes a very important problem and we can phrase it like this: If this loop encounters 
    # an element in the dataset for which the unique (original molecule) index is already part of the test 
    # set, then we need to remove that element from the train set.
    # If we did not do this, then the test set itself would not actually contain unseen molecules compared 
    # to the train set, but rather it would only contain unseen protonation states, which is too weak of a 
    # difference for us here.
    deletion_count = 0
    indices = list(element_dict.keys())
    for index in indices:
        data = element_dict[index]
        label = np.argmax(data['targets'])
        
        if data['unique'] in unique_keys[label]:
            del element_dict[index]
            deletion_count += 1
            
    e.log(f'deleted {deletion_count} elements from train set due to test set overlap')
            
    # Now we just want to print some info about the target distribution in the selected test set.
    counter = Counter([np.argmax(data['targets']) for data in test_dict.values()])
    e.log(f'selected test indices by label: {counter}')
    
    return test_dict


experiment.run_if_main()
