"""
This experiment can be used to convert a CSV dataset into the disk cache version. This disk cache 
version represents the dataset immediately as the Tensor instances that are needed to train a model.

Such a disk cache conversion is primarily required for *really* large datasets - ones that do not 
completely fit into the RAM of the system. In such cases, the dataset has to be loaded from the 
the disk in chunks / smaller subsets. This is exactly what this experiment is supposed to do.

This experiment takes the path to a CSV SMILES dataset file as an input and then proceeds to convert 
that file into a folder containing two sub folders:
- test: A visual graph dataset folder containing the test set that was previously chosen.
- train: A folder consisting of multiple pickled files where each one contains the serialized versions 
  of the Tensor instances required for the training of a keras / kgcnn model.
"""
import os
import csv
import time
import pathlib
import random
import pickle
import datetime
import typing as t
from collections import Counter

import tensorflow as tf
import numpy as np
from kgcnn.data.utils import ragged_tensor_from_nested_numpy
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path
from visual_graph_datasets.processing.molecules import MoleculeProcessing
from visual_graph_datasets.data import VisualGraphDatasetWriter
from graph_attention_student.data import tensors_from_graphs
from visual_graph_datasets.experiments.generate_molecule_dataset_from_csv import PROCESSING


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
CSV_PATH: str = os.path.join(ASSETS_PATH, 'aggregators_binary.csv')
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
NUM_TEST: int = 200
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
DESTINATION_PATH: str = os.path.join(CACHE_PATH, 'aggregators_binary')
# :param CHUNK_SIZE:
#       This is the number of elements that should comprise a single training chunk. This is 
#       therefore the number of elements that should be able to comfortably fit into the 
#       system memory at the same time.
CHUNK_SIZE: int = 100_000


@Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)
def experiment(e: Experiment):
    e.log('start experiment...')
    
    # ~ Checking the destination path
    # If the given destination path already exists then we don't actually want to execute this 
    # script. We do it like that to prevent a potentially accidental deletion of an already existing 
    # cache folder - which would also require an expensive recomputation...
    if os.path.exists(e.DESTINATION_PATH):
        e.log(f'destination path "{e.DESTINATION_PATH}" already exists! Stopping experiment...')
        return
    else:
        os.mkdir(e.DESTINATION_PATH)
    
    # ~ Loading the source dataset
    # In the first...
    e.log(f'loading csv file @ {e.CSV_PATH}')
    element_list = []
    with open(e.CSV_PATH, mode='r') as file:
        dict_reader = csv.DictReader(file)
        for index, data in enumerate(dict_reader):
            
            targets = np.array([float(data[name]) for name in e.TARGET_COLUMNS])
            data['targets'] = targets
            
            element_list.append(data)
            
    e.log(f'loaded dataset with {len(element_list)} elements')
    
    counter = Counter([np.argmax(data['targets']) for data in element_list])
    e.log(f'target class counts: {counter}')
    
    # Now we turn the dataset into a dict, because a dict will be the easier data structure 
    # to work with for the following steps.
    # The keys of this dict will be arbitrarily assigned integer indices and the values will 
    # be the actual dict structures that represent the elements.
    element_dict: t.Dict[int, dict] = dict(enumerate(element_list))
    
    # ~ selecting the test set
    # As the first processing step of the loaded dataset we actually want to select the 
    # test set. The actual selection of the test set elements is implemented in a hook because 
    # we might want to customize that with custom behavior.
    # We then need to save the test set as a visual graph dataset into the destination 
    # folder!
    
    @e.hook('choose_test_set')
    def choose_test_set(e: Experiment,
                        element_dict: t.List[dict]
                        ):
        indices = list(element_dict.keys())
        test_indices = random.sample(indices, k=e.NUM_TEST)
        
        test_dict = {}
        for index in test_indices:
            test_dict[index] = element_dict[index]
            del element_dict[index]
            
        return test_dict
    
    test_dict: t.Dict[int, dict] = e.apply_hook(
        'choose_test_set',
        element_dict=element_dict,
    ) 
    
    # Now we need to convert that test set into a visual graph dataset, which we do by 
    # iterating through it all and using a Processing instance as well as a 
    # VisualGraphDatasetWriter instance
    test_path = os.path.join(e.DESTINATION_PATH, 'test')
    os.mkdir(test_path)
    
    e.log('setting up the Processing instance...')
    # processing = MoleculeProcessing()
    processing = PROCESSING

    writer = VisualGraphDatasetWriter(path=test_path)
    
    e.log(f'processing the test set as a VGD...')
    e.log(f'path: {test_path} - num: {len(test_dict)}')
    for i, (index, data) in enumerate(test_dict.items()):
        processing.create(
            value=data[e.SMILES_COLUMN],
            additional_graph_data={'graph_labels': data['targets']},
            index=str(index),
            writer=writer,
        )
        if i % 100 == 0:
            e.log(f' * ({i:03d}/{e.NUM_TEST}) processed')
    
    # ~ oversampling
    # We potentially want to oversample the dataset according to the target value distribution. 
    # If one targets happens to occur significantly less in the dataset then it would make sense 
    # to use multiple copies of those samples for the training of a machine learning model such 
    # that all classes are represented approx. equally.
    e.log('oversampling the dataset...')
    element_list = []
    for data in element_dict.values(): 
        target = np.argmax(data['targets'])
        factor = e.TARGET_OVERSAMPLING_FACTORS[target]
        element_list += [data] * factor
            
    e.log(f'after oversampling: {len(element_list)} elements')
            
    # ~ shuffling the dataset
    if e.SHUFFLE_DATASET:
        e.log('shuffling order of dataset elements...')
        random.shuffle(element_list)
        
    element_dict = dict(enumerate(element_list))
    num_elements = len(element_dict)
        
    # ~ processing the training dataset into a tensor cache
    train_path = os.path.join(e.DESTINATION_PATH, 'train')
    os.mkdir(train_path)
    
    e.log(f'processing the train set as tensor cache...')
    e.log(f'path: {train_path} - num: {num_elements}')
    
    @e.hook('save_chunk')
    def save_chunk(e: Experiment,
                   path: str,
                   graphs: t.List[dict],
                   ):
        
        # First we convert the input graph representations themselves into the correct tensor formats
        # The result of this function is the tuple (node_attributes, edge_attributes, edge_indices)
        # of 3 ragged tensors
        x = tensors_from_graphs(graphs)
        
        # Then also we need to save the corresponding information about the target labels that the network 
        # will be trained on. These are saved in the "graph_labels" attribute of the graph dict itself.
        # Due to the special structure of the MEGAN model we also need to create empty explanation masks as 
        # mock targets for the explanation masks as well.
        targets = [graph['graph_labels'] for graph in graphs]
        ni = ragged_tensor_from_nested_numpy([np.zeros((graph['node_attributes'].shape[0], 1)) for graph in graphs])
        ei = ragged_tensor_from_nested_numpy([np.zeros((graph['edge_attributes'].shape[0], 1)) for graph in graphs])
        y = (targets, ni, ei)
    
        # Now we can actually save these tensors into a persistent file
        with open(path, mode='wb') as file:
            data = (x, y)
            content = pickle.dumps(data)
            file.write(content)
    
    graphs = []
    index = 0
    chunk_index = 0
    time_start = time.time()
    keys = list(element_dict.keys())
    while len(element_dict) > 0:
        
        key = keys[index]
        data = element_dict[key]
        graph = processing.process(
            data[e.SMILES_COLUMN],
            graph_labels=data['targets'],
        )
        if len(graph['edge_indices']) > 1:
            graphs.append(graph)
        
        del element_dict[key]
        index += 1
        
        # logging the progress
        if index % 10_000 == 0:
            time_elapsed = time.time() - time_start
            time_per_element = time_elapsed / (index + 1)
            time_remaining = len(element_dict) * time_per_element
            eta = datetime.datetime.now() + datetime.timedelta(seconds=time_remaining)
            e.log(f' * processed ({index}/{num_elements})'
                  f' - elapsed time: {time_elapsed/60:.2f} min'
                  f' - remaining time: {time_remaining/60:.2f} min'
                  f' - eta: {eta:%a %H:%M}')
        
        # If the graph buffer has reached the required size, we then save all that data 
        # as a new chunk to the disk.
        if len(graphs) >= e.CHUNK_SIZE or len(element_dict) == 0:        
            path = os.path.join(train_path, f'chunk_{chunk_index:03d}.pkl')
            e.log(f' * saving chunk {chunk_index} @ {path}')
            
            # :hook save_chunk:
            #       This hook should implement the saving of list of graphs as a single chunk of 
            #       into the tensor cache. In this process, the tuple consisting of training 
            #       input "x" and output "y" data should be saved as a single pickled file.
            e.apply_hook(
                'save_chunk',
                path=path,
                graphs=graphs,
            )
            graphs = []
            chunk_index += 1

    chunk_files = os.listdir(train_path)
    e.log(f'saved {len(chunk_files)} chunks to the disk')


experiment.run_if_main()