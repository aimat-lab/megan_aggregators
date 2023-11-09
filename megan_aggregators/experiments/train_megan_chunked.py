"""
This experiment trains a MEGAN model to predicting a given molecule as either an aggregator or a non-aggregator. 

The main difference with this experiment is that it is specifically designed to support the training with a very large 
dataset. Using such a very large dataset creates various challanges. For example the dataset is considered so large that 
it will not fit into RAM memory of most conventional computing systems. This means that the dataset will have to be loaded 
into memory only in subsets / chunks which have to be processed sequentially.
"""
import os
import random
import pathlib
import logging
import pickle
import typing as t
from collections import Counter

import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
import matplotlib.pyplot as plt
from kgcnn.data.utils import ragged_tensor_from_nested_numpy
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path
from graph_attention_student.models.megan import Megan
from graph_attention_student.training import NoLoss
from graph_attention_student.data import tensors_from_graphs
from visual_graph_datasets.data import VisualGraphDatasetReader

from megan_aggregators.utils import NULL_LOGGER

PATH = pathlib.Path(__file__).parent.absolute()
CACHE_PATH = os.path.join(PATH, 'cache')

# == DATASET PARAMETERS ==
# These are the parameters related to the dataset and the processing thereof.

# :param VISUAL GRAH_DATASET:
#       This has to be the absolute string path pointing towards the visual graph dataset to be used for the training 
#       of the model.
VISUAL_GRAPH_DATASET: str = '/home/jonas/.visual_graph_datasets/datasets/rb_adv_motifs'
# :param NUM_TEST:
#       This is the number of elements to sample for the test set on which the performance of the model will be 
#       evaluated
NUM_TEST: int = 1000
# :param CHUNK_SIZE:
#       This is the size of the chunks into which the dataset will be split during the pre-processing. So this is 
#       the number of elements that will be loaded into the actual memory at the same time. Therefore this number 
#       has to be chosen appropriately for the available hardware.
CHUNK_SIZE: int = 1000
# :param OVERSAMPLING_FACTORS:
#       This is a dictionary where the keys are the integer identifiers for the possible classes of the dataset
#       and the associated values are the corresponding oversampling factors to be applied to the elements of that 
#       class. An oversampling factor of 5 for example means that every element of that class will be duplicated 5 
#       times as part of the dataset. In this specific case, the oversampling is generally necessary because the 
#       underlying dataset is highly imbalanced (meaning that one class appears significantly more often than the other)
OVERSAMPLING_FACTORS: dict = {
    0: 1,   # non-aggregator
    1: 2,  # aggregator
}

# == TRAINING PARAMETERS ==
# These are the parameters that are relevant to the training process itself, so for example the number of epochs or 
# the batch size.

# :param BATCH_SIZE:
#       The number of elements to use in one batch during training
BATCH_SIZE: int = 32
# :param EPOCHS:
#       This is the number of epochs to train the model for
EPOCHS: int = 2

# == MODEL PARAMETERS ==
# These are the parameters that control the setup and configuration of the MEGAN model architecture to be trained.

IMPORTANCE_FACTOR: float = 0.0
IMPORTANCE_CHANNELS: int = 2
UNITS: t.List[int] = [32, 32, 32]
FINAL_UNITS: t.List[int] = [32, 16, 2]


__DEBUG__ = True


@Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)
def experiment(e: Experiment):
    e.log('starting experiment...')
    
    # ~ pre-processing the dataset
    # ---
    
    assert os.path.exists(e.VISUAL_GRAPH_DATASET), 'Given dataset path does not exist on the local system!'
    
    e.log('setting up the dataser reader...')
    reader = VisualGraphDatasetReader(
        path=e.VISUAL_GRAPH_DATASET,
        logger=e.logger,    
    )
    dataset_length = len(reader)
    e.log(f'processing a dataset with {dataset_length} elements')
    
    @e.hook('save_cache_graphs')
    def save_cache_graphs(e: Experiment,
                          graphs: t.List[dict],
                          path: str,
                          ):

        # Then we serialize and save the input. The network input consists of three different tensors which provide the 
        # information about the graph nodes, edges and edge index lists.
        # We need to save these three tensors into distinct files.
        x = tensors_from_graphs(graphs)
        with open(os.path.join(path, 'x.pkl'), mode='wb') as file:
            pickle.dump(x, file)
    
        # The corresponding target labels for those graphs we actually have to save as a numpy array and not a tensor!
        targets = np.array([graph['graph_labels'] for graph in graphs], dtype=np.float32)
        ni = ragged_tensor_from_nested_numpy([np.zeros((graph['node_attributes'].shape[1], e.IMPORTANCE_CHANNELS)) for graph in graphs])
        ei = ragged_tensor_from_nested_numpy([np.zeros((graph['edge_attributes'].shape[1], e.IMPORTANCE_CHANNELS)) for graph in graphs])
        y = (targets, ni, ei)
        with open(os.path.join(path, 'y.pkl'), mode='wb') as file:
            pickle.dump(y, file)
        
    @e.hook('load_cache_graphs')
    def load_cache_graphs(e: Experiment,
                          path: str,
                          ):
        
        with open(os.path.join(path, 'x.pkl'), mode='rb') as file:
            x = pickle.load(file)
    
        with open(os.path.join(path, 'y.pkl'), mode='rb') as file:
            y = pickle.load(file)
    
        return x, y
    
    # We save all the raw tensor data in a custom folder in the overall experiment cache folder. However, to make this more 
    # efficient we only actually perform the whole dataset pre-processing if that folder does not exist yet. If it does exist 
    # we are simply going to re-use that.
    cache_path = os.path.join(CACHE_PATH, f'train_megan_chunked__{e.name}__cache')
    # If ...
    test_path = os.path.join(CACHE_PATH, f'train_megan_chunked__{e.name}__test')
    e.log(f'cache path: {cache_path} - test path: {test_path}')
    if not os.path.exists(cache_path):
        
        os.mkdir(cache_path)
        
        # the test set
        # sampling the test set is actually a bit of a problem here. Certainly the easiest option would be to simple take the 
        # first or last N elements of the dataset in general as the test set. However, in this dataset we can expect that there 
        # is an ordering bias - which means that it is not inherently shuffled enough to get a representative sample like this.
        # Now the alternative we are going to do instead: We are going to sample a certain number of elements from each of the 
        # dataset chunks randomly to make up the test set.
        test_data_map: t.Dict[int, dict] = {}
        num_chunks = len(reader.chunk_map)
        num_test_per_chunk = int(e.NUM_TEST / num_chunks)
        
        index_class_map = {}
        # This will be a temporary dict structure where the keys are the global integer indices of the elements and the the values 
        # are the actual data elements.
        train_data_map: t.Dict[int, dict] = {}
        chunk_index = 0
        for i, data_map in enumerate(reader.chunk_iterator()):
            e.log(f'processing dataset chunk {i} with {len(data_map)} elements')
            
            # As the first step we extract a number of the elements for the test set.
            for index in random.sample(list(data_map.keys()), k=num_test_per_chunk):
                test_data_map[index] = data_map[index]
                del data_map[index]
            
            # Only afterwards we aggregate the remaining elements to the training dict 
            train_data_map.update(data_map)
            index_class_map.update({index: np.argmax(data['metadata']['graph']['graph_labels']) for index, data in data_map.items()})
            
            # If the collected data is now bigger than the selected chunk size then we actually process all of that data into 
            # tensors and then save all of those tensors as one big chunk to the cache folder.
            if len(train_data_map) >= e.CHUNK_SIZE:
                
                while len(train_data_map) >= e.CHUNK_SIZE:
                    
                    graphs = []
                    for index, data in list(train_data_map.items())[:e.CHUNK_SIZE]:
                        graph = data['metadata']['graph']
                        target = np.argmax(graph['graph_labels'])
                        graphs += [graph] * e.OVERSAMPLING_FACTORS[target]
                        
                        del train_data_map[index]
                                        
                    e.log(f'saving tensor cache {chunk_index} with {len(graphs)} elements...')
                    path = os.path.join(cache_path, f'{chunk_index:02d}')
                    os.mkdir(path)
                    
                    e.apply_hook(
                        'save_cache_graphs',
                        graphs=graphs,
                        path=path,
                    )
                    chunk_index += 1
                    graphs = []
                
        print(Counter(index_class_map.values()))
        
    # IF the cache folder does already exist we actually have to do the thing where the
    else:
        pass
        
    indices = list(range(dataset_length))
    test_indices = list(test_data_map.keys())
    train_indices = list(set(indices).difference(set(test_indices)))
    e['test_indices'] = test_indices
    e['train_indices'] = train_indices
    
    # ~ creating the model
    # At first we need to set up the model so that we can then use it for the training.

    @e.hook('create_model')
    def create_model(e: Experiment):
        
        e.log('creating MEGAN model...')
        model = Megan(
            units=e.UNITS,
            importance_factor=e.IMPORTANCE_FACTOR,
            importance_channels=e.IMPORTANCE_CHANNELS,
            final_units=e.FINAL_UNITS,
            final_activation='linear',
            concat_heads=False,
            use_bias=True,
            use_edge_features=True,
        )
        model.compile(
            optimizer=ks.optimizers.Adam(learning_rate=1e-3),
            # metrics=[ks.metrics.CategoricalAccuracy()],
            loss=[
                ks.losses.CategoricalCrossentropy(from_logits=True),
                NoLoss(),
                NoLoss(),
            ],
            loss_weights=[1, 0, 0],
            run_eagerly=False,
        )
        
        return model

    e.log('creating model...')
    model = e.apply_hook(
        'create_model',
    )

    # ~ training the model
    
    def training_generator(epochs: int,
                           batch_size: int,
                           ):
        
        chunk_paths = [path for name in os.listdir(cache_path) if (path := os.path.join(cache_path, name)) and os.path.isdir(path)]
        
        epoch_index = 0
        while epoch_index < epochs:
            
            e.log(f'starting epoch {epoch_index}...')
            for chunk_index, path in enumerate(chunk_paths):
                
                e.log(f'loading chunk {chunk_index} for training')
                x_chunk, y_chunk = e.apply_hook(
                    'load_cache_graphs',
                    path=path
                )
                chunk_size = len(y_chunk[0])

                batch_index = 0
                while batch_index < chunk_size:
                    num_batch = min(batch_size, chunk_size - batch_index)
                    x_batch = [v[batch_index:batch_index+num_batch] for v in x_chunk]
                    y_batch = [v[batch_index:batch_index+num_batch] for v in y_chunk]
                                        
                    batch_index += num_batch
                    # print(batch_index, x_batch, y_batch)
                    yield x_batch, y_batch
                    
            epoch_index += 1
                    
    
    e.log('setting up the dataset generator...')
    generator = training_generator(e.EPOCHS, e.BATCH_SIZE)
    model.fit(
        generator,
    )
    
    # ~ evaluating the model
    # after the training is done we can then evaluate the model on the test set.
    e.log(f'evaluating the model on {len(test_data_map)} elements...')
    indices_test = list(test_data_map.keys())
    graphs_test = [data['metadata']['graph'] for data in test_data_map.values()]
    
    predictions = model.predict_graphs(graphs_test)
    for index, graph, (out, ni, ei) in zip(indices_test, graphs_test, predictions):
        e[f'out/pred/{index}'] = out
        e[f'out/true/{index}'] = graph['graph_labels']
        
        e[f'ni/{index}'] = ni
        e[f'ei/{index}'] = ei
        
        
@experiment.analysis
def analysis(e: Experiment):
    
    e.log('starting the analysis...')
    
    
experiment.run_if_main()
    
    
    