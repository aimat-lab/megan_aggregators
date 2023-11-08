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
import typing as t

import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
import matplotlib.pyplot as plt
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path
from graph_attention_student.models.megan import Megan
from graph_attention_student.training import NoLoss
from graph_attention_student.data import tensors_from_graphs
from visual_graph_datasets.data import VisualGraphDatasetReader

from megan_aggregators.utils import NULL_LOGGER

# :param VISUAL GRAH_DATASET:
#       pass
VISUAL_GRAPH_DATASET: str = '~/.visual_graph_datasets/datasets/rb_dual_motifs'


class VgdGenerator():
    
    def __init__(self,
                 path: str,
                 indices: t.List[int],
                 epochs: int,
                 chunk_size: int = 100,
                 batch_size: int = 32,
                 logger: logging.Logger = NULL_LOGGER,
                 ):
        self.path = path
        self.indices = indices
        self.epochs = epochs
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.logger = logger
    
        # ~ derived properties
        # These properties are derived from the dataset
        self.reader = VisualGraphDatasetReader(logger=self.logger)
        self.length = len(self.indices)
    
        # ~ dynamic properties
        
        self.epoch = 0
        self.chunk_index = 0
        self.batch_index = 0
        self.index = 0
        # This data structure will save the index data map for the current chunk.
        self.index_data_map = {}
    
    def __next__(self):
        if self.epoch > self.epochs:
            raise StopIteration()
        
        # Now the first thing is: If there is no chunk currently loaded then we need to load the next chunk
        if self.batch_index >= self.chunk_size - 1:
            
            # A special case is if we are currently at the last chunk, then we need to load the first chunk again
            if self.chunk_index >= self.length - 1:
                self.logger.info('starting over at first chunk')
                self.chunk_index = 0
                random.shuffle(self.indices)
            
            self.logger.info('loading new chunk...')
            num_chunk = min(self.chunk_size, self.length - self.chunk_index)
            indices = self.indices[self.chunk_index:self.chunk_index+num_chunk]
            self.elements: t.List[dict] = self.reader.read_indices(indices)
            self.graphs: t.List[dict] = [data for data in self.index_data_map.values()]
            self.x = tensors_from_graphs(self.graphs)
            self.y = np.array([data['targets'] for data in self.index_data_map.values()])
            self.chunk_index += num_chunk
            self.batch_index = 0  # in a new chunk we need to reset the batch counting
            
        # Then inside a chunk we move on to the next batch, select all the graphs for that batch and then return 
        # the corresponding tensor representation
        num_chunk = min(self.chunk_size, self.length - self.chunk_index)
        num_batch = min(self.batch_size, num_chunk - self.batch_index)
        x_batch = [tens[self.batch_index:self.batch_index+num_batch] for tens in self.x]
        y_batch = self.y[self.batch_index:self.batch_index+num_batch]
        self.batch_index += num_batch
        
        # ! Have to change
        return x_batch, y_batch
    
    def __len__(self):
        return self.length
    
    def __iter__(self):
        return self


@Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)
def experiment(e: Experiment):
    e.log('starting experiment...')
    
    # ~ checking the dataset
    # In this first step we cant load the entire dataset into memory because it is too large, but we can at least check if 
    # the dataset even exists and is valid
    assert os.path.exists(e.VISUAL_GRAPH_DATASET), 'Given dataset path does not exist on the local system!'
    
    # ~ creating the model
    # At first we need to set up the model so that we can then use it for the training.

    @e.hook('create_model')
    def create_model():
        
        e.log('creating MEGAN model...')
        model = Megan(
            
        )
        model.compile(
            optimizer=ks.optimizers.Adam(learning_rate=1e-3),
            metrics=[ks.metrics.CategoricalAccuracy()],
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
    
    e.log('setting up the dataset generator...')
    VgdGenerator(
        path=e.VISUAL_GRAPH_DATASET,
        logger=e.logger,
    )
    
    
    
    
    