import os
import csv
import time
import random
import math
import datetime
import queue
import multiprocessing
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

import torch
import msgpack
import numpy as np
import matplotlib.pyplot as plt
from dimorphite_dl import DimorphiteDL

from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path
from visual_graph_datasets.util import dynamic_import
from visual_graph_datasets.data import VisualGraphDatasetWriter
from visual_graph_datasets.processing.molecules import MoleculeProcessing
from graph_attention_student.torch.data import data_list_from_graphs
from megan_aggregators.utils import EXPERIMENTS_PATH
from megan_aggregators.data import default, ext_hook

# == SOURCE PARAMETERS == 

#SOURCE_PATH: str = os.path.join(EXPERIMENTS_PATH, 'assets', 'aggregators_new.csv')
SOURCE_PATH = os.path.join(EXPERIMENTS_PATH, 'assets', 'aggregators_binary.csv')
PROCESSING_PATH: str = os.path.join(EXPERIMENTS_PATH, 'assets', 'process.py')

# == PROCESSING PARAMETERS ==

NUM_TEST: int = 2_000
MIN_PH: float = 6.4
MAX_PH: float = 8.4
MAX_VARIANTS: int = 128
USE_COORDINATES: bool = True
CHUNK_SIZE: int = 250_000
IMAGE_WIDTH: int = 1000
IMAGE_HEIGHT: int = 1000


__TESTING__ = False
__DEBUG__ = True

experiment = Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)

class ProcessingWorker(multiprocessing.Process):
    
    def __init__(self,
                 input_queue: multiprocessing.Queue,
                 output_queue: multiprocessing.Queue,
                 processing_path: str,
                 use_coordinates: bool,
                 ):
        super(ProcessingWorker, self).__init__()
        
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.use_coordinates = use_coordinates
        
        module = dynamic_import(processing_path)
        self.processing = module.processing
        
    def run(self):
        
        for data in iter(self.input_queue.get, None):
            
            try:
                
                smiles = data['smiles']
                graph = self.processing.process(
                    smiles,
                    #use_node_coordinates=False,
                )
                # unfortunately there are rare cases where the processing results in infinity values for 
                # the node attributes. This would cause serious problems further down the line for the 
                # machine learning models, which is why we quick fix it here by setting those to zero instead.
                if np.isnan(graph['node_attributes']).any() or np.isnan(graph['edge_attributes']).any():
                    print('nan values in graph')
                    continue
                
                if np.isinf(graph['node_attributes']).any() or np.isinf(graph['edge_attributes']).any():
                    print('inf values in graph')
                    continue
                
                if 'num_variants' in data:
                    graph['graph_weight'] = 1.0 / data['num_variants']
                
            except Exception as exc:
                print('exception in processing:', exc)
                continue
            
            graph_labels = np.array([
                data['non-aggregator'],
                data['aggregator'],
            ], dtype=float)
            graph['graph_labels'] = graph_labels
            
            output = msgpack.packb(graph, default=default)
            self.output_queue.put(output)


@experiment.hook('save_chunk', default=True, replace=False)
def save_chunk(e: Experiment,
               graphs: list[dict],
               index: int,
               folder_path: str,
               ) -> None:
    
    e.log(f' * saving chunk {index}...')
    e.log(f'   converting to data list...')
    data_list = data_list_from_graphs(graphs)
    chunk_path = os.path.join(folder_path, f'chunk_{index:03d}.pt')
    e.log(f'   saving file...')
    torch.save(data_list, chunk_path)
    
    e.log(f'   done.')
    del data_list

@experiment.hook('create_test_set', default=True, replace=False)
def create_test_set(e: Experiment,
                    dataset: list[dict],
                    file_name: str = 'dataset_test',
                    protonate: bool = False,
                    ) -> None:
    
    target_indices_map: dict[int, list] = defaultdict(list)
    for index, data in dataset.items():
        target = int(np.argmax([data['non-aggregator'], data['aggregator']]))
        target_indices_map[target].append(index)
        
    test_indices = [
        *random.sample(target_indices_map[0], NUM_TEST // 2),
        *random.sample(target_indices_map[1], NUM_TEST // 2),
    ]
    e.log(f'sampled {len(test_indices)} test indices...')
    
    # ~ protonation
    
    e.log('protonating test set...')
    dl = DimorphiteDL(
        min_ph=e.MIN_PH,
        max_ph=e.MAX_PH,
        #max_variants=1,
        pka_precision=0.1,
    )
    
    dataset_test: dict[int, dict] = {}
    for index in test_indices:
        data = dataset[index]
        smiles = data['smiles']
        if protonate:
            smiles = dl.protonate(data['smiles'])[0:1]
            
        dataset_test[index] = {
            'index': index,
            'smiles': smiles,
            'smiles_base': data['smiles'],
            'non-aggregator': data['non-aggregator'],
            'aggregator': data['aggregator'],
        }
            
        # We need to delete that element from the base dataset then to avoid leakage of test time 
        # information into the test dataset.
        del dataset[index]
            
    # ~ saving as csv
    e.log('saving test dataset as csv...')
    dataset_test_path = os.path.join(e.path, f'{file_name}.csv')
    with open(dataset_test_path, 'w') as file:
        writer = csv.DictWriter(file, fieldnames=next(iter(dataset_test.values())).keys())
        writer.writeheader()
        writer.writerows(dataset_test.values())

    # ~ saving as vgd
    e.log('saving test set as visual graph dataset...')
    dataset_folder_path = os.path.join(e.path, file_name)
    os.makedirs(dataset_folder_path, exist_ok=True)
    processing: MoleculeProcessing = dynamic_import(e.PROCESSING_PATH).processing
    writer = VisualGraphDatasetWriter(
        path=dataset_folder_path,
        chunk_size=10_000,
    )
    for index, data in dataset_test.items():
        graph_labels = np.array([data['non-aggregator'], data['aggregator']], dtype=float)
        graph = processing.process(data['smiles'])
        graph['graph_labels'] = graph_labels
        
        fig, node_positions = processing.visualize_as_figure(
            data['smiles'], 
            width=e.IMAGE_WIDTH, 
            height=e.IMAGE_HEIGHT
        )
        graph['node_positions'] = node_positions
        
        metadata = {
            'index': index,
            'smiles': data['smiles'],
            'smiles_base': data['smiles_base'],
            'target': graph_labels,
            'graph': graph,
        }
        
        writer.write(
            name=str(index),
            metadata=metadata,
            figure=fig,
        )

    return dataset_test

@experiment
def experiment(e: Experiment): 
    
    e.log('starting experiment...')
    
    e.log('creating dataset folder...')
    dest_path = os.path.join(e.path, 'dataset')
    os.makedirs(dest_path, exist_ok=True)
    
    # ~ initializing parallel processing
    
    e.log('creating multiprocessing queues...')
    input_queue = multiprocessing.Queue(maxsize=1000)
    output_queue = multiprocessing.Queue(maxsize=1000)
    
    e.log('creating multiprocessing processes...')
    workers = []
    for _ in range(os.cpu_count()):
        worker = ProcessingWorker(
            input_queue=input_queue,
            output_queue=output_queue,
            processing_path=e.PROCESSING_PATH,
            use_coordinates=e.USE_COORDINATES,
        )
        worker.start()
        workers.append(worker)
        e.log(f' * started worker {_}')
        
    
    # ~ data loading
    
    e.log('loading data...')
    dataset = {}
    with open(SOURCE_PATH) as file:
        reader = csv.DictReader(file)
        for index, data in enumerate(reader):
            data['index'] = index
            dataset[index] = data
   
    if e.__TESTING__:
        e.log('testing mode - scaling down...')
        
        e.NUM_TEST = 200
        e.CHUNK_SIZE = 10_000
        
        dataset_ = dict(random.sample(list(dataset.items()), 10_000))
        dataset = dataset_
                
    e.log(f'loaded {len(dataset)} elements...')
    
    # ~ determine test set
    # We want to sample the test set from the un-processed dataset because this spares us the hassle 
    # of having to clean up the processed dataset afterwards regarding all the base molecules that 
    # were used in the test set.
    e.log('sampling test set...')
    dataset_test: dict[int, dict] = e.apply_hook(
        'create_test_set', 
        dataset=dataset,
        file_name='dataset_test',
    )

    e.log('sampling validation set...')
    dataset_val: dict[int, dict] = e.apply_hook(
        'create_test_set', 
        dataset=dataset,
        file_name='dataset_val',
    )
    
    e.log(f'train elements: {len(dataset)} - test elements: {len(dataset_test)} - val elements: {len(dataset_val)}...')

    # ~ protonation pre-processing
    
    e.log('protonating with dimorphite...')
    dl = DimorphiteDL(
        min_ph=e.MIN_PH,
        max_ph=e.MAX_PH,
        max_variants=e.MAX_VARIANTS,
        pka_precision=0.1,
    )
    
    index = 0
    dataset_protonated: dict[int, dict] = {}
    for c, data in enumerate(dataset.values()):
        smiles = data['smiles']
        smiles_variants = dl.protonate(smiles)
        num_variants = len(smiles_variants)
        
        for smiles_protonated in smiles_variants:
            dataset_protonated[index] = {
                'index': index,
                'index_base': data['index'],
                'smiles': smiles_protonated,
                'smiles_base': smiles,
                'num_variants': num_variants,
                'non-aggregator': data['non-aggregator'],
                'aggregator': data['aggregator'],
            }
            index += 1
            
        if c % 10_000 == 0:
            e.log(f' * {c}/{len(dataset)} processed')
            
    e.log(f'protonated dataset has {len(dataset_protonated)} elements...')
    
    e.log('saving protonated dataset CSV...')
    dataset_protonated_path = os.path.join(e.path, 'aggregators_protonated.csv')
    with open(dataset_protonated_path, 'w') as file:
        writer = csv.DictWriter(file, fieldnames=next(iter(dataset_protonated.values())).keys())
        writer.writeheader()
        writer.writerows(list(dataset_protonated.values()))
        
    # ~ oversampling
    # The aggregator dataset is a severly imbalanced dataset. There are a lot more non-aggs than aggs.
    # To address this problem we are going to oversample the minority labels such that there is approximately 
    # the same amount of non-aggs and aggs.
    e.log('oversampling minority labels...')
    # for that we are first going to construct this data structure whose keys are the target labels 
    # 0 (nonagg) / 1 (agg) and the values are lists with the corresponding indices in the dataset.
    indices = list(range(len(dataset_protonated)))
    target_indices_map: dict[int, list] = defaultdict(list)
    for index, data in dataset_protonated.items():
        target = int(np.argmax([data['non-aggregator'], data['aggregator']]))
        target_indices_map[target].append(index)
        
    # Now we can calculate the necessary oversampling factor like this:
    oversampling_factor = len(target_indices_map[0]) / len(target_indices_map[1])
    e.log(f'oversampling factor: {oversampling_factor} (0: {len(target_indices_map[0])} / 1: {len(target_indices_map[1])})')
        
    dataset_oversampled: dict[int, dict] = {}
    for c, index in enumerate(target_indices_map[0] + target_indices_map[1] * math.floor(oversampling_factor)):
        dataset_oversampled[c] = dataset_protonated[index]
        
    e.log(f'ovesampled dataset has {len(dataset_oversampled)} elements...')
        
    # ~ graph dataset
    
    e.log('processing dataset to graphs...')
    
    # Before processing we want to shuffle the dataset because we want the target labels to be equally 
    # distributed for the different chunks. It would be bad if we had chunks that only consisted of one 
    # type of target label.
    indices = list(dataset_oversampled.keys())
    num_elements = len(indices)
    random.shuffle(indices)
    
    count = 0
    prev_count = 0
    
    chunk_index = 0
    graphs_chunk: list[dict] = []
    start_time = time.time()
    while len(indices) != 0 or not input_queue.empty() or not output_queue.empty():
        
        while not input_queue.full() and len(indices) > 0:
            index = indices.pop()
            data = dataset_oversampled[index]
            input_queue.put(data)
            
        try:
            while True:
                output = output_queue.get(False)
                graph = msgpack.unpackb(output, ext_hook=ext_hook)
                graphs_chunk.append(graph)
                count += 1
            
        except queue.Empty:
            pass
    
        if len(graphs_chunk) > e.CHUNK_SIZE:
            e.apply_hook(
                'save_chunk',
                graphs=graphs_chunk,
                index=chunk_index,
                folder_path=dest_path,
            )
            del graphs_chunk
            graphs_chunk = []
            chunk_index += 1
            
        if count % 5_000 == 0 and count != prev_count:
            prev_count = count
            duration = time.time() - start_time
            num_remaining = num_elements - (count + 1)
            time_per_element = duration / (count + 1)
            remaining_time = time_per_element * num_remaining
            eta = datetime.datetime.now() + datetime.timedelta(seconds=remaining_time)
            e.log(f' * ({count:05d}/{num_elements})'
                  f' - remaining time: {remaining_time:.2f} s'
                  f' - eta: {eta:%a %d.%m %H:%M}')
            
    e.apply_hook(
        'save_chunk',
        graphs=graphs_chunk,
        index=chunk_index,
        folder_path=dest_path,
    )
            
    # ~ closing workers
    for worker in workers:
        input_queue.put(None)
        input_queue.put(None)
        worker.terminate()
        worker.join()
        
    e.log('done.')
        
experiment.run_if_main()