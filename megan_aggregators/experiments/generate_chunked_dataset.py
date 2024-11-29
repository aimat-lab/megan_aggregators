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
import rdkit.Chem as Chem
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

# :param SOURCE_PATH:
#       The path to the source CSV file that contains the SMILES strings and the corresponding
#       target labels for the dataset that is to be processed into a visual graph dataset.
SOURCE_PATH: str = os.path.join(EXPERIMENTS_PATH, 'assets', 'aggregators_new.csv')
#SOURCE_PATH = os.path.join(EXPERIMENTS_PATH, 'assets', 'aggregators_combined.csv')
# :param PROCESSING_PATH:
#       The path to the processing module that is used to process the SMILES into molecular 
#       graph representations that are then saved as visual graph datasets.
PROCESSING_PATH: str = os.path.join(EXPERIMENTS_PATH, 'assets', 'process.py')
# :param CLASS_0_KEY:
#       The column name in the dataset CSV file which contains the target values for the 0 (inactive)
#       class. 
CLASS_0_KEY: str = 'non-aggregator'
# :param CLASS_1_KEY:
#       The column name in the dataset CSV file which contains the target values for the 1 (active)
#       class.
CLASS_1_KEY: str = 'aggregator'
# :param DATASET_NAME:
#       The name of the dataset that is being processed. This is used to create the folder structure
#       for the dataset.
DATASET_NAME: str = 'aggregators'


# == PROCESSING PARAMETERS ==

# :param NUM_TEST: 
#       The number of elements to be sampled as the test set.
NUM_TEST: int = 1_000
# :param NUM_VAL:
#       The number of elements to be sampled as the validation set.
NUM_VAL: int = 200
# :param USE_DIMORPHITE:
#       Whether to use the dimorphite protonation for the molecules in the training dataset or not.
#       This is used as an augmentation to increase the training dataset size.
USE_DIMORPHITE: bool = True
# :param MIN_PH:
#       The minimum pH value used for the dimorphite protonation that is applied to every 
#       molecule in the training dataset as an augmentation to increase the training size.
MIN_PH: float = 6.4
# :param MAX_PH:
#       The maximum pH value used for the dimorphite protonation that is applied to every
#       molecule in the training dataset as an augmentation to increase the training size.
MAX_PH: float = 8.4
# :param MAX_VARIANTS:
#       The maximum number of protonation variants that are generated for each molecule in the
#       training dataset.
MAX_VARIANTS: int = 128
# :param PKA_PRECISION:
#       The precision of the pKa values that are used for the dimorphite protonation. This is
#       used to determine the number of protonation variants that are generated for each molecule.
PKA_PRECISION: float = 0.1
# :param USE_COORDINATES:
#       Whether to use the node coordinates in the graph dataset or not. If this is True RDKIT
#       will be used to generate simple equilibrium conformations/geometries for each molecule.
#       This will significantly increase the processing time.
USE_COORDINATES: bool = True
# :param CHUNK_SIZE:
#       The number of elements that are included in one chunk of the training dataset. The training 
#       dataset is directely exported as torch tensor files in the end. These are chunked to avoid 
#       memory issues during the processing and training on systems with limited memory.
CHUNK_SIZE: int = 250_000
# :param IMAGE_WIDTH:
#       The width of the image that is generated for each molecule in the visual graph dataset.
IMAGE_WIDTH: int = 1000
# :param IMAGE_HEIGHT:
#       The height of the image that is generated for each molecule in the visual graph dataset.
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
                 class_0_key: str,
                 class_1_key: str,
                 **kwargs,
                 ):
        super(ProcessingWorker, self).__init__(**kwargs)
        
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.use_coordinates = use_coordinates
        
        self.class_0_key = class_0_key
        self.class_1_key = class_1_key
        
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
                data[self.class_0_key],
                data[self.class_1_key],
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
    
    
@experiment.hook('sample_test_indices', default=True, replace=False)
def sample_test_indices(e: Experiment,
                        dataset: list[dict],
                        num_elements: int,
                        ) -> list[int]:
    
    target_indices_map: dict[int, list] = defaultdict(list)
    for index, data in dataset.items():
        target = int(np.argmax([data[e.CLASS_0_KEY], data[e.CLASS_1_KEY]]))
        target_indices_map[target].append(index)
        
    target_len_map: dict[int, int] = {k: len(v) for k, v in target_indices_map.items()}
    e.log('target len map: ' + str(target_len_map))
        
    test_indices = [
        *random.sample(target_indices_map[0], num_elements // 2),
        *random.sample(target_indices_map[1], num_elements // 2),
    ]
    return test_indices
    

@experiment.hook('create_test_set', default=True, replace=False)
def create_test_set(e: Experiment,
                    dataset: list[dict],
                    file_name: str = 'dataset_test',
                    num_elements: int = 1_000,
                    protonate: bool = False,
                    ) -> None:
    
    e.log('sampling test indices...')
    test_indices = e.apply_hook(
        'sample_test_indices',
        dataset=dataset,
        num_elements=num_elements,
    )
    
    # ~ protonation
    
    e.log('protonating test set...')
    dl = DimorphiteDL(
        min_ph=e.MIN_PH,
        max_ph=e.MAX_PH,
        pka_precision=e.PKA_PRECISION,
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
            e.CLASS_0_KEY: data[e.CLASS_0_KEY],
            e.CLASS_1_KEY: data[e.CLASS_1_KEY],
        }
            
        # We need to delete that element from the base dataset then to avoid leakage of test time 
        # information into the test dataset.
        if random.random() < 0.7:
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
        graph_labels = np.array([data[e.CLASS_0_KEY], data[e.CLASS_1_KEY]], dtype=float)
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
            class_0_key=e.CLASS_0_KEY,
            class_1_key=e.CLASS_1_KEY,
            daemon=True,
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
   
    # To test this experiment we mainly have to reduce the runtime of all the processing steps
    # which means that we reduce the number of test/val elements as well as the dataset size 
    # in general through subsampling such that the entire code should be run much more quickly.
    if e.__TESTING__:
        
        e.log('testing mode - scaling down...')
        
        e.NUM_TEST = 100
        e.NUM_VAL = 100
        
        e.CHUNK_SIZE = 10_000
        
        dataset_ = dict(random.sample(list(dataset.items()), 10_000))
        dataset = dataset_
                
    e.log(f'loaded {len(dataset)} elements...')
    
    # ~ filtering data
    
    e.log('filtering data...')
    
    for index, data in dataset.items():
        
        if '.' in data['smiles']:
            del dataset[index]
            e.log(f' * removing element {index} due to disconnection - {data["smiles"]}')
            
        if not Chem.MolFromSmiles(data['smiles']):
            del dataset[index]
            e.log(f' * removing element {index} due to invalid SMILES - {data["smiles"]}')
    
    # ~ determine test set
    # We want to sample the test set from the un-processed dataset because this spares us the hassle 
    # of having to clean up the processed dataset afterwards regarding all the base molecules that 
    # were used in the test set.
    e.log(f'sampling test set with {e.NUM_TEST} elements...')
    dataset_test: dict[int, dict] = e.apply_hook(
        'create_test_set', 
        dataset=dataset,
        file_name='dataset_test',
        num_elements=e.NUM_TEST,
    )

    # ~ determine validation set
    # We also want to sample a validation set from the remaining dataset, for which we can use the same 
    # function as we want the validation and test set to be essentially the same structure - just different 
    # elements.
    e.log(f'sampling validation set with {e.NUM_VAL} elements...')
    dataset_val: dict[int, dict] = e.apply_hook(
        'create_test_set', 
        dataset=dataset,
        file_name='dataset_val',
        num_elements=e.NUM_VAL,
    )
    
    e.log(f'train elements: {len(dataset)} - test elements: {len(dataset_test)} - val elements: {len(dataset_val)}...')

    # ~ protonation pre-processing
    
    e.log('protonating with dimorphite...')
    
    e.apply_hook(
        'before_protonation',
        dataset=dataset
    )
    
    # 04.08.2021: Added the option to disable the protonation pre-processing.
    if e.USE_DIMORPHITE:
        
        dl = DimorphiteDL(
            min_ph=e.MIN_PH,
            max_ph=e.MAX_PH,
            max_variants=e.MAX_VARIANTS,
            pka_precision=e.PKA_PRECISION,
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
                    e.CLASS_0_KEY: data[e.CLASS_0_KEY],
                    e.CLASS_1_KEY: data[e.CLASS_1_KEY],
                }
                index += 1
                
            if c % 10_000 == 0:
                e.log(f' * {c}/{len(dataset)} processed - {index} protonated')
                
    else:
        
        e.log('skipping protonation!')
        dataset_protonated = dataset
            
    e.log(f'protonated dataset has {len(dataset_protonated)} elements...')
    
    e.log('saving protonated dataset CSV...')
    dataset_protonated_path = os.path.join(e.path, 'dataset_protonated.csv')
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
    indices = list(dataset_protonated.keys())
    target_indices_map: dict[int, list] = defaultdict(list)
    for index, data in dataset_protonated.items():
        target = int(np.argmax([data[e.CLASS_0_KEY], data[e.CLASS_1_KEY]]))
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