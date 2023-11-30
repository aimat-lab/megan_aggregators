"""
This experiment evaluates an ensemble of MEGAN models on a visual graph dataset that 
is specified through the model parameters.

**VARIABLE NAMES**

In this file the following variable names are used:
- B: the number of graphs in the evaluation set
- H: the number of models in the ensemble
"""
import os
import json
import pathlib
import typing as t

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from scipy.special import softmax
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace
from visual_graph_datasets.config import Config
from visual_graph_datasets.web import ensure_dataset
from visual_graph_datasets.data import VisualGraphDatasetReader
from graph_attention_student.models import Megan2
from graph_attention_student.models import load_model

from megan_aggregators.utils import ASSETS_PATH
from megan_aggregators.models import ModelEnsemble


PATH = pathlib.Path(__file__).parent.absolute()

# == DATASET PARAMETERS ==
# These are the parameters related to the model files

VISUAL_GRAPH_DATASET: str = os.path.join(PATH, 'cache', 'aggregators_binary_protonated', 'test')
TEST_INDICES_PATH: t.Optional[str] = None

# == MODEL PARAMETERS ==

MODEL_PATHS: t.List[str] = [
    os.path.join(ASSETS_PATH, 'models', 'model_0'),
    os.path.join(ASSETS_PATH, 'models', 'model_1'),
    os.path.join(ASSETS_PATH, 'models', 'model_2'),
    os.path.join(ASSETS_PATH, 'models', 'model_3'),
    os.path.join(ASSETS_PATH, 'models', 'model_4'),
]


@Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)
def experiment(e: Experiment):
    e.log('starting experiment...')
    
    # ~ loading dataset
    # At first we need to load the dataset from the disk. This dataset is expected to be a 
    # "visual graph dataset". In this format, the dataset is represented as a folder.

    config = Config()
    config.load()

    if os.path.exists(e.VISUAL_GRAPH_DATASET):
        dataset_path = e.VISUAL_GRAPH_DATASET
    else: 
        dataset_path = ensure_dataset(
            dataset_name=e.VISUAL_GRAPH_DATASET,
            config=config,
            logger=e.logger,
        )
           
    e.log('loading the dataset...')
    reader = VisualGraphDatasetReader(
        path=dataset_path,
        logger=e.logger,
    )
    index_data_map = reader.read()
    num_elements = len(index_data_map)
    e.log(f'loaded dataset with {num_elements} elements')
    
    # ~ the test split
    # Now that we have loaded the dataset into the memory we know take the test elements 
    # from that dataset.
    # This test split is optionally defined by the TEST_INDICES_PATH. This can be an absolute 
    # path pointing to a JSON file containing the list of test indices. This value can also be 
    # None, in which case we simply use the whole dataset for the evaluation.
    
    if e.TEST_INDICES_PATH is not None:
        e.log(f'loading test indices from {e.TEST_INDICES_PATH}')
        with open(e.TEST_INDICES_PATH, mode='r') as file:
            content = file.read()
            indices = json.loads(content)
            
    else:
        indices = list(index_data_map.keys())
        e.log(f'no test indices specified, using the entire dataset')
    
    graphs = [index_data_map[index]['metadata']['graph'] for index in indices]
    e.log(f'using {len(graphs)} elements for evaluation')
    
    # ~ loading the ensemble
    # 
    
    e.log('loading the models ans constructing the ensemble...')
    models = []
    for path in e.MODEL_PATHS:
        model = load_model(path)
        models.append(model)
    
    e.log(f'loaded {len(models)} models')
    
    ensemble = ModelEnsemble(models)
    e.log(f'constructed the ensemble')

    # Now that we have the model ensemble we can use it to make the predictions for all the 
    # graphs in the evaluation dataset.
    # The "predict_graph" method returns a list of numpy arrays where each array represents the 
    # output predictions for one of the graphs for each of the models from the ensemble (-1 dimension)
    # We know aggregate this by doing a consensus decision and we can also calculate the model 
    # uncertainty.
    e.log('aggregating the individual predictions...')
    predictions: t.List[np.ndarray] = ensemble.predict_graphs()
    
    consensuses: t.List[int] = []
    uncertainties = t.List[int] = []
    for index, graph, pred in zip(index, graphs, predictions):
        # pred: (B, H)
        consensus = np.mean(pred, axis=-1)
        consensuses.append(consensus)
        
        uncertainty = np.std(pred, axis=-1) 
        uncertainties.append(uncertainty)
        
        
    values_true = [arr[1] for arr in consensuses]
    values_pred = [index_data_map['metadata']['target'][1] for index in indices]
    
    # ~ calculating evaluation metrics
    acc_value = accuracy_score(values_true, values_pred)
    e.log(f'evaluation results:')
    e.log(f' * acc: {acc_value:.3f}')
            
experiment.run_if_main()