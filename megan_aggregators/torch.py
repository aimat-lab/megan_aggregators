import os
import pathlib

import numpy as np
from scipy.special import softmax
from graph_attention_student.torch.megan import Megan

from megan_aggregators.utils import load_processing
from megan_aggregators.utils import generate_counterfactuals_with_model
from megan_aggregators.utils import EXPERIMENTS_PATH
from megan_aggregators.utils import ASSETS_PATH
from megan_aggregators.utils import MODEL_FOLDER_PATH, MODEL_CHECKPOINT_PATH


def load_model(model_path: str = MODEL_CHECKPOINT_PATH
               ) -> Megan:
    """
    This function will load the model which is persistently saved on the disk using the given 
    absolute ``model_path`` to the checkpoint file.
    
    This function defines the path to the currently best performing model as the default model 
    path. Therefore, it is possible to invoke this function without any arguments to load the 
    default model.
    
    :param model_path: The absolute path to the checkpoint file of the model that is to be loaded.
        Default is the currently best performing model.
        
    :return: The loaded model.
    """
    model = Megan.load_from_checkpoint(model_path)
    model.eval()
    
    return model


def predict_aggregator(smiles: str,
                       model_path: str = MODEL_CHECKPOINT_PATH,
                       ) -> float:
    """
    Given the ``smiles`` string representation of a molecule, this function will predict its 
    probability of being an aggregator according to the model loaded from the given ``model_path``.
    
    This prediction is done by first processing the SMILES string into a graph representation using 
    the loaded Processing instance, then loading the model from the given path and performing a 
    forward pass of the the model using the graph representation of the molecule.
    
    **PREDICTING BINARY LABEL**
    
    The function returns the float probability of the molecule being an aggregator as a value 
    between 0 and 1. To get a binary label, the user can apply a threshold of 0.5 to the returned
    probability.
    
    .. code-block:: python
    
        probab: float = predict_aggregator(smiles)
        label: int = 1 if probab > 0.5 else 0
        label: str = 'aggregator' if label == 1 else 'non-aggregator'
    
    :returns: The float probability of the molecule being an aggregator.
    """
    # ~ processing the SMILES into graph
    # The first thing we have to do is to process the SMILES string into a graph representation
    # this is done by the processing instance.
    processing = load_processing()
    graph = processing.process(smiles)
    
    # ~ loading the model
    model = load_model(model_path)
    
    # ~ making the prediction
    # model.forward_graphs expects a list of graphs, so we have to wrap the graph into a list.
    # The function will return a list of dictionaries with the keys being the names of different 
    # properties that are computed by the forward pass of the model - the predicted target value 
    # being only one of them.
    info: dict[str, np.ndarray] = model.forward_graphs([graph])[0]
    
    # The model predicts the classification LOGITS, so we have to apply the softmax function to
    # get the probabilities.
    out_pred = softmax(info['graph_output'])
    probab: float = float(out_pred[0])
    
    return probab


def explain_aggregation(smiles: str,
                        model_path: str = MODEL_CHECKPOINT_PATH,
                        show_plot: bool = False,
                        ) -> tuple[dict, np.ndarray, np.ndarray]:
    # ~ processing the SMILES into graph
    # The first thing we have to do is to process the SMILES string into a graph representation
    # this is done by the processing instance.
    processing = load_processing()
    graph = processing.process(smiles)
    graph['graph_smiles'] = smiles
    
    # ~ loading the model
    model = load_model(model_path)
    
    # ~ making the prediction
    # model.forward_graphs expects a list of graphs, so we have to wrap the graph into a list.
    # The function will return a list of dictionaries with the keys being the names of different 
    # properties that are computed by the forward pass of the model.
    info: dict[str, np.ndarray] = model.forward_graphs([graph])[0]
    
    # These are arrays containing values between 0 and 1 which are assigned to each node / edge 
    # of the original input graph, indicating the importance of the node / edge for the prediction.
    # For each element, two values are assigned  representing the importance for the two classes 
    # aggregator and non-aggregator.
    # node_importances: (num_nodes, 2)
    node_importances = info['node_importance']
    # edge_importances: (num_edges, 2)
    edge_importances = info['edge_importance']

    return graph, node_importances, edge_importances
    

def generate_counterfactuals(smiles: str,
                             num: int,
                             model_path: str = MODEL_CHECKPOINT_PATH,
                             ) -> list[tuple[str, np.ndarray]]:
    """
    
    """
    
    # ~ loading the model from disk
    model = load_model(model_path)
    processing = load_processing()
    
    # ~ generating the counterfactuals
    # This function from the utils module actually contains the main implementation for the 
    # generation of the counterfactuals and just requires the model instance on which to base 
    # the counterfactuals.
    return generate_counterfactuals_with_model(
        model=model,
        smiles=smiles,
        processing=processing,
        num=num,
    )
    