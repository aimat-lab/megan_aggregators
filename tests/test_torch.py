"""
Unit tests for the "megan_aggregators.torch.py" module.
"""
import os
import pytest

import numpy as np
from graph_attention_student.torch.megan import Megan

from megan_aggregators.utils import load_processing
from megan_aggregators.torch import load_model
from megan_aggregators.torch import predict_aggregator
from megan_aggregators.torch import generate_counterfactuals


def test_load_model_basically_works():
    """
    The "load_model" function is supposed to return the default Megan model that is shipped with the 
    package. This model is pre-trained and immediatly able to make a prediction about the 
    aggregation classification.
    """
    model: Megan = load_model()
    assert isinstance(model, Megan)
    
    # Then we just test the prediction for one example to make sure that it generally works.
    processing = load_processing()
    smiles = 'CCC(CCN)CCC'
    graph = processing.process(smiles)
    
    info = model.forward_graphs([graph])[0]
    assert isinstance(info, dict)
    assert 'graph_output' in info
    assert info['graph_output'].shape == (2, )
    
    
def test_predict_aggregator_basically_works():
    """
    The "predict_aggregator" function is supposed to return the probability of a given molecule being 
    an aggregator. This probability is a float value between 0 and 1.
    """
    smiles = 'CCC(CCN)CCC'
    probab = predict_aggregator(smiles)
    print(probab)
    assert isinstance(probab, float)
    assert probab > 0.0
    
    
def test_generate_counterfactuals_basically_works():
    """
    The "generate_counterfactuals" function is supposed to return a list of counterfactuals in the 
    format of a list of tuples where the first value of the tuple is the counterfactual SMILES string 
    and the second value is the predicted output of the model.
    """
    # We'll simply test the function with a single example to make sure that it generally works.
    smiles = 'CCC(CCN)CCC'
    counterfactuals = generate_counterfactuals(
        smiles=smiles,
        num=10,
    )
    print(counterfactuals)
    assert isinstance(counterfactuals, list)
    assert len(counterfactuals) == 10
    for s, a, d in counterfactuals:
        assert isinstance(s, str)
        assert s != smiles
        
        assert isinstance(d, float)
        assert d > 0.0