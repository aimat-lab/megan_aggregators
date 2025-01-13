import os
import pytest

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted

from visual_graph_datasets.data import load_visual_graph_element
from visual_graph_datasets.processing.molecules import MoleculeProcessing

from megan_aggregators.models import load_model
from megan_aggregators.utils import get_version
from megan_aggregators.utils import render_latex
from megan_aggregators.utils import plot_roc_curve
from megan_aggregators.utils import visualize_explanations
from megan_aggregators.utils import load_processing
from megan_aggregators.utils import create_report
from megan_aggregators.utils import plot_calibration_curve
from megan_aggregators.utils import create_confidence_histograms
from megan_aggregators.utils import IntegerOutputClassifier

from .util import ASSETS_PATH, ARTIFACTS_PATH

mpl.use('Agg')


def test_create_confidence_histograms():
    """
    03.08.24: Given the true labels and the predicted probabilities, the ``create_confidence_histograms``
    function should create several histogram plots that show the confidence distribution of the model
    for each class.
    """
    y_true = np.random.randint(2, size=(100, 2))
    y_pred = np.random.uniform(0, 1, size=(100, 2))

    fig = create_confidence_histograms(
        y_true=y_true,
        y_pred=y_pred,
    )
    fig_path = os.path.join(ARTIFACTS_PATH, 'test_create_confidence_histograms.pdf')
    fig.savefig(fig_path)
    assert os.path.exists(fig_path)
    

def test_plot_calibration_curve():
    """
    03.08.24: Given the true labels and the predicted probabilities, the ``plot_calibration_curve`` function
    should create a calibration curve plot that shows the relationship between the predicted probabilities
    and the true labels.
    """
    y_true = np.random.randint(2, size=(100, ))
    y_pred = np.random.uniform(0, 1, size=(100, ))
    
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.set_title('Calibration Curve')
    plot_calibration_curve(
        ax=ax,
        y_true=y_true,
        y_pred=y_pred,
    )
    
    fig_path = os.path.join(ARTIFACTS_PATH, 'test_plot_calibration_curve.pdf')
    fig.savefig(fig_path)
    assert os.path.exists(fig_path)


def test_get_version():
    version = get_version()
    assert isinstance(version, str)
    assert version != ''


def test_render_latex():
    output_path = os.path.join(ASSETS_PATH, 'out.pdf')
    render_latex({'content': '$\pi = 3.141$'}, output_path)
    assert os.path.exists(output_path)


def test_plot_roc_curve_basically_works():
    # First of all we need to create a test dataset for it
    num_elements = 100
    y_true = np.random.randint(2, size=(num_elements, ))
    y_pred = np.random.uniform(0, 1, size=(num_elements, ))

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.set_title('ROC Curve')
    plot_roc_curve(
        ax=ax,
        y_true=y_true,
        y_pred=y_pred,
        label='agg vs. non-agg'
    )
    ax.legend()

    path = os.path.join(ARTIFACTS_PATH, 'roc_curve.pdf')
    fig.savefig(path)
    
@pytest.mark.parametrize(
    "num_channels",
    [2, 3]
)
def test_visualize_explanations(num_channels):
    
    processing = load_processing()
    
    data = load_visual_graph_element(
        path=ASSETS_PATH,
        name='test'
    )
    smiles = data['metadata']['smiles']
    graph = data['metadata']['graph']
    
    node_importances = np.random.uniform(0, 1, size=(len(graph['node_indices']), num_channels))
    edge_importances = np.random.uniform(0, 1, size=(len(graph['edge_indices']), num_channels))
    
    fig = visualize_explanations(
        smiles,
        processing=processing,
        node_importances=node_importances,
        edge_importances=edge_importances,
    )
    assert isinstance(fig, plt.Figure)

    path = os.path.join(ARTIFACTS_PATH, f'test_visualize_explanations__{num_channels}.pdf')
    fig.savefig(path)
    
    
def test_load_processing_basically_works():
    """
    The "load_processing" module is supposed to return the default MoleculeProcessing instance which 
    can be used to transform an input SMILES representation of a molecule into a graph dict 
    representation which can be used by the default MEGAN model to make it's predictions.
    """
    
    processing = load_processing()
    assert isinstance(processing, MoleculeProcessing)
    
    # Just testing a single example to make sure there are no errors in general.
    smiles = 'CCC(CCN)CCC'
    graph = processing.process(smiles)
    assert isinstance(graph, dict)
    assert len(graph) != 0
    
    
def test_integer_output_classifier_basically_works():
    """
    The IntegerOutputClassifier is a simple wrapper around a sklearn classifier model such that the 
    output array of the predict method is actually a intger typed array.
    """
    model = RandomForestClassifier()
    model_wrapper = IntegerOutputClassifier(model)
    
    assert isinstance(model_wrapper, IntegerOutputClassifier)
    # Additionally the output of this method needs to be hardcoded as true regardless of the 
    # fitted state of the model.
    assert model_wrapper.__sklearn_is_fitted__() == True
    # This should therefore not cause an issue!
    check_is_fitted(model_wrapper)
    
    # Now we can check if the output of the predict method is actually an integer array.
    X = np.random.uniform(0, 1, size=(100, 10))
    y = np.random.randint(2, size=(100, ))
    model.fit(X, y)
    y_pred = model_wrapper.predict(X)
    assert isinstance(y_pred, np.ndarray)
    assert y_pred.dtype == np.int64