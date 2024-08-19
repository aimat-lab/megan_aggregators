import os
import pytest

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

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


def test_create_report():
    """
    The create_report function should create a PDF file with multiple pages where it is possible to add 
    multiple image and text elements to each of the pages. 
    """
    images = [
        os.path.join(ASSETS_PATH, 'image_0.png'),
        os.path.join(ASSETS_PATH, 'image_1.png'),
    ]
    
    pages: list[dict] = []
    for page_index in range(3):
        
        page = {
            'title': f'Page {page_index}',
            'images': images,
            'texts': [
                'Some text',
                '<strong>Title.</strong> Some other text',
            ]
        }
        pages.append(page)
    
    pdf_path = os.path.join(ARTIFACTS_PATH, 'test_create_report.pdf')
    create_report(
        pages=pages,
        path=pdf_path,
    )


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