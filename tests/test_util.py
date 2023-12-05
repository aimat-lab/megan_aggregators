import os
import pytest

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow.keras as ks

from visual_graph_datasets.data import load_visual_graph_element

from megan_aggregators.models import load_model
from megan_aggregators.utils import get_version
from megan_aggregators.utils import render_latex
from megan_aggregators.utils import plot_roc_curve
from megan_aggregators.utils import visualize_explanations
from megan_aggregators.utils import load_processing

from .util import ASSETS_PATH, ARTIFACTS_PATH

mpl.use('Agg')


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


def test_load_model():
    """
    The load_model function should return a valid keras model, more specifically the currently best trained
    version of the megan model for this task.
    """
    model = load_model()
    assert isinstance(model, ks.models.Model)
    
    
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