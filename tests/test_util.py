import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow.keras as ks

from megan_aggregators.util import get_version
from megan_aggregators.util import render_latex
from megan_aggregators.util import plot_roc_curve
from megan_aggregators.util import load_model

from .util import ASSETS_PATH, ARTIFACTS_PATH

mpl.use('TkAgg')


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
    The load_model functions should return a valid keras model, more specifically the currently best trained
    version of the megan model for this task.
    """
    model = load_model()
    assert isinstance(model, ks.models.Model)
