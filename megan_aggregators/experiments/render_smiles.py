"""
This string will be saved to the experiment's archive folder as the "experiment description"

CHANGELOG

0.1.0 - initial version
"""
import os
import pathlib

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from pycomex.experiment import Experiment
from pycomex.util import Skippable
from visual_graph_datasets.visualization.base import create_frameless_figure
from visual_graph_datasets.visualization.molecules import mol_from_smiles
from visual_graph_datasets.visualization.molecules import visualize_molecular_graph_from_mol

mpl.use('TkAgg')
PATH = pathlib.Path(__file__).parent.absolute()

SHORT_DESCRIPTION = (
    'This will be the short description when listing the experiments from the command line'
)

# == GENERATION PARAMETERS ==
SMILES_LIST = [
    'C1C=CC=CC=1',
    'CN1CCOCC1',
    'ClC1C=CC=CC=1',
    'CC(N)=O.CC(O)=O',
    'C1C=CC=NC=1',
    'C1C=CC(SC=C2)=C2C=1',
]
IMAGE_WIDTH = 500
IMAGE_HEIGHT = 500

# == EXPERIMENT PARAMETERS ==
BASE_PATH = PATH
NAMESPACE = 'results/render_smiles'
DEBUG = True
with Skippable(), (e := Experiment(BASE_PATH, NAMESPACE, globals())):
    # "e.info" should be used instead of "print". It will use python's "logging" module to not only
    # print the content ot the console but also write it into a log file at the same time.
    e.info('starting experiment...')

    for c, smiles in enumerate(SMILES_LIST):
        fig, ax = create_frameless_figure(width=IMAGE_WIDTH, height=IMAGE_HEIGHT)
        mol = mol_from_smiles(smiles)
        visualize_molecular_graph_from_mol(
            ax=ax,
            mol=mol,
            image_width=IMAGE_WIDTH,
            image_height=IMAGE_HEIGHT,
        )
        fig_path = os.path.join(e.path, f'{c:02d}.png')
        fig.savefig(fig_path)
        plt.close(fig)


# == ANALYSIS ==
with Skippable(), e.analysis:
    e.info('starting analysis...')
