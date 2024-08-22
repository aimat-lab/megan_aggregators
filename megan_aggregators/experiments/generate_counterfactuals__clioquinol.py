"""
This experiment extends the base experiment "generate_counterfactuals" with a specific base molecule
for which the counterfactuals are generated. That molecule being "Clioquinol".
This molecule is a known aggregator which means that the counterfactuals are supposed to flip the 
label to non-aggregator.
"""
import os

from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from megan_aggregators.utils import EXPERIMENTS_PATH

# == EXPERIMENT PARAMETERS ==
# These parameters define the behavior of the counterfactual generation. This includes for 
# example, most importantly, the SMILES string of the molecule for which the counterfactuals
# are to be generated.

# :param SMILES:
#       The SMILES string of the molecule for which the counterfactuals are to be generated.
SMILES = 'C1=CC2=C(C(=C(C=C2Cl)I)O)N=C1'
# :param NUM_COUNTERFACTUALS:
#       The number of counterfactuals to be generated for the given molecule.
NUM_COUNTERFACTUALS: int = 10
# :param NEIGHBORHOOD_RANGE:
#       The range of the neighborhood to be used for the counterfactual generation. This parameter
#       determines the maximum number of graph edits to be considered for the counterfactual search.
#       If this parameter is 1, for example, all the graphs which are one graph edit away from the 
#       original are considered during the counterfactual search. For values higher than 1, all of 
#       the neighbors are iteratively expanded by single graph edits as well.
NEIGHBORHOOD_RANGE: int = 2
# :param MODEL_PATH:
#       The path to the model that is to be used for the counterfactual generation. This has to be an
#       absolute string path to an existing checkpoint file that represents a stored model.
#MODEL_PATH: str = os.path.join(EXPERIMENTS_PATH, 'results', 'vgd_torch_chunked_megan__aggregators_binary', '20_08_2024__16_03__i9qk', 'model.ckpt')
MODEL_PATH: str = os.path.join(EXPERIMENTS_PATH, 'results', 'vgd_torch_chunked_megan__aggregators_binary', 'debug', 'model_best.ckpt')

# == VISUALIZATION PARAMETERS ==
# :param IMAGE_WIDTH:
#       The width of the images to be generated for the visualizations of the counterfactuals.
IMAGE_WIDTH = 1000
# :param IMAGE_HEIGHT:
#       The height of the images to be generated for the visualizations of the counterfactuals.
IMAGE_HEIGHT = 1000
# :param FIG_SIZE:
#       The size of the figures to be generated for the visualizations of the counterfactuals.
FIG_SIZE = 6

experiment = Experiment.extend(
    'generate_counterfactuals.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)

experiment.run_if_main()