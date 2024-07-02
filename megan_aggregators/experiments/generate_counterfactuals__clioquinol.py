"""
This experiment extends the base experiment "generate_counterfactuals" with a specific base molecule
for which the counterfactuals are generated. That molecule being "Clioquinol".
This molecule is a known aggregator which means that the counterfactuals are supposed to flip the 
label to non-aggregator.
"""
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

# == EXPERIMENT PARAMETERS ==
# These parameters define the behavior of the counterfactual generation. This includes for 
# example, most importantly, the SMILES string of the molecule for which the counterfactuals
# are to be generated.

# :param SMILES:
#       The SMILES string of the molecule for which the counterfactuals are to be generated.
SMILES = 'C1=CC2=C(C(=C(C=C2Cl)I)O)N=C1 '
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
# :param CLASS_NAMES:
#       This dictionariy defines the integer indices of the classes as the keys and the corresponding 
#       values are the human readable names of the classes so that they can be properly labeled in the 
#       visualizations.
CLASS_NAMES = {
    0: 'aggregator',
    1: 'non-aggregator',
}

experiment = Experiment.extend(
    'generate_counterfactuals.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)

experiment.run_if_main()