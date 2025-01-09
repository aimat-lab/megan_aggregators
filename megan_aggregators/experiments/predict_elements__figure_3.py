from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

# == INPUT PARAMETERS ==
# These parameters define the input of the experiment. This mainly includes the elements for which the 
# actual predictions should be made.

# :param ELEMENTS:
#       This list defines the elements for which the model predictions should be generated. THis is a list of 
#       dictionaries where each dictionary contains the information about one element. This dictionary may contain 
#       any additional information about the element but should at least contain a "smiles" property which defines 
#       the smile string representation of the corresponding molecule from which the graph representation can be 
#       constructed.
ELEMENTS: list[str] = [
    {'name': 'A', 'smiles': 'Cc2nccc3c1ccccc1[nH]c23', 'reference': 'c2nccc3c1ccccc1[nH]c23'},
    {'name': 'B', 'smiles': 'Oc2nccc3c1ccccc1[nH]c23', 'reference': 'c2nccc3c1ccccc1[nH]c23'},
    {'name': 'C', 'smiles': 'Oc1cc3c(cn1)[nH]c2ccccc23', 'reference': 'c2nccc3c1ccccc1[nH]c23'},
    {'name': 'D', 'smiles': 'Sc1cc3c(cn1)[nH]c2ccccc23', 'reference': 'c2nccc3c1ccccc1[nH]c23'},
    {'name': 'E', 'smiles': 'c1ccc3c(c1)[nH]c2cnccc23', 'reference': 'c2nccc3c1ccccc1[nH]c23'},
    {'name': 'F', 'smiles': 'Cc2cccc3c1ccncc1[nH]c23', 'reference': 'c2nccc3c1ccccc1[nH]c23'},
    {'name': 'G', 'smiles': 'Cc1cc3c(cn1)[nH]c2ccccc23', 'reference': 'c2nccc3c1ccccc1[nH]c23'},
    {'name': 'H', 'smiles': 'Sc2cccc3c1ccncc1[nH]c23', 'reference': 'c2nccc3c1ccccc1[nH]c23'},
]

# == EXPERIMENT PARAMETERS ==

experiment = Experiment.extend(
    'predict_elements.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)

experiment.run_if_main()