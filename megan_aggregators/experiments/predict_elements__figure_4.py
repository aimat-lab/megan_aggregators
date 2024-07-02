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

ELEMENTS: list[dict] = [
    {'name': 'I', 'smiles': 'CCCc1cccc(O)n1'},
    {'name': 'J', 'smiles': 'CCCc1cc(O)ccn1'},
    {'name': 'K', 'smiles': 'CCC=C1CC=CC(O)=N1'},
    {'name': 'L', 'smiles': 'CCCc1ccc(O)cn1'},
    {'name': 'M', 'smiles': 'CCCc1cccc(N)n1'},
    {'name': 'N', 'smiles': 'CC=CC1CC=CC(O)=N1'},
    # Other variants
    {'name': 'alt 1', 'smiles': 'CCCC1=CC=CC=N1'},
    {'name': 'alt 2', 'smiles': 'CCCC1=CN=CC=C1'},
    {'name': 'alt 3', 'smiles': 'CCCC1=CC=NC=C1'},
    {'name': 'alt 4', 'smiles': 'CCCC1=CC=CC(S)N1'},
    {'name': 'alt 5', 'smiles': 'CCCC1=CC=CC(N1)OC'},
    {'name': 'alt 6', 'smiles': 'CCCc1ccc(S)cn1'},
    {'name': 'alt 7', 'smiles': 'CCCC1=CC(O)NC=C1'},
]

# == EXPERIMENT PARAMETERS ==

experiment = Experiment.extend(
    'predict_elements.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)

experiment.run_if_main()