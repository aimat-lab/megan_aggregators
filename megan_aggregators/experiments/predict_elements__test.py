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
    {'name': 'COc4cccc(C3CC(c1ccc(F)cc1)Nc2nc(N)nn23)c4', 'smiles': 'COc4cccc(C3CC(c1ccc(F)cc1)Nc2nc(N)nn23)c4'},
    {'name': 'Cc4cccc(C3CC(c1ccccc1)Nc2ncnn23)c4', 'smiles': 'Cc4cccc(C3CC(c1ccccc1)Nc2ncnn23)c4'},
    {'name': 'COc4ccc(C3CC(c1cccc(C)c1)n2ncnc2N3)cc4', 'smiles': 'COc4ccc(C3CC(c1cccc(C)c1)n2ncnc2N3)cc4'},
    {'name': 'Cc4cccc(C3CC(c1ccc(F)cc1)Nc2nc(N)nn23)c4', 'smiles': 'Cc4cccc(C3CC(c1ccc(F)cc1)Nc2nc(N)nn23)c4'},
    {'name': 'COc4cccc(C3CC(c1cccc(OC)c1)n2nc(N)nc2N3)c4', 'smiles': 'COc4cccc(C3CC(c1cccc(OC)c1)n2nc(N)nc2N3)c4'},
    {'name': 'COc4cccc(C3CC(c1ccc(C)cc1)Nc2nc(N)nn23)c4', 'smiles': 'COc4cccc(C3CC(c1ccc(C)cc1)Nc2nc(N)nn23)c4'},
    {'name': 'Cc1ccccc1C4CC(c2ccccc2)Nc3ncnn34', 'smiles': 'Cc1ccccc1C4CC(c2ccccc2)Nc3ncnn34'},
    {'name': 'COc4cccc(C3CC(c1ccc(C)cc1)n2nc(N)nc2N3)c4', 'smiles': 'COc4cccc(C3CC(c1ccc(C)cc1)n2nc(N)nc2N3)c4'},
    {'name': 'COc4cccc(C3CC(c1ccccc1OC)n2ncnc2N3)c4', 'smiles': 'COc4cccc(C3CC(c1ccccc1OC)n2ncnc2N3)c4'},
    {'name': 'CC(=O)OC3CCC4C2CCc1cc(O)ccc1C2CCC34C', 'smiles': 'CC(=O)OC3CCC4C2CCc1cc(O)ccc1C2CCC34C'},
    {'name': 'Oc1cc(O)nc(SCc2ccccc2Cl)n1', 'smiles': 'Oc1cc(O)nc(SCc2ccccc2Cl)n1'},
]

# == EXPERIMENT PARAMETERS ==

experiment = Experiment.extend(
    'predict_elements.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)

experiment.run_if_main()