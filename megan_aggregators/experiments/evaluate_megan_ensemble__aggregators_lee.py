"""
This experiment extends "evaluate_megan_ensemble" towards the evaluation on a specific, external 
test set that was used in by Lee et al. as well.
"""
import typing as t

from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path


# == DATASET PARAMETERS ==
# These are the parameters related to the model files

# :param VISUAL_GRAPH_DATASET:
#       This string has to define the visual graph dataset on which the ensemble should be evaluated 
#       on. This string can either be a valid absolute path to a vgd folder on the local system or 
#       alternatively a valid string identifier for a vgd on the remote file share location, in which 
#       case the dataset will be downloaded first.
VISUAL_GRAPH_DATASET: str = '/media/ssd/.visual_graph_datasets/datasets/aggregators_lee'
# :param TEST_INDICES_PATH:
#       This is optionally the string path to a JSON file containing a list of indices for the dataset. 
#       if provided, these indices will define on which elements of the dataset, the evaluation will be 
#       performed. If this is None, then the entire given dataset will be used.
TEST_INDICES_PATH: t.Optional[str] = None

# == EVALUATION PARAMETERS ==
# These parameters define the evaluation behavior of the experiment.

# :param NUM_MISTAKES:
#       This defines the integer number of elements that will be picked as the most confident mistakes. 
#       the most confident mistakes are those samples of the dataset where the model makes the largest 
#       mistake w.r.t. to the ground truth targets, yet at the same time the ensemble has the lowest 
#       uncertainty.
NUM_MISTAKES: int = 50

experiment = Experiment.extend(
    'evaluate_megan_ensemble.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)

experiment.run_if_main()