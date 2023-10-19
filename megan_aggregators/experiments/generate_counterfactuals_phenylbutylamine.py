from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

SMILES = 'NCCCCC1=CC=CC=C1'

experiment = Experiment.extend(
    'generate_counterfactuals.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)

experiment.run_if_main()
