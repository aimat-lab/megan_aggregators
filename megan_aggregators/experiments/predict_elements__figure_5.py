import os

import matplotlib.pyplot as plt
from scipy.special import softmax
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
    {'name': 'O', 'smiles': 'CC1=C(N=CC=C1)O', 'energy': -1.18},
    {'name': 'P', 'smiles': 'CC1=NC(=CC=C1)O', 'energy': -1.2},  
    {'name': 'Q', 'smiles': 'C1=NC(=CC=C1)O', 'energy': -1.17},
    {'name': 'R', 'smiles': 'CC1=CN=C(C=C1)O', 'energy': -1.17},
    {'name': 'X', 'smiles': 'CC1=NC(=CC=C1)N', 'energy': -0.8},
    {'name': 'Y', 'smiles': 'OC1=CC=NC=C1', 'energy': -0.72},
    {'name': 'Z', 'smiles': 'OC1=CN=CC=C1', 'energy': -0.72},
    {'name': 'W', 'smiles': 'CC1=CC=CC=N1', 'energy': -0.38},
    {'name': 'V', 'smiles': 'CC1=CC=CC=C1O', 'energy': -0.5},
    {'name': 'U', 'smiles': 'CC1=C(C=CC=N1)O', 'energy': -0.55},
    {'name': 'T', 'smiles': 'CC1=NC=C(C=C1)O', 'energy': -0.75},
    {'name': 'S', 'smiles': 'CC1=NC=CC(=C1)O', 'energy': -0.8},
]

# == EXPERIMENT PARAMETERS ==

experiment = Experiment.extend(
    'predict_elements.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)

@experiment.hook('evaluate', default=False, replace=True)
def evaluate(e: Experiment,
             elements: list[dict],
             graphs: list[dict],
             infos: list[dict]
             ) -> None:
    
    e.log('plotting binding energy vs. confidence...')
    
    fig, ax = plt.subplots(
        ncols=1, 
        nrows=1,
        figsize=(5, 5),
    )
    ax.set_xlabel('binding energy')
    ax.set_ylabel('aggregator confidence')
    
    for element, graph, info in zip(elements, graphs, infos):
        confidence = softmax(info['graph_output'])[0]
        energy = element['energy']
        
        ax.scatter(
            energy, confidence,
            color='black',
        )
        ax.text(
            energy, confidence,
            element['name'],
            color='black'
        )
        
    fig_path = os.path.join(e.path, 'energy_vs_confidence.pdf')
    fig.savefig(fig_path)


experiment.run_if_main()