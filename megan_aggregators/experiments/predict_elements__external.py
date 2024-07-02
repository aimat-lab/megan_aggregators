import os
import csv
import pathlib

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import softmax
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

PATH = pathlib.Path(__file__).parent.absolute()

# == INPUT PARAMETERS ==
# These parameters define the input of the experiment. This mainly includes the elements for which the 
# actual predictions should be made.

# :param CSV_PATH:
#       This is the absolute path to the CSV file from which the elements are to be loaded. This CSV file
#       should contain the information about the elements.
CSV_PATH = os.path.join(PATH, 'assets', 'external.csv')
# :param PROTONATE_ELEMENTS:
#       This parameter defines whether the elements should be protonated before they are processed. The protonation
#       is done with the DimorphiteDL tool and is calculated based on the SMILES representation of the input 
#       elements.
PROTONATE_ELEMENTS: bool = False

# == EXPERIMENT PARAMETERS ==

experiment = Experiment.extend(
    'predict_elements.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)


@experiment.hook('load_elements', default=False, replace=True)
def load_elements(e: Experiment) -> list[dict]:
    e.log('loading the elements from external CSV file...')
    
    elements = []
    with open(e.CSV_PATH) as file:
        dict_reader = csv.DictReader(file)
        for c, row in enumerate(dict_reader):
            row['smiles'] = max(row['smiles'].split('.'), key=len)
            elements.append(row)
            
    return elements


@experiment.hook('process_element', default=False, replace=True)
def process_element(e: Experiment, 
                    element: dict,
                    graph: dict,
                    info: dict,
                    ) -> dict:
    
    out = softmax(info['graph_output'])
    label = np.argmax(out)
    
    e.log(f' * {element["smiles"]} - label: {label}')
    data = {
        'name': element['name'] if 'name' in element else '',
        'smiles': element['smiles'],
        'aggregator': round(out[0], 5),
        'non_aggregator': round(out[1], 5),
        'pred': 'aggregator' if label == 0 else 'non-aggregator',
        'true': 'aggregator' if element['agg'] == '1' else 'non-aggregator',
    }
    
    return data


@experiment.hook('evaluate', default=False, replace=True)
def evaluate(e: Experiment,
             elements: list[dict],
             graphs: list[dict],
             infos: list[dict]
             ) -> None:
    
    e.log('evaluating the prediction accuracy...')
    labels_true = [1 - int(element['agg']) for element in elements]
    labels_pred = [np.argmax(info['graph_output']) for info in infos]
    
    acc_value = accuracy_score(labels_true, labels_pred)
    e.log(f' * accuracy: {acc_value:.3f}')
    
    # ~ confusion matrix
    e.log('plotting confusion matrix...')
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))

    # Generate the confusion matrix
    conf_mat = confusion_matrix(labels_true, labels_pred)
    sns.heatmap(
        conf_mat, 
        ax=ax,
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['Aggregator', 'Non-Aggregator'],
        yticklabels=['Aggregator', 'Non-Aggregator'],
    )
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title(f'Confusion Matrix\n'
                 f'Accuracy: {acc_value*100:.3f}%')
    fig_path = os.path.join(e.path, 'confusion_matrix.pdf')
    fig.savefig(fig_path)

experiment.run_if_main()