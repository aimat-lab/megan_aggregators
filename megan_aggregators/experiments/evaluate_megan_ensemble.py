"""
This experiment evaluates an ensemble of MEGAN models on a visual graph dataset that 
is specified through the model parameters.

**VARIABLE NAMES**

In this file the following variable names are used:
- B: the number of graphs in the evaluation set
- H: the number of models in the ensemble
"""
import os
import json
import pathlib
import typing as t

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from scipy.special import softmax
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace
from visual_graph_datasets.config import Config
from visual_graph_datasets.web import ensure_dataset
from visual_graph_datasets.data import VisualGraphDatasetReader
from visual_graph_datasets.visualization.base import draw_image
from visual_graph_datasets.processing.base import draw_image
from visual_graph_datasets.visualization.importances import create_importances_pdf
from visual_graph_datasets.visualization.importances import plot_node_importances_background
from visual_graph_datasets.visualization.importances import plot_edge_importances_background
from graph_attention_student.models import Megan2
from graph_attention_student.models import load_model

from megan_aggregators.utils import ASSETS_PATH
from megan_aggregators.models import ModelEnsemble

mpl.use('Agg')


PATH = pathlib.Path(__file__).parent.absolute()

# == DATASET PARAMETERS ==
# These are the parameters related to the model files

# :param VISUAL_GRAPH_DATASET:
#       This string has to define the visual graph dataset on which the ensemble should be evaluated 
#       on. This string can either be a valid absolute path to a vgd folder on the local system or 
#       alternatively a valid string identifier for a vgd on the remote file share location, in which 
#       case the dataset will be downloaded first.
VISUAL_GRAPH_DATASET: str = os.path.join(PATH, 'cache', 'aggregators_binary_protonated', 'test')
# :param TEST_INDICES_PATH:
#       This is optionally the string path to a JSON file containing a list of indices for the dataset. 
#       if provided, these indices will define on which elements of the dataset, the evaluation will be 
#       performed. If this is None, then the entire given dataset will be used.
TEST_INDICES_PATH: t.Optional[str] = None
# :param TARGET_NAMES:
#       This is a dictionary which defines human readable names for the target values that can be 
#       found in the dataset. The keys of this dict structure are integer indices of the targets
#       and the values are the string names.
TARGET_NAMES: t.Dict[int, str] = {
    0: 'non-aggregator',
    1: 'aggregator'
}

# == MODEL PARAMETERS ==
# These parameters define the models that will be used as part of the ensemble

# :param MODEL_PATHS:
#       This is a list of valid absolute string paths that point to existing model folders on the local 
#       system. Each element in this list should represent one model that will ultimately be used as 
#       part of the ensemble.
MODEL_PATHS: t.List[str] = [
    os.path.join(ASSETS_PATH, 'models', 'model_0'),
    os.path.join(ASSETS_PATH, 'models', 'model_1'),
    # os.path.join(ASSETS_PATH, 'models', 'model_2'),
    os.path.join(ASSETS_PATH, 'models', 'model_3'),
    os.path.join(ASSETS_PATH, 'models', 'model_4'),
    # os.path.join(ASSETS_PATH, 'models', 'model_5'),
    os.path.join(ASSETS_PATH, 'models', 'model_6'),
]

# == EVALUATION PARAMETERS ==
# These parameters define the evaluation behavior of the experiment.

# :param NUM_MISTAKES:
#       This defines the integer number of elements that will be picked as the most confident mistakes. 
#       the most confident mistakes are those samples of the dataset where the model makes the largest 
#       mistake w.r.t. to the ground truth targets, yet at the same time the ensemble has the lowest 
#       uncertainty.
NUM_MISTAKES: int = 50
# :param NUM_EXAMPLES:
#       This is the number of elements that are used as examples to visualize the ensemble explanations 
#       that are created by the ensemble.
NUM_EXAMPLES: int = 100

__DEBUG__ = True

@Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)
def experiment(e: Experiment):
    e.log('starting experiment...')
    
    # ~ loading dataset
    # At first we need to load the dataset from the disk. This dataset is expected to be a 
    # "visual graph dataset". In this format, the dataset is represented as a folder.

    config = Config()
    config.load()

    if os.path.exists(e.VISUAL_GRAPH_DATASET):
        dataset_path = e.VISUAL_GRAPH_DATASET
    else: 
        dataset_path = ensure_dataset(
            dataset_name=e.VISUAL_GRAPH_DATASET,
            config=config,
            logger=e.logger,
        )
           
    e.log('loading the dataset...')
    reader = VisualGraphDatasetReader(
        path=dataset_path,
        logger=e.logger,
    )
    index_data_map = reader.read()
    num_elements = len(index_data_map)
    e.log(f'loaded dataset with {num_elements} elements')
    
    # ~ the test split
    # Now that we have loaded the dataset into the memory we know take the test elements 
    # from that dataset.
    # This test split is optionally defined by the TEST_INDICES_PATH. This can be an absolute 
    # path pointing to a JSON file containing the list of test indices. This value can also be 
    # None, in which case we simply use the whole dataset for the evaluation.
    
    if e.TEST_INDICES_PATH is not None:
        e.log(f'loading test indices from {e.TEST_INDICES_PATH}')
        with open(e.TEST_INDICES_PATH, mode='r') as file:
            content = file.read()
            indices = json.loads(content)
            
    else:
        indices = list(index_data_map.keys())
        e.log(f'no test indices specified, using the entire dataset')
    
    graphs = [index_data_map[index]['metadata']['graph'] for index in indices]
    e.log(f'using {len(graphs)} elements for evaluation')
    
    # ~ loading the ensemble
    # Before we can do any predictions / evaluations we first have to load the model ensemble. For 
    # that we load all the individual models from their given paths on the disk and then construct 
    # the ensemble with the list of all the model objects.
    
    e.log('loading the models ans constructing the ensemble...')
    models = []
    for path in e.MODEL_PATHS:
        model = load_model(path)
        models.append(model)
    
    num_models = len(models)
    e.log(f'loaded {num_models} models')
    
    ensemble = ModelEnsemble(models)
    e.log(f'constructed the ensemble')

    # Now that we have the model ensemble we can use it to make the predictions for all the 
    # graphs in the evaluation dataset.
    # The "predict_graph" method returns a list of numpy arrays where each array represents the 
    # output predictions for one of the graphs for each of the models from the ensemble (-1 dimension)
    # We know aggregate this by doing a consensus decision and we can also calculate the model 
    # uncertainty.
    e.log('aggregating the individual predictions...')
    predictions: t.List[np.ndarray] = ensemble.predict_graphs_all(graphs)
    
    consensuses: t.List[int] = []
    uncertainties: t.List[int] = []
    for index, graph, pred in zip(indices, graphs, predictions):
        # pred: (B, H)
        pred = softmax(pred, axis=0)
        consensus = np.mean(pred, axis=-1)
        consensuses.append(consensus)
        graph['graph_labels_consensus'] = consensus
        
        uncertainty = np.std(pred, axis=-1)
        # q_top = np.quantile(pred, 0.75, axis=-1)
        # q_bot = np.quantile(pred, 0.25, axis=-1)
        # uncertainty = q_top - q_bot 
        uncertainties.append(uncertainty)
        graph['graph_labels_uncertainty'] = uncertainty
        
        
    values_pred = [arr[1] for arr in consensuses]
    values_true = [graph['graph_labels'][1] for graph in graphs]
    
    labels_pred = [np.argmax(arr) for arr in consensuses]
    labels_true = [np.argmax(graph['graph_labels']) for graph in graphs]
    
    # ~ calculating evaluation metrics
    e.log('evaluating results...')
    
    acc_value = accuracy_score(labels_true, labels_pred)
    e.log(f'evaluation results:')
    e.log(f' * acc: {acc_value:.3f}')
            
    f1_value = f1_score(labels_true, labels_pred)
    e.log(f' * f1: {f1_value:.3f}')
            
    auc_value = roc_auc_score(values_true, values_pred)
    e.log(f' * auc: {auc_value:.3f}')
    
    # ~ plotting the ROC AUC curve
    fpr, tpr, _ = roc_curve(values_true, values_pred)
    fig, ax = plt.subplots(
        ncols=1,
        nrows=1,
        figsize=(10, 10)
    )
    ax.plot(
        fpr, tpr,
        color='darkorange',
        lw=2,
        label=f'AUC: {auc_value:.3f}'
    )
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', label='random')
    ax.set_title(f'Receiver Operating Characteristic\n'
                 f'Accuracy: {acc_value:.3f} - F1: {f1_value:.3f}')
    ax.legend()
    
    fig_path = os.path.join(e.path, 'roc.pdf')
    fig.savefig(fig_path)
    plt.close(fig)
    
    # ~ plotting uncertainty vs error.
    # Another plot that will be very interesting here is to plot the model's error against the uncertainty. 
    # Because we are using an ensemble here, there is a relatively easy way to get a measure of the model's classification 
    # uncertainty by simply using the standard deviation over the individual predictions. 
    
    error_values: t.List[float] = [abs(value_true - value_pred) for value_true, value_pred in zip(values_true, values_pred)]
    uncertainty_values: t.List[float] = [arr[1] for arr in uncertainties]
    fig_uncertainty, ax_uncertainty = plt.subplots(
        ncols=1,
        nrows=1,
        figsize=(10, 10),
    )
    ax_uncertainty.scatter(
        error_values, 
        uncertainty_values,
        color='gray',
        label='test set',
    )
    ax_uncertainty.set_title('Uncertainty vs Error')
    ax_uncertainty.set_xlabel('Prediction Error')
    ax_uncertainty.set_ylabel('Ensemble Uncertainty')
    
    fig_path = os.path.join(e.path, 'uncertainty.pdf')
    fig_uncertainty.savefig(fig_path)
    
    # ~ extracting the confident mistakes
    # When looking at the uncertainty vs. error plot there is an interesting pattern there where it almost forms 
    # a half-dome: For low error predictions there is low uncertainty and then at first as the error increases so 
    # does the uncertainty. But then the uncertainty decreases again for very high errors. That basically means 
    # there are samples for which ALL of the models are very confidently wrong...
    # It makes sense to extract these samples and look at them to see if there is some kind of pattern there.
    e.log('extracting high confidence mistakes...')
    index_error_tuples = [(index, error, uncertainty) 
                          for index, error, uncertainty in zip(indices, error_values, uncertainty_values)]
    index_error_tuples.sort(key=lambda tpl: tpl[1] - 2 * tpl[2], reverse=True)
    index_error_tuples = index_error_tuples[:e.NUM_MISTAKES]
    mistake_indices, mistake_errors, mistake_uncertainties = zip(*index_error_tuples)
    e.log(f'identified {len(mistake_indices)} high confidence mistakes')
    
    pdf_path = os.path.join(e.path, 'confident_mistakes.pdf')
    with PdfPages(pdf_path) as pdf:
        for index in mistake_indices:
            data = index_data_map[index]
            graph = data['metadata']['graph']
            value_true = graph['graph_labels'][1]
            value_pred = graph['graph_labels_consensus'][1]
            
            label_true = round(value_true)
            label_pred = round(value_pred)
            
            fig, ax = plt.subplots(
                ncols=1,
                nrows=1,
                figsize=(10, 10)
            )
            draw_image(ax, data['image_path'])
            ax.set_title(f'index: {index}\n'
                         f'{data["metadata"]["smiles"]}\n'
                         f'true: {e.TARGET_NAMES[label_true]} ({value_true}) - ' 
                         f'pred: {e.TARGET_NAMES[label_pred]} ({value_pred:.3f})')
            
            pdf.savefig(fig)
            plt.close(fig)
            
    
    ax_uncertainty.scatter(
        mistake_errors, mistake_uncertainties,
        color='red',
        label='confident mistakes'
    )
    ax_uncertainty.legend()
    
    fig_path = os.path.join(e.path, 'uncertainty2.pdf')
    fig_uncertainty.savefig(fig_path)
    
    # ~ model similarity
    # In this section we want to see how similar the predictions of the various models are to each other 
    # for that purpose we will create a matrix of linear correlation matrices for the individual models.
    
    e.log('plotting model pairwise correlation matrix...')
    # model_values: (B, H)
    model_values = np.array([np.argmax(arr, axis=0) for arr in predictions])
    corr = np.corrcoef(model_values.T)
    
    fig, ax = plt.subplots(
        ncols=1,
        nrows=1,
        figsize=(10, 10),
    )
    cax = ax.matshow(corr, cmap='coolwarm', vmin=0, vmax=1)
    for (i, j), value in np.ndenumerate(corr):
        ax.text(
            j, i,
            f'{value:.2f}',
            ha='center',
            va='center',
            color='black'
        )
        
    fig.colorbar(cax)
    
    ax.set_title('Model Correlation Matrix')
    
    fig_path = os.path.join(e.path, 'model_correlation.pdf')
    fig.savefig(fig_path)
    
    # ~ ensemble explanations
    # It is also possible to create explanations from the ensemble method.
    
    e.log('creating ensemble explanations...')
    explanations: t.List[tuple] = ensemble.explain_graphs_all(graphs)
    # all_node_importances: (V, K, H)
    # all_edge_importances: (E, K, H)
    all_node_importances, all_edge_importances = list(zip(*explanations))
    node_importances = [np.median(ni, axis=-1) for ni in all_node_importances]
    edge_importances = [np.median(ei, axis=-1) for ei in all_edge_importances]
    
    e.log(f'plotting {e.NUM_EXAMPLES} example explanations...')
    examples_path = os.path.join(e.path, 'example explananations.pdf')
    create_importances_pdf(
        graph_list=graphs[:e.NUM_EXAMPLES],
        image_path_list=[index_data_map[index]['image_path'] for index in indices[:e.NUM_EXAMPLES]],
        node_positions_list=[graph['node_positions'] for graph in graphs[:e.NUM_EXAMPLES]],
        importances_map={
            'consensus': (
                node_importances[:e.NUM_EXAMPLES], 
                edge_importances[:e.NUM_EXAMPLES],
            ),
        },
        plot_node_importances_cb=plot_node_importances_background,
        plot_edge_importances_cb=plot_edge_importances_background,
        output_path=examples_path,
        logger=e.logger,
        log_step=10,
    )
        
    
            
experiment.run_if_main()