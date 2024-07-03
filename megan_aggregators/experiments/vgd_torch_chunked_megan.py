import os
import random
import typing as t

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from imageio.v2 import imread
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import average_precision_score
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from matplotlib.backends.backend_pdf import PdfPages
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path
from visual_graph_datasets.data import VisualGraphDatasetReader
from visual_graph_datasets.visualization.base import draw_image
from visual_graph_datasets.visualization.importances import create_importances_pdf
from visual_graph_datasets.visualization.importances import plot_node_importances_background
from visual_graph_datasets.visualization.importances import plot_edge_importances_background
from graph_attention_student.visualization import plot_leave_one_out_analysis
from graph_attention_student.visualization import plot_regression_fit
from graph_attention_student.torch.megan import Megan
from graph_attention_student.torch.model import AbstractGraphModel
from graph_attention_student.torch.data import data_list_from_graphs
from graph_attention_student.utils import array_normalize

from megan_aggregators.utils import ChunkedDataset, MultiChunkedDataset, GraphDataset
from megan_aggregators.utils import EXPERIMENTS_PATH
from megan_aggregators.utils import plot_roc_curve
from megan_aggregators.utils import plot_confusion_matrix


# == DATASET PARAMETERS ==
# These parameters determine where to load the dataset for the training.

# :param CHUNKED_DATASET_PATH:
#       This parameter is supposed to be a string path to the folder that contains the chunked version of the dataset.
#       This folder should contain two subfolders "train" and "test" which in turn contain the chunked version of the
#       training and test set in visual graph dataset format respectively.
CHUNKED_DATASET_PATH: str = os.path.join(EXPERIMENTS_PATH, 'assets', 'cache', 'rb_dual_motifs')
# :param DATASET_TYPE:
#       This is the parameter that determines the type of the dataset. It may be either "classification" or "regression".
DATASET_TYPE: str = 'regression'
# :param TARGET_NAMES:
#       This dictionary is supposed to contain the human readable names of the target values of the dataset. The keys
#       are the integer indices of the target values and the values are the human readable names of the target values.
TARGET_NAMES: dict = {
    0: 'logS',
}

# == MODEL PARAMETERS ==
# These parameters determine the model that is to be trained.

# :param NUM_CHANNELS:
#       The number of explanation channels for the model.
NUM_CHANNELS: int = 2
# :param CHANNEL_INFOS:
#       This dictionary can be used to add additional information about the explanation channels that 
#       are used in this experiment. The integer keys of the dict are the indices of the channels
#       and the values are dictionaries that contain the information about the channel with that index.
#       This dict has to have as many entries as there are explanation channels defined for the 
#       model. The info dict for each channel may contain a "name" string entry for a human readable name 
#       asssociated with that channel and a "color" entry to define a color of that channel in the 
#       visualizations.
CHANNEL_INFOS: dict = {
    0: {
        'name': 'negative',
        'color': 'skyblue',
    },
    1: {
        'name': 'positive',
        'color': 'coral',
    }
}
# :param UNITS:
#       This list determines the layer structure of the model's graph encoder part. Each element in 
#       this list represents one layer, where the integer value determines the number of hidden units 
#       in that layer of the encoder network.
UNITS: t.List[int] = [32, 32, 32]
# :param HIDDEN_UNITS:
#       The number of hidden units in the MLPs within the message passing layers.
HIDDEN_UNITS: int = 256
# :param IMPORTANCE_UNITS:
#       This list determines the layer structure of the importance MLP which determines the node importance 
#       weights from the node embeddings of the graph. 
#       Each element in this list represents one layer where the integer value determines the number of hidden 
#       units in that layer.
IMPORTANCE_UNITS: t.List[int] = [32, ]
# :param PROJECTION_LAYERS:
#       This list determines the layer structure of the MLP's that act as the channel-specific projections.
#       Each element in this list represents one layer where the integer value determines the number of hidden
#       units in that layer.
PROJECTION_UNITS: t.List[int] = []
# :param FINAL_UNITS:
#       This list determines the layer structure of the model's final prediction MLP. Each element in 
#       this list represents one layer, where the integer value determines the number of hidden units 
#       in that layer of the prediction network.
#       Note that the last value of this list determines the output shape of the entire network and 
#       therefore has to match the number of target values given in the dataset.
FINAL_UNITS: t.List[int] = [1]
# :param IMPORTANCE_FACTOR:
#       This is the coefficient that is used to scale the explanation co-training loss during training.
#       Roughly, the higher this value, the more the model will prioritize the explanations during training.
IMPORTANCE_FACTOR: float = 1.0
# :param IMPORTANCE_OFFSET:
#       This parameter more or less controls how expansive the explanations are - how much of the graph they
#       tend to cover. Higher values tend to lead to more expansive explanations while lower values tend to 
#       lead to sparser explanations. Typical value range 0.5 - 1.5
IMPORTANCE_OFFSET: float = 0.8
# :param SPARSITY_FACTOR:
#       This is the coefficient that is used to scale the explanation sparsity loss during training.
#       The higher this value the more explanation sparsity (less and more discrete explanation masks)
#       is promoted.
SPARSITY_FACTOR: float = 1.0
# :param FIDELITY_FACTOR:
#       This parameter controls the coefficient of the explanation fidelity loss during training. The higher
#       this value, the more the model will be trained to create explanations that actually influence the
#       model's behavior with a positive fidelity (according to their pre-defined interpretation).
#       If this value is set to 0.0, the explanation fidelity loss is completely disabled (==higher computational
#       efficiency).
FIDELITY_FACTOR: float = 0.0
# :param REGRESSION_REFERENCE:
#       When dealing with regression tasks, an important hyperparameter to set is this reference value in the 
#       range of possible target values, which will determine what part of the dataset is to be considered as 
#       negative / positive in regard to the negative and the positive explanation channel. A good first choice 
#       for this parameter is the average target value of the training dataset. Depending on the results for 
#       that choice it is possible to adjust the value to adjust the explanations.
REGRESSION_REFERENCE: t.Optional[float] = 0.0
# :param REGRESSION_MARGIN:
#       When converting the regression problem into the negative/positive classification problem for the 
#       explanation co-training, this determines the margin for the thresholding. Instead of using the regression
#       reference as a hard threshold, values have to be at least this margin value lower/higher than the 
#       regression reference to be considered a class sample.
REGRESSION_MARGIN: t.Optional[float] = 0.0
# :param NORMALIZE_EMBEDDING:
#       This boolean value determines whether the graph embeddings are normalized to a unit length or not.
#       If this is true, the embedding of each individual explanation channel will be L2 normalized such that 
#       it is projected onto the unit sphere.
NORMALIZE_EMBEDDING: bool = True
# :param ATTENTION_AGGREGATION:
#       This string literal determines the strategy which is used to aggregate the edge attention logits over 
#       the various message passing layers in the graph encoder part of the network. This may be one of the 
#       following values: 'sum', 'max', 'min'.
ATTENTION_AGGREGATION: str = 'min'
# :param CONTRASTIVE_FACTOR:
#       This is the factor of the contrastive representation learning loss of the network. If this value is 0 
#       the contrastive repr. learning is completely disabled (increases computational efficiency). The higher 
#       this value the more the contrastive learning will influence the network during training.
CONTRASTIVE_FACTOR: float = 1.0
# :param CONTRASTIVE_NOISE:
#       This float value determines the noise level that is applied when generating the positive augmentations 
#       during the contrastive learning process.
CONTRASTIVE_NOISE: float = 0.0
# :param CONTRASTIVE_TEMP:
#       This float value is a hyperparameter that controls the "temperature" of the contrastive learning loss.
#       The higher this value, the more the contrastive learning will be smoothed out. The lower this value,
#       the more the contrastive learning will be focused on the most similar pairs of embeddings.
CONTRASTIVE_TEMP: float = 1.0
# :param CONTRASTIVE_BETA:
#       This is the float value from the paper about the hard negative mining called the concentration 
#       parameter. It determines how much the contrastive loss is focused on the hardest negative samples.
CONTRASTIVE_BETA: float = 0.1
# :param CONTRASTIVE_TAU:
#       This float value is a hyperparameters of the de-biasing improvement of the contrastive learning loss. 
#       This value should be chosen as roughly the inverse of the number of expected concepts. So as an example 
#       if it is expected that each explanation consists of roughly 10 distinct concepts, this should be chosen 
#       as 1/10 = 0.1
CONTRASTIVE_TAU: float = 0.1
# :param PREDICTION_FACTOR:
#       This is a float value that determines the factor by which the main prediction loss is being scaled 
#       durign the model training. Changing this from 1.0 should usually not be necessary except for regression
#       tasks with a vastly different target value scale.
PREDICTION_FACTOR: float = 1.0
# :param LABEL_SMOOTHING:
#       This is a float value that determines the amount of label smoothing to be applied on the classification 
#       target values. This regularizes the model to not be too confident about the target values and can help
#       to prevent overfitting.
LABEL_SMOOTHING: float = 0.1
# :param CLASS_WEIGHTS:
#       This is a list that determines the class weights that are applied during the training of the model. 
#       This list should have as many values as there are target classes in the given classification task.
#       Each value in this list corresponds to the same target in the models output vector. The value determines 
#       the weight with which the gradients related to that class are scaled during the training process. 
#       choosing one weight higher than the other will make the model focus more on the class with the higher
#       weight. This can be used as one method do deal with an unbalanced dataset.
CLASS_WEIGTHS: list[float] = [1.0, 1.0]
# :param ENCODER_DROPOUT_RATE:
#       This float value determines the dropout rate that is being applied to the node embedding vector after 
#       each layer of the message passing part of the network. This can be used to regularize the model and
#       prevent overfitting.
ENCODER_DROPOUT_RATE: float = 0.0
# :param FINAL_DROPOUT_RATE:
#       This float value determines the dropout rate that is being applied to the final prediction vector of the
#       model. This can be used to regularize the model and prevent overfitting.
FINAL_DROPOUT_RATE: float = 0.0
# :param OUTPUT_NORM:
#       This float value determines the normalization factor that is applied to the output of the model. This
#       can be used to scale the output of the model to a specific range. This is used to tackle the classification 
#       overconfidence problem where the model is too confident about its predictions. By setting a normalization 
#       factor the model can be forced to be less confident about its predictions.
OUTPUT_NORM: t.Optional[float] = 3.0

# == TRAINING PARAMETERS ==
# These parameters determine the training process itself.

# :param BATCH_SIZE:
#       The number of elements to be processed in one batch during the training process.
BATCH_SIZE: int = 200
# :param EPOCHS:
#       The number of epochs to train the model for.
EPOCHS: int = 200
# :param LEARNING_RATE:
#       The learning rate for the model training process.
LEARNING_RATE: float = 1e-3


__DEBUG__ = True
__TESTING__ = True

experiment = Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)

@experiment.hook('evaluate_model', default=False, replace=False)
def evaluate_model(e: Experiment,
                    model: AbstractGraphModel,
                    trainer: pl.Trainer,
                    index_data_map: dict,
                    ) -> None:
    """
    This hook is called during the main training loop AFTER the model has been fully trained to evaluate 
    the performance of the trained model on the test set. The hook receives the trained model as a parameter,
    as well as the repetition index, the full index_data_map dataset and the train and test indices.
    
    This default implementation only implements an evaluation of the model w.r.t. to the main property 
    prediction task. For a regression tasks it calculates R2, MAE metric and visualizes the regression plot.
    For classification tasks, it calculates Accuracy AUC and creates the confusion matrix.
    
    Additionally, this function will create a plot that visualizes the various training plots over the training 
    epochs if a pytorch lightning log can be found in the archive folder.

    It also visualizes the example graphs (with the chosen NUM_EXAMPLES indices from the test set) together 
    with some information about the node embeddings of those graphs into PDF file.
    """
    e.log('evaluating model prediction performance...')
    model.eval()

    test_indices = list(index_data_map.keys())
    graphs_test = [index_data_map[i]['metadata']['graph'] for i in test_indices]
    
    num_test = min(len(test_indices), 100)
    example_indices = random.sample(test_indices, k=num_test)
    metadatas_example = [index_data_map[i]['metadata'] for i in example_indices]
    graphs_example = [metadata['graph'] for metadata in metadatas_example]
    
    # out_true np.ndarray: (B, O)
    out_true = np.array([graph['graph_labels'] for graph in graphs_test])
    # out_pred np.ndarray: (B, O)
    out_pred = model.predict_graphs(graphs_test)
    
    # ~ exporting the test set
    # Here we want to export the test set predictions into an independent CSV file. This is not strictly necessary 
    # as all the predictions will be saved in the experiment storage anyways, but having it directly in a CSV file 
    # makes it easier to communicate / share the results.
    e.log('exporting test set predictions as CSV...')
    metadatas = []
    for index, graph, out in zip(test_indices, graphs_test, out_pred):
        metadata = index_data_map[index]['metadata']
        metadata['graph']['graph_output'] = out
        metadatas.append(metadata)

    csv_path = os.path.join(e.path, 'test.csv')
    #export_metadatas_csv(metadatas, csv_path)
    
    # ~ task specific metrics
    # In this section we generate the performance metrics and artifacts for models depending on the specific 
    # tasks type because regression and classification tasks need be treated differently.

    num_classes = out_pred.shape[1]
    e.log(f'classification with {num_classes} classes')
    
    # labels_true: (B, )
    labels_true = np.argmax(out_true, axis=-1)
    # labels_pred: (B, )
    labels_pred = np.argmax(out_pred, axis=-1)
    
    acc_value = accuracy_score(labels_true, labels_pred)
    e[f'acc'] = acc_value
    e.log(f' * acc: {acc_value:.3f}')
    
    e.log('plotting confusion matrix...')
    
    cm = confusion_matrix(labels_true, labels_pred)
    
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 5))
    ticklabels = list(e.TARGET_NAMES.values())
    sns.heatmap(
        cm, 
        ax=ax, 
        annot=True, 
        fmt='02d',
        cmap='viridis',
        xticklabels=ticklabels,
        yticklabels=ticklabels,
        linewidths=0,
    )
    fig.savefig(os.path.join(e.path, 'confusion_matrix.pdf'))
    plt.close(fig)
    
    # Only if the classification has exactly 2 clases we can calculate additional metrics for 
    # binary classification as well, such as the AUROC score and the F1 metric.
    if num_classes == 2:
        
        f1_value = f1_score(labels_true, labels_pred)
        e[f'f1'] = f1_value
        e.log(f' * f1: {f1_value:.3f}')
    
        auc_value = roc_auc_score(out_true[:, 1], out_pred[:, 1])
        e[f'auc'] = auc_value
        e.log(f' * auc: {auc_value:.3f}')
        
        # ~ roc curve
        e.log('plotting the AUC curve...')
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
        ax.set_title('Receiver Operating Curve')
        plot_roc_curve(ax, out_true[:, 1], out_pred[:, 1])
        e.commit_fig('roc_curve.png', fig)
        
        # ~ confusion matrix
        e.log('plotting confusion matrix...')
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
        ax.set_title('Confusion Matrix')
        plot_confusion_matrix(ax, labels_true, labels_pred, target_names=list(e.TARGET_NAMES.values()))
        e.commit_fig('confusion_matrix.png', fig)
        
    # ~ evaluating explanations
    e.log('evaluating Megan explanations...')
        
    # out_true np.ndarray: (B, O)
    out_true = np.array([graph['graph_labels'] for graph in graphs_test])
    # out_pred np.ndarray: (B, O)
    out_pred = model.predict_graphs(graphs_test)

    e.log('visualizing the example graphs...')
    graphs_example = [index_data_map[i]['metadata']['graph'] for i in example_indices]
    example_infos: t.List[dict] = model.forward_graphs(graphs_example)
    create_importances_pdf(
        graph_list=graphs_example,
        image_path_list=[index_data_map[i]['image_path'] for i in example_indices],
        node_positions_list=[graph['node_positions'] for graph in graphs_example],
        importances_map={
            'megan': (
                [info['node_importance'] for info in example_infos],
                [info['edge_importance'] for info in example_infos],
            )
        },
        output_path=os.path.join(e.path, 'example_explanations.pdf'),
        plot_node_importances_cb=plot_node_importances_background,
        plot_edge_importances_cb=plot_edge_importances_background,
    )
    
    # ~ explanation fidelity analysis
    # Explanation fidelity is a metric that essentially tells how much a given attributional explanation mask 
    # actually influences the behavior of the model. It is usually measured as the deviation of the model output 
    # if the areas highlighted by the explanation are removed from the input data structure. 
    # The fidelity of the MEGAN model is a special case because it can be calculated for every explanation 
    # channel independently
    
    e.log(f'calculating explanation fidelity...')
    
    # leave_one_out: (B, O, K) numpy
    leave_one_out = model.leave_one_out_deviations(graphs_test)
    fig = plot_leave_one_out_analysis(
        leave_one_out,
        num_channels=e.NUM_CHANNELS,
        num_targets=e.FINAL_UNITS[-1],
    )
    fig.savefig(os.path.join(e.path, 'leave_one_out.pdf'))

    
@experiment.hook('validate_model', default=False, replace=False)
def validate_model(e: Experiment,
                   model: Megan,
                   dataset: dict,
                   ) -> dict:
    
    device = model.device
    model.to('cpu')
    model.eval()
    
    result = {}
    
    graphs = [data['metadata']['graph'] for data in dataset.values()]
    infos = model.forward_graphs(graphs)
    
    out_pred = np.array([info['graph_output'] for info in infos])
    out_true = np.array([graph['graph_labels'] for graph in graphs])
    
    labels_pred = [np.argmax(out) for out in out_pred]
    labels_true = [np.argmax(out) for out in out_true]
    
    acc_value = accuracy_score(labels_true, labels_pred)
    ap_value = average_precision_score(out_true, out_pred)
    auc_value = roc_auc_score(out_true, out_pred)
    
    result['accuracy'] = acc_value
    result['average_precision'] = ap_value
    result['roc_auc'] = auc_value 
    
    model.train()
    model.to(device)
    
    return result, out_true, out_pred

@experiment.hook('plot_example_explanations', default=False, replace=False)
def plot_example_explanations(e: Experiment,
                              model: Megan,
                              dataset: dict,
                              ) -> None:
    """
    This hook will generate a figure that plots the explanations derived from the given model for 
    all the elements contained in the given dataset. Returns the Figure object.
    """
    num_elements = len(dataset)
    fig, rows = plt.subplots(
        nrows=2, 
        ncols=num_elements, 
        figsize=(num_elements * 12, 20), 
        squeeze=False
    )
    
    graphs = [data['metadata']['graph'] for data in dataset.values()]
    infos = model.forward_graphs(graphs)
    
    for i, graph, info, (index, data) in zip(range(num_elements), graphs, infos, dataset.items()):
        
        image_path = data['image_path']
        image = imread(image_path)
        extent = [0, image.shape[0], 0, image.shape[1]]
    
        node_importances = array_normalize(info['node_importance'])
        edge_importances = array_normalize(info['edge_importance'])
    
        for k in range(2):
            
            ax = rows[k][i]
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            ax.imshow(image, extent=extent)
            
            plot_node_importances_background(
                g=graph,
                ax=ax,
                node_positions=graph['node_positions'],
                node_importances=node_importances[:, k],
            )
            
            plot_edge_importances_background(
                g=graph,
                ax=ax,
                node_positions=graph['node_positions'],
                edge_importances=edge_importances[:, k],
            )
            
            if k == 0:
                ax.set_title(f'index: {index}\n'
                             f'true: {graph["graph_labels"]} - pred: {info["graph_output"]}')
                
            if i == 0:
                ax.set_ylabel(f'channel {k} - {e.CHANNEL_INFOS[k]["name"]}')

    return fig


@experiment
def experiment(e: Experiment):
    
    e.log('starting experiment...')
    
    if e.__TESTING__:
        e.log('experiment in testing mode...')
        e.EPOCHS = 10
    
    # ~ loading the test dataset
    # For a chunked dataset, the test set is saved separately from the training set. The test set is saved in the format 
    # of a visual graph dataset.
    
    e.log(f'loading the test dataset @ {e.CHUNKED_DATASET_PATH}...')
    test_path = os.path.join(e.CHUNKED_DATASET_PATH, 'test')
    reader = VisualGraphDatasetReader(
        path=test_path,
        logger=e.logger,
        log_step=1000,
    )
    index_data_map: t.Dict[int, dict] = reader.read()
    indices = list(index_data_map.keys())
    example_graph = list(index_data_map.values())[0]['metadata']['graph']
    e['node_dim'] = example_graph['node_attributes'].shape[1]
    e['edge_dim'] = example_graph['edge_attributes'].shape[1]
    e['output_dim'] = example_graph['graph_labels'].shape[0]
    e.log(f'loaded {len(index_data_map)} test elements')
    
    graphs = [data['metadata']['graph'] for data in index_data_map.values()]    
    data_list = data_list_from_graphs(graphs)
    
    val_path = os.path.join(e.CHUNKED_DATASET_PATH, 'val')
    if os.path.exists(val_path):
        e.log(f'loading the validation dataset @ {val_path}...')
        reader_val = VisualGraphDatasetReader(
            path=val_path,
            logger=e.logger,
            log_step=1000,
        )
        dataset_val = reader_val.read()
        e.log(f'loaded {len(dataset_val)} validation elements')
    else: 
        e.log('no validation dataset found, using the test set for validation...')
        dataset_val = index_data_map
        
    # 02.07.24
    # The external dataset is a very small dataset of a few relatively certainly known elements
    # that can be used to compare model performance across different model versions as this 
    # dataset always stays the same. For better debugging we will actually evaluate on this 
    # dataset cont. during training.
    ext_path = os.path.join(e.CHUNKED_DATASET_PATH, 'ext')
    if os.path.exists(ext_path):
        e.log(f'loading the external dataset @ {ext_path}...')
        reader_ext = VisualGraphDatasetReader(
            path=ext_path,
            logger=e.logger,
            log_step=100,   
        )
        dataset_ext = reader_ext.read()
        e.log(f'loaded {len(dataset_ext)} external elements')
    else:
        e.log('no external dataset found, using the test set instead...')
        dataset_ext = index_data_map
    
    # ~ checking for input problems
    for index, data in index_data_map.items():
        graph = data['metadata']['graph']
        if np.isnan(graph['node_attributes']).any():
            e.log(f' * {index} - nan in node attributes')
        if np.isnan(graph['edge_attributes']).any():
            e.log(f' * {index} - nan in edge attributes')
        
    # ~ model training
    # The model training is the function that is the thing
    
    e.log('constructing chunked dataset...')
    train_path = os.path.join(CHUNKED_DATASET_PATH, 'train')
    
    # For testing we construct the training dataset from the test set that we've already 
    # loaded as that will most likely be the solution which requires the least amount 
    # of *additional* time
    if e.__TESTING__:
        dataset = GraphDataset(index_data_map)
    else:
        dataset = MultiChunkedDataset(path=train_path, num_chunks=3)
    
    train_loader = DataLoader(dataset, batch_size=e.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(data_list, batch_size=e.BATCH_SIZE, shuffle=False)
    
    e['validation_history'] = []
    e['validation_best'] = 0.0
    
    class ValidationCallback(pl.callbacks.Callback):
        
        def __init__(self, ):
            pl.callbacks.Callback.__init__(self)
            self.model_path = os.path.join(e.path, 'model_best.ckpt')
            self.result_path = os.path.join(e.path, 'result_best.json')
            
        def on_train_epoch_end(self, trainer, module):
            
            device = model.device
            model.to('cpu')
            model.eval()
            
            # ~ tracking training variables
            # First of all we are going to track the training metrics, which includes all the different kinds
            # of losses that the MEGAN model is using during the training process.
            e.track('loss_prediction', float(trainer.callback_metrics['loss_pred']))
            e.track('loss_explanation', float(trainer.callback_metrics['loss_expl']))
            e.track('loss_sparsity', float(trainer.callback_metrics['loss_spar']))
            e.track('loss_fidelity', float(trainer.callback_metrics['loss_fid']))
            
            # ~ validation
            # After each training epoch we want to validate the model on the validation set. This is done by
            e.log('validating model...')
            result, out_true, out_pred = e.apply_hook(
                'validate_model',
                model=model,
                dataset=dataset_val,
            )
            e['validation_history'].append(result)
            e.track_many({f'val_{key}': value for key, value in result.items()})
            e.log(f'validation'
                  f' - accuracy: {result["accuracy"]:.2f}' 
                  f' - average precision: {result["average_precision"]:.2f}'
                  f' - auc: {result["roc_auc"]:.2f}')

            # Then we want to compare the validation performance of the model with the previous 
            # best model performance and if the new model is better, we will save it to the disk
            # replacing the previous best model.
            if result['accuracy'] > e['validation_best']:
                e['validation_best'] = result['accuracy']
                e.log(' * new best model found!')
                model.save(self.model_path)
                e.commit_json(self.result_path, result)
                
            # ~ tracking visual artifacts
            # In addition to the numeric metrics we also want to track the evolution of the 
            # models behavior visually by including for example a confusion matrix, the AUROC curve 
            # and some example explanations. These will hopefully be useful for debugging.
            fig, ax = plt.subplots(figsize=(8, 8))
            plot_roc_curve(ax, out_true[:, 1], out_pred[:, 1])
            e.track('val_roc_curve', fig)
            
            fig, ax = plt.subplots(figsize=(8, 8))
            labels_true = np.argmax(out_true, axis=-1)
            labels_pred = np.argmax(out_pred, axis=-1)
            plot_confusion_matrix(ax, labels_true, labels_pred, target_names=list(e.TARGET_NAMES.values()))
            e.track('val_confusion_matrix', fig)
            
            fig = e.apply_hook(
                'plot_example_explanations',
                model=model,
                dataset=dict(list(dataset_val.items())[:4]),
            )
            e.track('val_example_explanations', fig)

            # ~ external dataset
            # We also want to determine the model performance for the external dataset. This is a small 
            # dataset that stays fixed across different model version and contains molecules with 
            # relatively certain labels.
            
            e.log('validating model on external dataset...')
            result, _, _ = e.apply_hook(
                'validate_model',
                model=model,
                dataset=dataset_ext,
            )
            e.track_many({f'ext_{key}': value for key, value in result.items()})
            
            model.to(device)
            model.train()
    
    
    e.log('constructing the model...')
    e.log(f' * units: {e.UNITS}')
    e.log(f' * hidden units: {e.HIDDEN_UNITS}')
    e.log(f' * label smoothing: {e.LABEL_SMOOTHING}')
    e.log(f' * class weights: {e.CLASS_WEIGTHS}')
    e.log(f' * encoder dropout rate: {e.ENCODER_DROPOUT_RATE}')
    e.log(f' * final dropout rate: {e.FINAL_DROPOUT_RATE}')
    e.log(f' * output norm: {e.OUTPUT_NORM}')
    e.log(f' * sparsity factor: {e.SPARSITY_FACTOR}')
    e.log(f' * fidelity factor: {e.FIDELITY_FACTOR}')
    model = Megan(
        node_dim=e['node_dim'],
        edge_dim=e['edge_dim'],
        units=e.UNITS,
        hidden_units=e.HIDDEN_UNITS,
        importance_units=e.IMPORTANCE_UNITS,
        # only if this is a not-None value, the explanation co-training of the model is actually
        # enabled. The explanation co-training works differently for regression and classification tasks
        projection_units=e.PROJECTION_UNITS,
        importance_mode=e.DATASET_TYPE,
        importance_target='node',
        final_units=e.FINAL_UNITS,
        num_channels=e.NUM_CHANNELS,
        importance_factor=e.IMPORTANCE_FACTOR,
        importance_offset=e.IMPORTANCE_OFFSET,
        sparsity_factor=e.SPARSITY_FACTOR,
        fidelity_factor=e.FIDELITY_FACTOR,
        regression_reference=e.REGRESSION_REFERENCE,
        regression_margin=e.REGRESSION_MARGIN,
        attention_aggregation=e.ATTENTION_AGGREGATION,
        prediction_mode=e.DATASET_TYPE,
        prediction_factor=e.PREDICTION_FACTOR,
        normalize_embedding=e.NORMALIZE_EMBEDDING,
        contrastive_factor=e.CONTRASTIVE_FACTOR,
        contrastive_temp=e.CONTRASTIVE_TEMP,
        contrastive_noise=e.CONTRASTIVE_NOISE,
        contrastive_beta=e.CONTRASTIVE_BETA,
        contrastive_tau=e.CONTRASTIVE_TAU,
        learning_rate=e.LEARNING_RATE,
        label_smoothing=e.LABEL_SMOOTHING,
        class_weights=e.CLASS_WEIGTHS,
        encoder_dropout_rate=e.ENCODER_DROPOUT_RATE,
        final_dropout_rate=e.FINAL_DROPOUT_RATE,
        output_norm=e.OUTPUT_NORM,
    )
    
    e.log('starting model training...')
    trainer = pl.Trainer(
        max_epochs=e.EPOCHS,
        # logger=CSVLogger('logs', name='megan'),
        callbacks=ValidationCallback(),
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=test_loader,
    )
    model.eval()
    model.to('cpu')

    # ~ saving the model
    e.log('saving the model...')
    model_path = os.path.join(e.path, 'model.ckpt')
    trainer.save_checkpoint(model_path)
    
    # ~ evaluating the model
    # After the training process is done, we can evaluate the model on the test set.
    e.log('evaluating the model...')
    e.apply_hook(
        'evaluate_model', 
        model=model,
        trainer=trainer, 
        index_data_map=index_data_map
    )


@experiment.analysis
def analysis(e: Experiment):
    
    e.log('staring analysis...')
    
    e.log('loading the model from memory...')
    model_path = os.path.join(e.path, 'model.ckpt')
    model = Megan.load_from_checkpoint(model_path)
    e.log(f'loaded model {model.__class__.__name__}')

    
experiment.run_if_main()