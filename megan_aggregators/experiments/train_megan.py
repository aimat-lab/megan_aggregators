"""
This module trains a MEGAN model on the "aggregators_binary" dataset. The experiment generates some initial
dataset analysis, performs the training and ultimately visualizes the explanations created by the model on
the test set.

CHANGELOG

0.1.0 - 25.02.2023 - initial version

0.2.0 - 27.02.2023 - Added the calculation of fidelity and also split the explanation visualizations into
two PDF files: One which contains onl the correct predictions and one which contains the very bad
predictions. Also fixed the model saving and loading process.
"""
import os
import random
import pathlib
import typing as t

import tensorflow as tf
import tensorflow.keras as ks
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.spatial.distance import cityblock
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from pycomex.experiment import Experiment
from pycomex.util import Skippable
from visual_graph_datasets.data import load_visual_graph_dataset
from visual_graph_datasets.visualization.importances import create_importances_pdf
from visual_graph_datasets.visualization.importances import plot_node_importances_border
from visual_graph_datasets.visualization.importances import plot_edge_importances_border
from graph_attention_student.data import process_graph_dataset
from graph_attention_student.models import Megan
from graph_attention_student.training import LogProgressCallback
from graph_attention_student.training import NoLoss
from graph_attention_student.util import array_normalize
from kgcnn.data.utils import ragged_tensor_from_nested_numpy

mpl.use('TkAgg')
random.seed(3)
PATH = pathlib.Path(__file__).parent.absolute()

SHORT_DESCRIPTION = (
    'This will be the short description when listing the experiments from the command line'
)

# == DATASET PARAMETERS ==
# :param VISUAL_GRAPH_DATASET_PATH:
#       This has to be the path of the folder which contains the visual graph dataset "aggregators_binary"
VISUAL_GRAPH_DATASET_PATH: str = '/media/ssd/Programming/visual_graph_datasets/visual_graph_datasets/experiments/results/generate_molecule_dataset_from_csv_aggregators_binary/initial/aggregators_binary'

CLASS_NAMES: dict[int, str] = {
    0: 'aggregator',
    1: 'non-aggregator'
}
# :param NUM_TEST:
#       The number of elements to be selected for the test set. Note that the test set will be drafted from
#       the same number of elements from both classes
NUM_TEST: int = 3000
# :param NUM_EXAMPLES:
#       The number of examples *from the test set*, whose explanations should be visualized in the final PDF
#       If this number is smaller than the test size, elements will be selected randomly.
NUM_EXAMPLES: int = 300

# == MODEL PARAMETERS ==
# :param UNITS:
#       The number of hidden units in each of the convolutional layers of the network. The number of
#       elements in this list determines the convolutional depth of the network
UNITS = [32, 32, 32]
# :param DROPOUT_RATE:
#       Dropout rate for the node embeddings between each convolutional layer.
DROPOUT_RATE = 0.1
# :param IMPORTANCE_FACTOR:
#       This is the weighting coefficient of the explanation co-training loss. If this value is at 1.0
#       then the main prediction loss and the explanation loss have the same weight.
IMPORTANCE_FACTOR = 1.0
# :param IMPORTANCE_MULTIPLIER:
#       This is a hyper parameter of the additional explanation training procedure performed for MEGAN
#       models when (IMPORTANCE_FACTOR != 0). This parameter determines what the expected size of the
#       explanations is. If this parameter is reduced, the explanations will generally consist of less
#       elements (nodes, edges). Vice versa, a larger value will make explanations consist of more elements.
IMPORTANCE_MULTIPLIER = 1.8
# :param IMPORTANCE_UNITS:
#       The number of hidden units in the explanation-only part of the network. Generally advised to keep
#       this part shallow.
IMPORTANCE_UNITS = [32]
# :param SPARSITY_FACTOR:
#       The coefficient for the sparsity regularization loss.
SPARSITY_FACTOR = 6.0
# :param CONCAT_HEADS:
#       A boolean flag which determines how the MEGAN attention heads behave. For True, the individual
#       results of the attention heads are concatenated. For False, the individual results are averaged.
#       Generally False recommended as it leads to less overfitting.
CONCAT_HEADS = False
# :param FINAL_UNITS:
#       The hidden units of the MLP tail end of the prediction network. Note that the last value of this
#       list MUST be the same as the size of the target value vector for the dataset (aka number of
#       classes)
FINAL_UNITS = [32, 16, 2]

# == TRAINING PARAMETERS ==
# :param DEVICE:
#       The tensorflow device ID to be used for the training. CPU is advised as this dataset is very large
#       and needs a lot of memory, which most GPUs with average amounts of video memory cant handle.
DEVICE = 'cpu:0'
# :param EPOCHS:
#       The number of epochs to train the network for
EPOCHS = 10
# :param BATCH_SIZE:
#       The number of elements to be used in one batch during the training process. smaller batch sizes
#       have turned out to work better.
BATCH_SIZE = 32
# :param OPTIMIZER_CB:
#       A callback function with no arguments, which should return a valid keras Optimizer object, which
#       will be used to train the network.
OPTIMIZER_CB = lambda: ks.optimizers.Adam(learning_rate=0.001)

# == EVALUATION PARAMETERS ==
LOG_STEP_EVAL = 10_000
NUM_BINS = 50
COLOR_NEUTRAL = 'gray'
CLASS_COLORS = {
    0: '#FAA7D2',
    1: '#D6A8FA'
}

# == EXPERIMENT PARAMETERS ==
BASE_PATH = PATH
NAMESPACE = 'results/train_megan'
DEBUG = True
with Skippable(), (e := Experiment(BASE_PATH, NAMESPACE, globals())):
    # "e.info" should be used instead of "print". It will use python's "logging" module to not only
    # print the content ot the console but also write it into a log file at the same time.
    e.info('starting experiment...')

    e.info('starting analysis...')
    e.status()

    # ~ loading the dataset
    # "index_data_map" is a dictionary, whose keys are the integer indices of each of the dataset elements
    # and the values are again dictionaries which contain all the important information for one element.
    metadata_map, index_data_map = load_visual_graph_dataset(
        path=VISUAL_GRAPH_DATASET_PATH,
        logger=e.logger,
        log_step=LOG_STEP_EVAL,
        metadata_contains_index=True,
        subset=40000,
    )
    dataset_size = len(index_data_map)
    index_max = max(index_data_map.keys())
    e.status()

    # ~ analyzing the dataset
    # Before actually training the model, we want to analyze some properties of the dataset to get a
    # a feeling for it.
    e.info('analyzing dataset properties...')

    # 1. The distribution of the graph sizes
    e.info('plotting graph size distribution...')
    graph_sizes = [len(data['metadata']['graph']['node_indices']) for data in index_data_map.values()]
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
    ax.set_title(f'Graph Size Distribution\n'
                 f'{dataset_size} Elements')
    ax.hist(
        x=graph_sizes,
        bins=NUM_BINS,
        color=COLOR_NEUTRAL,
    )
    e.commit_fig('graph_size_distribution.pdf', fig)
    plt.close(fig)

    # 2. The class distribution
    e.info('plotting class distribution...')
    targets = [np.argmax(data['metadata']['target']) for data in index_data_map.values()]
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
    ax.set_title(f'Class Label Distribution\n'
                 f'{dataset_size} Elements')
    ax.hist(
        x=targets,
        bins=2,
        color=COLOR_NEUTRAL,
    )
    e.commit_fig('class_label_distribution.pdf', fig)
    plt.close(fig)

    # 3. The graph size distribution for both of the classes separately
    e.info('plotting graph size distribution for each class...')
    graph_sizes_map = {
        0: [],
        1: []
    }
    indices_map = {
        0: [],
        1: []
    }
    for index, data in index_data_map.items():
        target = int(np.argmax(data['metadata']['target']))
        graph_size = len(data['metadata']['graph']['node_indices'])
        graph_sizes_map[target].append(graph_size)
        indices_map[target].append(index)

    fig, rows = plt.subplots(ncols=1, nrows=2, figsize=(10, 20), squeeze=False)
    fig.suptitle('graph size distribution by class label')
    x_lim = (min(graph_sizes), max(graph_sizes))
    for c, row in enumerate(rows):
        ax = row[0]
        ax.set_title(f'class {c} - "{CLASS_NAMES[c]}"')
        ax.hist(
            x=graph_sizes_map[c],
            bins=NUM_BINS,
            range=x_lim,
            color=CLASS_COLORS[c]
        )
        ax.set_xlim(x_lim)

    e.commit_fig('graph_size_by_class.pdf', fig)
    plt.close(fig)

    # ~ processing the dataset
    # This dataset ultimately has to be converted into the tensorflow tensors to train the model. The first
    # step in this is to collect all the actual graph representations in a list
    dataset: t.List[dict] = [None for _ in range(index_max + 1)]
    for index, data in index_data_map.items():
        g = data['metadata']['graph']
        g['graph_labels'] = data['metadata']['target']
        g['node_importances'] = np.zeros(shape=(len(g['node_indices']), 2))
        g['edge_importances'] = np.zeros(shape=(len(g['edge_indices']), 2))
        dataset[index] = g

    # Here we decide on which elements should belong to the train and the test set.
    e.info('determining train test split...')
    dataset_indices = list(index_data_map.keys())
    dataset_indices_set = set(dataset_indices)
    # For the test set we would like to have a balanced distribution of classes, which means we are going
    # to sample exactly half of the indices from each class' elements
    test_indices_0 = random.sample(indices_map[0], k=int(NUM_TEST / 2))
    test_indices_1 = random.sample(indices_map[1], k=int(NUM_TEST / 2))
    test_indices = test_indices_0 + test_indices_1
    test_indices_set = set(test_indices)
    train_indices_set = dataset_indices_set.difference(test_indices_set)
    train_indices = list(train_indices_set)
    # oversampling the minority class by factor 10
    train_indices_0 = [index for index in indices_map[0] if index not in test_indices_0]
    for i in range(30):
        train_indices += train_indices_0

    # The "examples" are those elements of the test set for which the visualizations will be created
    num_examples = min(NUM_EXAMPLES, NUM_TEST)
    example_indices = random.sample(test_indices, k=num_examples)
    e['test_indices'] = test_indices
    e['example_indices'] = example_indices

    with tf.device(DEVICE):
        # Now using the indices split and the dataset list we can now create the tensors
        e.info('converting dataset to tensors...')
        x_train, y_train, x_test, y_test = process_graph_dataset(
            dataset=dataset,
            train_indices=train_indices,
            test_indices=test_indices
        )
        e.status()

        # ~ Setting up the model
        e.info('setting up model...')
        model: ks.models.Model = Megan(
            units=UNITS,
            use_bias=True,
            dropout_rate=DROPOUT_RATE,
            use_edge_features=True,
            importance_units=IMPORTANCE_UNITS,
            importance_channels=2,
            importance_factor=IMPORTANCE_FACTOR,
            importance_multiplier=IMPORTANCE_MULTIPLIER,
            sparsity_factor=SPARSITY_FACTOR,
            concat_heads=CONCAT_HEADS,
            final_units=FINAL_UNITS,
            final_activation='softmax',
            use_graph_attributes=False,
        )

        model.compile(
            optimizer=OPTIMIZER_CB(),
            metrics=[ks.metrics.CategoricalAccuracy()],
            loss=[
                ks.losses.CategoricalCrossentropy(),
                NoLoss(),
                NoLoss(),
            ],
            loss_weights=[
                1,
                0,
                0
            ],
        )

        e.info('starting model training...')
        hist = model.fit(
            x=x_train,
            y=y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[
                LogProgressCallback(e.logger, 'val_output_1_categorical_accuracy', 1)
            ],
            validation_freq=1,
            validation_data=(x_test, y_test),
            verbose=0,
        )
        history = hist.history
        e['history'] = history
        e['weights'] = model.get_weights()

        # ~ Evaluating the model performance on the test set
        e.info('evaluating model...')
        # First we retrieve all the test predictions of the model as numpy arrays
        out_pred, ni_pred, ei_pred = [v.numpy() for v in model(x_test, training=False)]
        out_true, _, _ = y_test

        acc = history['val_output_1_categorical_accuracy'][-1]
        auc = roc_auc_score(out_true.flatten(), out_pred.flatten())
        e['auc'] = auc
        e.info(f'final test performance '
               f' - acc: {acc:.2f} '
               f'- auc: {auc:.2f}')

        e.info('saving test set results...')
        for c, index in enumerate(test_indices):
            out = out_pred[c]
            ni = array_normalize(ni_pred[c])
            ei = array_normalize(ei_pred[c])
            e[f'out/pred/{index}'] = out
            e[f'out/true/{index}'] = index_data_map[index]['metadata']['target']
            e[f'ni/{index}'] = np.array(ni)
            e[f'ei/{index}'] = np.array(ei)

        # Calculating the fidelities
        e.info('calculating fidelity...')
        for k, name in CLASS_NAMES.items():
            e.info(f'calculating deviations for masking of channel {k} - {name}...')
            # First of all we need to construct the appropriate binary masks from each of the explanations
            # of the current channel and then we need to query the model again with those masks.
            node_masks = []
            for c, index in enumerate(test_indices):
                ni = e[f'ni/{index}']
                node_mask = np.ones_like(ni)
                node_mask[:, k] = 0
                node_masks.append(node_mask)

            node_masks_tensor = ragged_tensor_from_nested_numpy(node_masks)
            out_mod, _, _ = model(x_test, node_importances_mask=node_masks_tensor)
            out_mod = out_mod.numpy()
            for c, index in enumerate(test_indices):
                mod = out_mod[c]
                e[f'out/mod/{k}/{index}'] = e[f'out/pred/{index}'][k] - mod[k]

        for c, index in enumerate(test_indices):
            value = float(sum(e[f'out/mod/{k}/{index}'] for k in CLASS_NAMES.keys()))
            e[f'fid/{index}'] = value

        # From the individual local perturbations of the predicted output we can now calculate the overall
        # fidelity by summing over all the individual ones
        fidelity = np.mean([e[f'fid/{index}'] for index in test_indices])
        e['fidelity'] = fidelity
        e.info(f'average explanation fidelity: {fidelity:.2f}')

        # ~ Visualizing results
        e.info('visualizing results...')

        # 1. The training curves of the model
        epochs = list(range(EPOCHS))
        fig, rows = plt.subplots(ncols=2, nrows=1, figsize=(20, 10), squeeze=False)
        ax_loss = rows[0][0]
        ax_loss.set_title('loss over training')
        ax_loss.set_xlabel('epochs')
        ax_loss.set_ylabel('categorical cross-entropy loss')
        ax_loss.plot(
            epochs, history['output_1_loss'],
            color=COLOR_NEUTRAL,
            ls='-',
            label='train'
        )
        ax_loss.plot(
            epochs, history['val_output_1_loss'],
            color=COLOR_NEUTRAL,
            ls='--',
            label='test'
        )
        ax_loss.legend()

        ax_metric = rows[0][1]
        ax_metric.set_title('accuracy over training')
        ax_metric.set_xlabel('epochs')
        ax_metric.set_ylabel('categorical accuracy')
        ax_metric.plot(
            epochs, history['output_1_categorical_accuracy'],
            color=COLOR_NEUTRAL,
            ls='-',
            label='train',
        )
        ax_metric.plot(
            epochs, history['val_output_1_categorical_accuracy'],
            color=COLOR_NEUTRAL,
            ls='--',
            label='test'
        )
        ax_metric.legend()

        e.commit_fig('training.pdf', fig)
        plt.close(fig)

        # 2. A confusion matrix
        labels_pred = [np.argmax(v) for v in out_pred]
        labels_true = [np.argmax(v) for v in out_true]
        cm = confusion_matrix(labels_true, labels_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=list(CLASS_NAMES.values()))
        disp.plot()
        e.commit_fig('confusion_matrix.pdf', disp.figure_)
        plt.close(disp.figure_)

        # 3. The main explanations
        # However, for these we want to filter for the elements, where the model actually makes a correct
        # prediction
        # This complicated expression filters all the test indices for only the correct predictions and
        # also sorts the elements by explanation fidelity
        correct_indices, _ = zip(*sorted(
            [(index, e[f'fid/{index}'])
             for index in example_indices
             if np.argmax(e[f'out/true/{index}']) == np.argmax(e[f'out/pred/{index}'])],
            key=lambda tupl: tupl[1],
            reverse=True,
        ))
        e.info(f'visualizing {len(correct_indices)} correct elements...')
        ni_correct = [e[f'ni/{index}'] for index in correct_indices]
        ei_correct = [e[f'ei/{index}'] for index in correct_indices]
        # Then we need to assemble various data structures for these elements here, which are needed for the
        # function which then creates the PDF with the explanation visualizations.
        graph_list = [index_data_map[i]['metadata']['graph'] for i in correct_indices]
        image_path_list = [index_data_map[i]['image_path'] for i in correct_indices]
        node_positions_list = [g['node_positions'] for g in graph_list]
        pdf_path = os.path.join(e.path, 'explanations_correct_predictions.pdf')
        labels_list = [(f'{index_data_map[i]["metadata"]["smiles"]}\n'
                        f'pred: {np.round(e[f"out/pred/{i}"], 2)} - '
                        f'true: {e[f"out/true/{i}"]}\n'
                        f'fidelity ch0: {e[f"out/mod/0/{i}"]:.2f}\n'
                        f'fidelity ch1: {e[f"out/mod/1/{i}"]:.2f}\n'
                        f'overall fidelity: {e[f"fid/{i}"]:.2f}')
                       for i in correct_indices]
        create_importances_pdf(
            graph_list=graph_list,
            image_path_list=image_path_list,
            node_positions_list=node_positions_list,
            labels_list=labels_list,
            importances_map={
                'megan': (ni_correct, ei_correct)
            },
            output_path=pdf_path,
            importance_channel_labels=list(CLASS_NAMES.values()),
            plot_node_importances_cb=plot_node_importances_border,
            plot_edge_importances_cb=plot_edge_importances_border,
            normalize_importances=True,
            logger=e.logger,
            log_step=100,
        )

        # 4. Explanations of bad elements
        # Another interesting set of explanations are the explanations for those elements of the test set
        # which are especially *false* aka where the model confidently predicts the wrong class
        e.info('visualizing explanations for particularly bad predictions...')
        # This somewhat complicated expression filters out indices of the elements of the particularly bad
        # predictions and sorts them by how bad they are.
        bad_indices, _ = zip(*sorted(
            [(index, dist)
             for index in test_indices
             if (dist := cityblock(e[f'out/pred/{index}'], e[f'out/true/{index}']) > 1.8)],
            key=lambda tupl: tupl[1],
            reverse=True,
        ))
        e.info(f'visualizing for {len(bad_indices)} samples')
        ni_bad = [e[f'ni/{index}'] for index in bad_indices]
        ei_bad = [e[f'ei/{index}'] for index in bad_indices]
        graph_list = [index_data_map[i]['metadata']['graph'] for i in bad_indices]
        image_path_list = [index_data_map[i]['image_path'] for i in bad_indices]
        node_positions_list = [g['node_positions'] for g in graph_list]
        pdf_path = os.path.join(e.path, 'explanations_bad_predictions.pdf')
        labels_list = [(f'{index_data_map[i]["metadata"]["smiles"]}\n'
                        f'pred: {np.round(e[f"out/pred/{i}"], 2)} - '
                        f'true: {e[f"out/true/{i}"]}')
                       for i in bad_indices]
        create_importances_pdf(
            graph_list=graph_list,
            image_path_list=image_path_list,
            node_positions_list=node_positions_list,
            labels_list=labels_list,
            importances_map={
                'megan': (ni_bad, ei_bad)
            },
            output_path=pdf_path,
            importance_channel_labels=list(CLASS_NAMES.values()),
            plot_node_importances_cb=plot_node_importances_border,
            plot_edge_importances_cb=plot_edge_importances_border,
            normalize_importances=True,
            logger=e.logger,
            log_step=100,
        )

        # ~ Saving the model
        model_path = os.path.join(e.path, 'model')
        e['model_path'] = model_path
        model.save(model_path)


# == ANALYSIS ==
with Skippable(), e.analysis:

    from graph_attention_student.keras import CUSTOM_OBJECTS

    # Here we make sure that the model can be loaded from it's persistent representation on the disk
    e.info('loading model from persistent representation...')
    with ks.utils.custom_object_scope(CUSTOM_OBJECTS):
        model = ks.models.load_model(e['model_path'])