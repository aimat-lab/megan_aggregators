import os
import pathlib
import random
import typing as t

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as ks
from pycomex.experiment import Experiment
from pycomex.util import Skippable
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from kgcnn.data.utils import ragged_tensor_from_nested_numpy
from visual_graph_datasets.data import load_visual_graph_dataset
from visual_graph_datasets.visualization.importances import create_importances_pdf
from visual_graph_datasets.visualization.importances import plot_node_importances_border
from visual_graph_datasets.visualization.importances import plot_edge_importances_border
from graph_attention_student.data import process_graph_dataset
from graph_attention_student.models import Megan
from graph_attention_student.training import NoLoss
from graph_attention_student.training import LogProgressCallback
from graph_attention_student.util import array_normalize
from graph_attention_student.util import binary_threshold
from graph_attention_student.util import render_latex
from graph_attention_student.util import latex_table
from graph_attention_student.util import latex_table_element_mean

PATH = pathlib.Path(__file__).parent.absolute()
mpl.use('TkAgg')
random.seed(1)

# == DATASET PARAMETERS ==
# :param VISUAL_GRAPH_DATASET_PATH:
#       This has to be the path of the folder which contains the visual graph dataset "aggregators_binary"
VISUAL_GRAPH_DATASET_PATH: str = '/media/ssd/Programming/visual_graph_datasets/visual_graph_datasets/experiments/results/generate_molecule_dataset_from_csv_aggregators_binary/initial/aggregators_binary'
# :param CLASS_NAMES:
#       This defines the human-readable names of the two classes of the dataset, which will be used in
#       all the created artifacts
CLASS_NAMES: dict[int, str] = {
    0: 'aggregator',
    1: 'non-aggregator'
}
CLASS_INDICES: t.List[int] = list(CLASS_NAMES.keys())
NUM_CLASSES = len(CLASS_NAMES)
CLASS_OVERSAMPLING_RATIOS: dict[int, int] = {
    0: 30,
    1: 1,
}
# :param NUM_TEST:
#       The number of elements to be selected for the test set. Note that the test set will be drafted from
#       the same number of elements from both classes
NUM_TEST: int = 3000
# :param NUM_EXAMPLES:
#       The number of examples *from the test set*, whose explanations should be visualized in the final PDF
#       If this number is smaller than the test size, elements will be selected randomly.
NUM_EXAMPLES: int = 500
SUBSET: t.Optional[int] = None

# == TRAINING PARAMETERS ==
DEVICE: str = 'cpu:0'
EPOCHS: int = 5
BATCH_SIZE: int = 32

# == MODEL PARAMETERS ==
# These are merely the base parameters
IMPORTANCE_CHANNELS: int = 2

# == CONSENSUS PARAMETERS ==
EXPLANATION_THRESHOLD: float = 0.2
RATIO_DATASET: float = 0.8

# == EXPERIMENT PARAMETERS ==
BASE_PATH = PATH
NAMESPACE = 'results/' + os.path.basename(__file__).strip('.py')
DEBUG = True
TESTING = False
with Skippable(), (e := Experiment(BASE_PATH, NAMESPACE, globals())):

    if TESTING:
        SUBSET = 100_000
        EPOCHS = 5
        NUM_EXAMPLES = 50

    e.info('this experiment trains a consensus model consisting of multiple MEGAN models')

    # ~ loading the dataset
    # "index_data_map" is a dictionary, whose keys are the integer indices of each of the dataset elements
    # and the values are again dictionaries which contain all the important information for one element.
    e.info('loading dataset...')
    metadata_map, index_data_map = load_visual_graph_dataset(
        path=VISUAL_GRAPH_DATASET_PATH,
        logger=e.logger,
        log_step=10_000,
        metadata_contains_index=True,
        subset=SUBSET,
    )
    dataset_size = len(index_data_map)
    index_max = max(index_data_map.keys())

    dataset = [None for _ in range(index_max + 1)]
    # In this dict we want to store separate index lists - one for each possible target class in the
    # dataset. We will later need this information as the basis for the oversampling of underrepresented
    # classes.
    class_dataset_indices: t.Dict[int, t.List[int]] = {i: [] for i in CLASS_INDICES}
    for index, data in index_data_map.items():
        g = data['metadata']['graph']

        g['node_importances'] = np.zeros(shape=(len(g['node_indices']), NUM_CLASSES))
        g['edge_importances'] = np.zeros(shape=(len(g['edge_indices']), NUM_CLASSES))

        class_index = int(np.argmax(g['graph_labels']))

        dataset[index] = g
        class_dataset_indices[class_index].append(index)

    e.info('loaded dataset with the following numbers of elements by class:\n' +
           '\n'.join([f'* {CLASS_NAMES[ci]}: {len(indices)}'
                      for ci, indices in class_dataset_indices.items()]))

    # ~ Creating the train test split

    # We start by creating the test set: We want the test set to have an equal number of elements from all
    # the possible classes so that the final test accuracy rating makes sense.
    # In this step it is important that we actually remove (pop) the indices from the lists so that we can
    # use the remaining ones to create the training dataset.
    test_indices = []
    num_test_class = int(NUM_TEST / NUM_CLASSES)
    for class_index, indices in class_dataset_indices.items():
        random.shuffle(indices)
        test_indices += [indices.pop(0) for _ in range(num_test_class)]

    # First of all we need to apply the oversampling to the dataset
    train_indices = []
    for class_index, oversampling_ratio in CLASS_OVERSAMPLING_RATIOS.items():
        indices = class_dataset_indices[class_index]
        train_indices += indices * int(oversampling_ratio)

    num_example = min(NUM_EXAMPLES, len(test_indices))
    example_indices = random.sample(test_indices, k=num_example)

    e.info(f'created dataset split with '
           f'{len(train_indices)} training elements, '
           f'{len(test_indices)} test elements and '
           f'{len(example_indices)} example elements for visualization')

    e['train_indices'] = train_indices
    e['test_indices'] = test_indices
    e['example_indices'] = example_indices

    with tf.device(DEVICE):

        # ~ Main model loop
        e.info('starting main training loop...')

        @e.hook('model_generator', default=True)
        def model_generator(_e):
            key = 'megan_0'
            model = Megan(
                units=[128, 128, 128],
                dropout_rate=0.2,
                final_units=[128, 64, 32, 2],
                final_activation='softmax',
                final_dropout_rate=0.05,
                importance_channels=IMPORTANCE_CHANNELS,
                importance_factor=1.0,
                importance_multiplier=1.4,
                sparsity_factor=5.0,
            )
            model.compile(
                loss=[
                    ks.losses.CategoricalCrossentropy(),
                    NoLoss(),
                    NoLoss(),
                ],
                loss_weights=[1, 0, 0],
                metrics=[ks.metrics.CategoricalAccuracy()],
                optimizer=ks.optimizers.Adam(learning_rate=0.001),
            )
            yield key, model

            key = 'megan_1'
            model = Megan(
                units=[128, 128, 128],
                dropout_rate=0.2,
                final_units=[128, 64, 32, 2],
                final_activation='softmax',
                final_dropout_rate=0.05,
                importance_channels=IMPORTANCE_CHANNELS,
                importance_factor=1.0,
                importance_multiplier=1.0,
                sparsity_factor=5.0,
            )
            model.compile(
                loss=[
                    ks.losses.CategoricalCrossentropy(),
                    NoLoss(),
                    NoLoss(),
                ],
                loss_weights=[1, 0, 0],
                metrics=[ks.metrics.CategoricalAccuracy()],
                optimizer=ks.optimizers.Adam(learning_rate=0.001),
            )
            yield key, model

            key = 'megan_2'
            model = Megan(
                units=[128, 128, 128],
                dropout_rate=0.2,
                final_units=[128, 64, 32, 2],
                final_activation='softmax',
                final_dropout_rate=0.05,
                importance_channels=IMPORTANCE_CHANNELS,
                importance_factor=1.0,
                importance_multiplier=0.8,
                sparsity_factor=5.0,
            )
            model.compile(
                loss=[
                    ks.losses.CategoricalCrossentropy(),
                    NoLoss(),
                    NoLoss(),
                ],
                loss_weights=[1, 0, 0],
                metrics=[ks.metrics.CategoricalAccuracy()],
                optimizer=ks.optimizers.Adam(learning_rate=0.001),
            )
            yield key, model

            key = 'megan_3'
            model = Megan(
                units=[128, 128, 128],
                dropout_rate=0.2,
                final_units=[128, 64, 32, 2],
                final_activation='softmax',
                final_dropout_rate=0.05,
                importance_channels=IMPORTANCE_CHANNELS,
                importance_factor=1.0,
                importance_multiplier=1.6,
                sparsity_factor=5.0,
            )
            model.compile(
                loss=[
                    ks.losses.CategoricalCrossentropy(),
                    NoLoss(),
                    NoLoss(),
                ],
                loss_weights=[1, 0, 0],
                metrics=[ks.metrics.CategoricalAccuracy()],
                optimizer=ks.optimizers.Adam(learning_rate=0.001),
            )
            yield key, model

            key = 'megan_4'
            model = Megan(
                units=[128, 128, 128],
                dropout_rate=0.2,
                final_units=[128, 64, 32, 2],
                final_activation='softmax',
                final_dropout_rate=0.05,
                importance_channels=IMPORTANCE_CHANNELS,
                importance_factor=1.0,
                importance_multiplier=1.6,
                sparsity_factor=5.0,
            )
            model.compile(
                loss=[
                    ks.losses.CategoricalCrossentropy(),
                    NoLoss(),
                    NoLoss(),
                ],
                loss_weights=[1, 0, 0],
                metrics=[ks.metrics.CategoricalAccuracy()],
                optimizer=ks.optimizers.Adam(learning_rate=0.001),
            )
            yield key, model

        e['keys'] = []
        for key, model in e.apply_hook('model_generator'):
            e['key'] = key
            e['keys'].append(key)
            e.info(f'starting training for model with key "{key}"...')
            key_path = os.path.join(e.path, key)
            os.mkdir(key_path)

            e.info(f'converting dataset into tensors...')
            e.info(f'using only {RATIO_DATASET} of training sample for the current model!')
            train_indices_sub = random.sample(train_indices, k=int(RATIO_DATASET * len(train_indices)))
            x_train, y_train, x_test, y_test = process_graph_dataset(
                dataset=dataset,
                train_indices=train_indices_sub,
                test_indices=test_indices,
                use_graph_attributes=False,
                use_importances=True,
            )

            e.info('starting model training')
            hist = model.fit(
                x=x_train,
                y=y_train,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                verbose=0,
                validation_data=(x_test, y_test),
                validation_freq=1,
                callbacks=[
                    LogProgressCallback(
                        logger=e.logger,
                        identifier='val_output_1_categorical_accuracy',
                        epoch_step=1,
                    )
                ]
            )
            history = hist.history
            e[f'history/{key}'] = history

            num_params = model.count_params()
            e[f'num_params/{key}'] = num_params
            e[f'model/{key}'] = model

            e.info(f'finished training for model "{key}" with {num_params} parameters')

            # ~ Evaluating the prediction performance
            e.info('evaluating on test set...')
            out_pred, ni_pred, ei_pred = [v.numpy() for v in model(x_test)]
            out_true, _, _ = y_test

            # First of all we are going to save all the predictions into the experiment storage
            for c, index in enumerate(test_indices):
                e[f'out/pred/{key}/{index}'] = out_pred[c]
                e[f'out/true/{index}'] = out_true[c]

                ni = array_normalize(ni_pred[c])
                ei = array_normalize(ei_pred[c])
                e[f'ni/{key}/{index}'] = ni
                e[f'ei/{key}/{index}'] = ei

                ni_binary = binary_threshold(ni, threshold=0.5)
                ei_binary = binary_threshold(ei, threshold=0.5)
                node_sparsity = np.mean(ni_binary)
                edge_sparsity = np.mean(ei_binary)
                e[f'node_sparsity/{key}/{index}'] = node_sparsity
                e[f'edge_sparsity/{key}/{index}'] = edge_sparsity

            out_pred_classes = [np.argmax(v) for v in out_pred]
            out_true_classes = [np.argmax(v) for v in out_true]
            acc = accuracy_score(out_true_classes, out_pred_classes)

            auc = roc_auc_score(out_true.flatten(), out_pred.flatten())

            e.info(f'prediction metrics:\n'
                   f' * acc: {acc:.3f}\n'
                   f' * auc: {auc:.3f}\n')

            e[f'acc/{key}'] = acc
            e[f'auc/{key}'] = auc

            # ~ Evaluating the explanations

            e.info('calculating fidelity...')

            # First of all we need to calculate the masked output deviations for each of the channels
            # and each of the elements in the dataset.
            for i in range(IMPORTANCE_CHANNELS):

                # To do that we construct the corresponding masks here. For each channel we construct the
                # mask such that only that channel will be removed from the final pooling operation.
                node_masks = []
                for c, index in enumerate(test_indices):
                    ni = ni_pred[c]
                    node_mask = np.ones_like(ni)
                    node_mask[:, i] = 0
                    node_masks.append(node_mask)

                node_masks_tensor = ragged_tensor_from_nested_numpy(node_masks)
                # Using these tensors we can query the model again to get the modified predictions outputs
                out_mod, _, _ = [v.numpy() for v in model(x_test, node_importances_mask=node_masks_tensor)]
                for c, index in enumerate(test_indices):
                    value = (out_pred[c][i] - out_mod[c][i])
                    e[f'fidelity_ast_contribution/{key}/{index}/{i}'] = value

            # Now we merge all the individual channel contributions into the final fidelity value
            for c, index in enumerate(test_indices):
                e[f'fidelity_ast/{key}/{index}'] = np.sum([
                    e[f'fidelity_ast_contribution/{key}/{index}/{i}']
                    for i in range(IMPORTANCE_CHANNELS)
                ])

            fidelities = [e[f'fidelity_ast/{key}/{index}'] for index in test_indices]
            e.info(f'fidelity_ast results:\n'
                   f' * median: {np.median(fidelities):.3f}\n'
                   f' * mean: {np.mean(fidelities):.3f}\n'
                   f' * std: {np.std(fidelities):.3f}\n')

            # ~ Saving the model
            e.info('saving model...')
            model_path = os.path.join(key_path, 'model')
            e[f'model_path/{key}'] = model_path
            model.save(model_path)


with Skippable(), e.analysis:

    e.info('starting analysis...')

    e.info('loading the dataset...')
    # First of all we need to load the dataset again here so that we can later on access the images for the
    # visualizations
    metadata_map, index_data_map = load_visual_graph_dataset(
        path=VISUAL_GRAPH_DATASET_PATH,
        logger=e.logger,
        log_step=10_000,
        metadata_contains_index=True,
        subset=SUBSET,
    )

    e.info('evaluating consensus of models...')
    test_indices = e['test_indices']
    keys = e['keys']
    for index in test_indices:
        g = index_data_map[index]['metadata']['graph']

        # Here we calculate the consensus target prediction as a weighted sum over all the individual
        # predictions, where the weight is the final test set accuracy of each model.
        values = []
        weights = []
        for key in keys:
            weights.append(float(e[f'acc/{key}']))
            values.append(e[f'out/pred/{key}/{index}'])

        value_avg = np.average(values, axis=0, weights=weights)
        e[f'out/pred/consensus/{index}'] = value_avg

        # Here we create the consensus binary explanations by doing a threshold majority vote for each
        # individual element (node/edge)
        ni = np.mean(np.array([e[f'ni/{key}/{index}'] for key in keys]), axis=0)
        ei = np.mean(np.array([e[f'ei/{key}/{index}'] for key in keys]), axis=0)
        ni = (ni >= EXPLANATION_THRESHOLD).astype(float)
        ei = (ei >= EXPLANATION_THRESHOLD).astype(float)
        e[f'ni/consensus/{index}'] = ni
        e[f'ei/consensus/{index}'] = ei

        e[f'node_sparsity/consensus/{index}'] = np.mean(ni)
        e[f'edge_sparsity/consensus/{index}'] = np.mean(ei)

        e[f'fidelity_ast/consensus/{index}'] = 0

    out_pred_class = [np.argmax(e[f'out/pred/consensus/{index}']) for index in test_indices]
    out_true_class = [np.argmax(e[f'out/true/{index}']) for index in test_indices]
    acc = accuracy_score(out_true_class, out_pred_class)
    e['acc/consensus'] = acc
    e.info(f'consensus accuracy: {acc:.3f}')

    # ~ Visualizing the examples
    e.info('creating visualizations...')
    keys = ['consensus'] + keys

    indices = example_indices
    graph_list = [index_data_map[i]['metadata']['graph'] for i in indices]
    image_path_list = [index_data_map[i]['image_path'] for i in indices]
    node_positions_list = [g['node_positions'] for g in graph_list]
    pdf_path = os.path.join(e.path, f'explanations_consensus.pdf')
    labels_list = [(f'{index_data_map[i]["metadata"]["smiles"]}\n'
                    f'pred: {np.round(e[f"out/pred/consensus/{i}"], 2)} - '
                    f'true: {e[f"out/true/{i}"]}\n')
                   for i in indices]
    create_importances_pdf(
        graph_list=graph_list,
        image_path_list=image_path_list,
        node_positions_list=node_positions_list,
        labels_list=labels_list,
        importances_map={
            key: (
                [e[f'ni/{key}/{index}'] for index in indices],
                [e[f'ei/{key}/{index}'] for index in indices]
            )
            for key in keys
        },
        output_path=pdf_path,
        importance_channel_labels=list(CLASS_NAMES.values()),
        plot_node_importances_cb=plot_node_importances_border,
        plot_edge_importances_cb=plot_edge_importances_border,
        normalize_importances=True,
        logger=e.logger,
        log_step=100,
    )

    # ~ Latex table with the results
    e.info('rendering latex table with the results...')
    column_names = [
        r'Model Key',
        r'$\text{Accuracy} \downarrow$',
        r'$\text{Node Sparsity} \downarrow$',
        r'$\text{Edge Sparsity} \downarrow$',
        r'$\text{Fidelity*} \uparrow$',
    ]
    rows = []
    for key in keys:
        row = []
        row.append(key.replace('_', ' '))
        row.append([e[f'acc/{key}']])
        row.append(list(e[f'node_sparsity/{key}'].values()))
        row.append(list(e[f'edge_sparsity/{key}'].values()))
        row.append(list(e[f'fidelity_ast/{key}'].values()))

        rows.append(row)

    content, table = latex_table(
        column_names=column_names,
        rows=rows,
        list_element_cb=latex_table_element_mean,
    )
    e.commit_raw('table.tex', table)
    output_path = os.path.join(e.path, 'table.pdf')
    render_latex({'content': table}, output_path=output_path)
    e.info('rendered latex table with the results')



