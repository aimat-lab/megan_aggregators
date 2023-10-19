import os
import csv
import pathlib
import random
import tempfile
import typing as t
from collections import Counter

import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import visual_graph_datasets.typing as tv
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace
from visual_graph_datasets.processing.molecules import MoleculeProcessing
from visual_graph_datasets.models import PredictGraphMixin
from visual_graph_datasets.visualization.base import create_frameless_figure
from visual_graph_datasets.visualization.base import draw_image
from visual_graph_datasets.visualization.importances import plot_node_importances_border
from visual_graph_datasets.visualization.importances import plot_edge_importances_border
from visual_graph_datasets.data import load_visual_graph_element
from vgd_counterfactuals.base import CounterfactualGenerator
from vgd_counterfactuals.generate.molecules import get_neighborhood
from vgd_counterfactuals.visualization import create_counterfactual_pdf

from megan_aggregators.utils import load_model
from megan_aggregators.utils import load_processing


mpl.use('TkAgg')
PATH = pathlib.Path(__file__).parent.absolute()
DATASET_PATH: str = os.path.join(PATH, 'assets', 'fda.csv')
SMILES_COLUMN: str = 'smiles'
NUM_EXAMPLES: int = 5
CLASS_NAME_MAP = {
    0: 'aggregator',
    1: 'non-aggregator'
}

DEVICE: str = 'cpu:0'
COUNTERFACTUAL_RANGE: int = 2

IMAGE_WIDTH: int = 1000
IMAGE_HEIGHT: int = 1000

__DEBUG__ = True


@Experiment(base_path=folder_path(__file__),
            namespace=file_namespace(__file__),
            glob=globals())
def experiment(e: Experiment):

    context = tf.device(DEVICE)
    context.__enter__()

    e.log('loading the standard model...')
    model: PredictGraphMixin = load_model()

    e.log('loading the processing facilities...')
    processing: MoleculeProcessing = load_processing()

    e.log('loading dataset from CSV file...')
    graphs: t.List[tv.GraphDict] = []
    with open(DATASET_PATH) as file:
        dict_reader = csv.DictReader(file)
        for c, row in enumerate(dict_reader):
            smiles = row[SMILES_COLUMN]
            graph = processing.process(smiles)
            graph.update(row)
            graphs.append(graph)

            if c % 100 == 0:
                e.log(f' * {c} done - none')

    e.log('making predictions...')
    predictions = model.predict_graphs(graphs)
    labels_pred = [np.argmax(out) for out, ni, ei in predictions]
    counter = Counter(labels_pred)
    for key, value in counter.items():
        e.log(f' * class {key}: {value} elements')

    e.log('creating prediction CSV...')
    csv_path = os.path.join(e.path, 'predictions.csv')
    prediction_confidence_map = {0: [], 1: []}
    with open(csv_path, mode='w') as file:
        dict_writer = csv.DictWriter(file, fieldnames=['smiles', 'label', 'agg', 'non_agg'])
        dict_writer.writeheader()
        for graph, (out, ni, ei) in zip(graphs, predictions):
            label = e.CLASS_NAME_MAP[np.argmax(out)]

            prediction_confidence_map[np.argmax(out)].append(np.max(out))
            dict_writer.writerow({
                'smiles': graph[SMILES_COLUMN],
                'label': label,
                'agg': f'{out[0]:.3f}',
                'non_agg': f'{out[1]:.3f}'
            })

    e.log('plotting confidence distributions...')
    fig, (ax_agg, ax_non) = plt.subplots(ncols=1, nrows=2, figsize=(10, 20))
    ax_agg.set_title('confidence distribution for "aggregator" predictions')
    ax_agg.hist(prediction_confidence_map[0], bins=20, color='lightgray')
    ax_agg.set_xlim([0.5, 1.0])
    ax_non.set_title('confidence distribution for "non-aggregator" predictions')
    ax_non.hist(prediction_confidence_map[1], bins=20, color='lightgray')
    ax_non.set_xlim([0.5, 1.0])
    e.commit_fig('confidence_distribution.pdf', fig)

    e.log('creating a CSV with only predicted aggregators...')
    aggregators_csv_path = os.path.join(e.path, 'aggregators.csv')
    with open(aggregators_csv_path, mode='w') as file:
        dict_writer = csv.DictWriter(file, fieldnames=['smiles'])
        dict_writer.writeheader()
        for graph, label in zip(graphs, labels_pred):
            if label == 0:
                dict_writer.writerow({'smiles': graph[SMILES_COLUMN]})

    e.log('processing examples...')
    examples_path = os.path.join(e.path, 'examples')
    os.mkdir(examples_path)

    generator = CounterfactualGenerator(
        model=model,
        processing=processing,
        neighborhood_func=get_neighborhood,
        distance_func=lambda org, mod: abs(org[0][0] - mod[0][0]),
        num_processes=16,
        batch_size=10_000,
    )

    graphs_example = random.sample(graphs, k=NUM_EXAMPLES)
    predictions_example = model.predict_graphs(graphs_example)
    c = 0
    for graph, (out, ni, ei) in zip(graphs_example, predictions_example):
        smiles = graph[SMILES_COLUMN]
        path = os.path.join(examples_path, f'{c:02d}')
        os.mkdir(path)

        label = np.argmax(out)
        # First of all we need to create the image
        data = processing.create(
            smiles,
            index='0',
            width=IMAGE_WIDTH,
            height=IMAGE_HEIGHT,
            output_path=path,
        )
        data = load_visual_graph_element(path, '0')
        graph.update(data['metadata']['graph'])
        node_positions = np.array(graph['node_positions'])

        # Then we want to plot the explanations
        fig, (ax_agg, ax_non) = plt.subplots(ncols=2, nrows=1, figsize=(20, 10))
        fig.suptitle(f'SMILES: {smiles}\n'
                     f'Predicted: {"Non-Aggregator" if label else "Aggregator"}')
        ax_agg.set_title('Aggregator Evidence')
        draw_image(ax_agg, data['image_path'])
        plot_node_importances_border(ax_agg, graph, node_positions, ni[:, 0])
        plot_edge_importances_border(ax_agg, graph, node_positions, ei[:, 0])

        ax_non.set_title('Non-Aggregator Evidence')
        draw_image(ax_non, data['image_path'])
        plot_node_importances_border(ax_non, graph, node_positions, ni[:, 1])
        plot_edge_importances_border(ax_non, graph, node_positions, ei[:, 1])

        importances_path = os.path.join(path, 'importances.pdf')
        fig.savefig(importances_path)

        # Now we also want to do the counterfactuals for this element
        for i in range(1, COUNTERFACTUAL_RANGE + 1):
            with tempfile.TemporaryDirectory() as temp_path:
                counterfactuals = generator.generate(
                    original=smiles,
                    path=temp_path,
                    k_results=5,
                    k_neighborhood=i,
                    image_width=IMAGE_WIDTH,
                    image_height=IMAGE_HEIGHT
                )
                counterfactuals_path = os.path.join(path, f'counterfactuals_{i}.pdf')
                create_counterfactual_pdf(
                    list(counterfactuals.values()),
                    output_path=counterfactuals_path,
                    original_element=data,
                )

        e.log(f' * {c} done')
        c += 1


experiment.run_if_main()
