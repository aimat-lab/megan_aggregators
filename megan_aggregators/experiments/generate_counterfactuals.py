import os
import pathlib
import tempfile

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import visual_graph_datasets.typing as tv
from matplotlib.backends.backend_pdf import PdfPages
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace
from visual_graph_datasets.data import load_visual_graph_element
from visual_graph_datasets.processing.molecules import MoleculeProcessing
from visual_graph_datasets.visualization.base import draw_image
from visual_graph_datasets.visualization.importances import plot_edge_importances_border
from visual_graph_datasets.visualization.importances import plot_node_importances_border
from graph_attention_student.models.megan import Megan
from vgd_counterfactuals.base import CounterfactualGenerator
from vgd_counterfactuals.generate.molecules import get_neighborhood

from megan_aggregators.utils import load_model, load_processing
mpl.use('TkAgg')
np.set_printoptions(precision=2)

SMILES = 'CCC(=O)CC1=CC=CC=C1'

NUM_COUNTERFACTUALS: int = 10
NEIGHBORHOOD_RANGE: int = 3
IMAGE_WIDTH = 1000
IMAGE_HEIGHT = 1000
FIG_SIZE = 6
CLASS_NAMES = {
    0: 'aggregator',
    1: 'non-aggregator',
}


def DISTANCE_FUNC(org, mod):
    label = np.argmax(org[0])
    return org[0][label] - mod[0][label]


__DEBUG__ = True


@Experiment(base_path=folder_path(__file__),
            namespace=file_namespace(__file__),
            glob=globals())
def experiment(e: Experiment):

    e.log('loading model...')
    model: Megan = load_model()
    processing: MoleculeProcessing = load_processing()

    with tempfile.TemporaryDirectory() as temp_path:
        e.log('generating the original element...')
        processing.create(
            e.SMILES,
            index='0',
            width=e.IMAGE_WIDTH,
            height=e.IMAGE_HEIGHT,
            output_path=e.path,
        )
        data_original = load_visual_graph_element(e.path, '0')
        graph_original = data_original['metadata']['graph']
        data_original['metadata']['prediction'] = model.predict_graph(graph_original)
        data_original['metadata']['distance'] = 0.0

        generator = CounterfactualGenerator(
            model=model,
            processing=processing,
            neighborhood_func=get_neighborhood,
            distance_func=e.DISTANCE_FUNC,
            logger=e.logger,
            num_processes=16,
            batch_size=10_000,
        )

        e.log('generating the counterfactuals...')
        for i in range(1, e.NEIGHBORHOOD_RANGE + 1):
            e.log(f' * neighborhood size: {i}')
            counterfactuals = generator.generate(
                e.SMILES,
                path=temp_path,
                k_results=e.NUM_COUNTERFACTUALS,
                k_neighborhood=i,
                image_width=e.IMAGE_WIDTH,
                image_height=e.IMAGE_HEIGHT,
            )
            data_list = [data_original, *counterfactuals.values()]
            graph_list = [data['metadata']['graph'] for data in data_list]
            leave_one_out = model.leave_one_out(graph_list, channel_funcs=[
                lambda org, mod: org[0] - mod[0],
                lambda org, mod: org[1] - mod[1],
            ])

            pdf_path = os.path.join(e.path, f'counterfactuals_{i}_edit_neighborhood.pdf')
            with PdfPages(pdf_path) as pdf:

                for c, data in enumerate(data_list):
                    graph = data['metadata']['graph']
                    node_positions = np.array(graph['node_positions'])
                    pred, ni, ei = [np.array(v) for v in data['metadata']['prediction']]
                    label = np.argmax(pred)

                    fig, (ax_im, ax_agg, ax_non) = plt.subplots(
                        ncols=3,
                        nrows=1,
                        figsize=(3 * e.FIG_SIZE, 1 * e.FIG_SIZE),
                    )
                    fig.suptitle(f'SMILES: {data["metadata"]["smiles"]}\n'
                                 f'prediction: {pred}  '
                                 f'(label: {label} - {e.CLASS_NAMES[label]})')

                    if c != 0:
                        fig.set_facecolor('mistyrose')

                    ax_im.set_title('original molecule')
                    ax_im.set_xticks([])
                    ax_im.set_yticks([])
                    draw_image(ax_im, data['image_path'])

                    ax_agg.set_title(f'class 0 - aggregator evidence\n'
                                     f'leave-one-out fidelity: {leave_one_out[c][0]:.2f}')
                    ax_agg.set_xticks([])
                    ax_agg.set_yticks([])
                    draw_image(ax_agg, data['image_path'])
                    plot_node_importances_border(ax_agg, graph, node_positions, ni[:, 0])
                    plot_edge_importances_border(ax_agg, graph, node_positions, ei[:, 0])

                    ax_non.set_title(f'class 1 - non-aggregator evidence\n'
                                     f'leave-one-out fidelity: {leave_one_out[c][1]:.2f}')
                    ax_non.set_xticks([])
                    ax_non.set_yticks([])
                    draw_image(ax_non, data['image_path'])
                    plot_node_importances_border(ax_non, graph, node_positions, ni[:, 1])
                    plot_edge_importances_border(ax_non, graph, node_positions, ei[:, 1])

                    pdf.savefig(fig)
                    plt.close(fig)


experiment.run_if_main()
