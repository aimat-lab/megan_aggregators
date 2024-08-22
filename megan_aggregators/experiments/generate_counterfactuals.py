"""
This experiment implements the generation of counterfactuals for a single base molecule given in 
SMILES format. This experiment in particular will generate counterfactuals for a range of neighborhood 
sizes up to a maximum neighborhood size. Neighborhood size in this context refers to the number of 
subsequent graph edits that are considered during the counterfactual search.

The experiment will create visualization PDFs for each neighborhood size where a fixed number of 
counterfactuals is visualized including the local attributional explanations coming from the 
MEGAN model.
"""
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
from visual_graph_datasets.util import dynamic_import
from visual_graph_datasets.processing.molecules import MoleculeProcessing
from visual_graph_datasets.visualization.base import draw_image
from visual_graph_datasets.visualization.importances import plot_edge_importances_background
from visual_graph_datasets.visualization.importances import plot_node_importances_background
from graph_attention_student.torch.megan import Megan
from vgd_counterfactuals.base import CounterfactualGenerator
from vgd_counterfactuals.generate.molecules import get_neighborhood

from megan_aggregators.utils import load_processing
from megan_aggregators.torch import load_model
from megan_aggregators.utils import EXPERIMENTS_PATH

mpl.use('TkAgg')
np.set_printoptions(precision=2)

# == EXPERIMENT PARAMETERS ==
# These parameters define the behavior of the counterfactual generation. This includes for 
# example, most importantly, the SMILES string of the molecule for which the counterfactuals
# are to be generated.

# :param PROCESSING_PATH:
#       The path to the processing module that is to be used for the counterfactual generation. This
#       has to be an absolute string path to an existing processing module that is supposed to be used
#       for the processing of the molecules.
PROCESSING_PATH: str = os.path.join(EXPERIMENTS_PATH, 'assets', 'process.py')
# :param MODEL_PATH:
#       The path to the model that is to be used for the counterfactual generation. This has to be an
#       absolute string path to an existing checkpoint file that represents a stored model.
MODEL_PATH: str = os.path.join(EXPERIMENTS_PATH, 'results', 'vgd_torch_chunked_megan__aggregators_binary', 'debug', 'model.ckpt')
# :param SMILES:
#       The SMILES string of the molecule for which the counterfactuals are to be generated.
SMILES = 'CCC(=O)CC1=CC=CC=C1'
# :param NUM_COUNTERFACTUALS:
#       The number of counterfactuals to be generated for the given molecule.
NUM_COUNTERFACTUALS: int = 10
# :param NEIGHBORHOOD_RANGE:
#       The range of the neighborhood to be used for the counterfactual generation. This parameter
#       determines the maximum number of graph edits to be considered for the counterfactual search.
#       If this parameter is 1, for example, all the graphs which are one graph edit away from the 
#       original are considered during the counterfactual search. For values higher than 1, all of 
#       the neighbors are iteratively expanded by single graph edits as well.
NEIGHBORHOOD_RANGE: int = 2

# == VISUALIZATION PARAMETERS ==
# :param IMAGE_WIDTH:
#       The width of the images to be generated for the visualizations of the counterfactuals.
IMAGE_WIDTH = 1000
# :param IMAGE_HEIGHT:
#       The height of the images to be generated for the visualizations of the counterfactuals.
IMAGE_HEIGHT = 1000
# :param FIG_SIZE:
#       The size of the figures to be generated for the visualizations of the counterfactuals.
FIG_SIZE = 6
# :param CLASS_NAMES:
#       This dictionariy defines the integer indices of the classes as the keys and the corresponding 
#       values are the human readable names of the classes so that they can be properly labeled in the 
#       visualizations.
CLASS_NAMES = {
    0: 'non-aggregator',
    1: 'aggregator',
}


def DISTANCE_FUNC(org, mod):
    label = np.argmax(org)
    return org[label] - mod[label]


__DEBUG__ = True

experiment = Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)

@experiment.hook('load_model', default=False, replace=True)
def load_model_from_disk(e: Experiment) -> Megan:
    """
    This hook is responsible for loading the persistent version of the model from the disk and 
    returning the functional model object instance.
    
    This default implementation will load the default model that is stored as part of the package.
    
    :returns: Megan model instance
    """
    model = Megan.load(e.MODEL_PATH)
    return model


@experiment
def experiment(e: Experiment):

    # ~ loading the necessary objects
    # to generate the counterfactuals we need to load a model for which the counterfactuals should be generated
    
    e.log(f'starting to load model...')
    # :hook load_model:
    #       This hooks is supposed to implement the loading of the model from it's persistent form. The return 
    #       value of this hook is supposed to be the loaded "Megan" model object.
    model: Megan = e.apply_hook('load_model')
    e.log(f'loaded model of class "{model.__class__.__name__}"')
    
    e.log(f'starting to load processing...')
    module = dynamic_import(e.PROCESSING_PATH)
    processing: MoleculeProcessing = module.processing
    e.log(f'loaded processing of class {processing.__class__.__name__}')

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
            predict_func=lambda model, graphs: model.predict_graphs(graphs).tolist(),
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
            e.log(f'   generated {len(counterfactuals)} counterfactuals')
            datas = [data_original, *counterfactuals.values()]
            graphs = [data['metadata']['graph'] for data in datas]
            
            e.log(f'  running model forward pass...')
            for graph in graphs:
                if 'node_importances' in graph:
                    del graph['node_importances']
                if 'edge_importances' in graph:
                    del graph['edge_importances']
                    
            infos = model.forward_graphs(graphs)
            devs = model.leave_one_out_deviations(graphs)
            for graph, info, dev in zip(graphs, infos, devs):
                graph['graph_prediction'] = info['graph_output']
                graph['node_importances'] = info['node_importance']
                graph['edge_importances'] = info['edge_importance']
                
                graph['graph_deviation'] = dev

            e.log(f'  generating the visualizations...')
            pdf_path = os.path.join(e.path, f'counterfactuals_{i}_edit_neighborhood.pdf')
            with PdfPages(pdf_path) as pdf:

                for c, data in enumerate(datas):
                    graph = data['metadata']['graph']
                    node_positions = np.array(graph['node_positions'])
                    
                    pred = graph['graph_prediction']
                    ni = graph['node_importances']
                    ei = graph['edge_importances']
                    dev = graph['graph_deviation']

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
                                     f'leave-one-out fidelity: {dev[0][0]:.2f}')
                    ax_agg.set_xticks([])
                    ax_agg.set_yticks([])
                    draw_image(ax_agg, data['image_path'])
                    plot_node_importances_background(ax_agg, graph, node_positions, ni[:, 0])
                    plot_edge_importances_background(ax_agg, graph, node_positions, ei[:, 0])

                    ax_non.set_title(f'class 1 - non-aggregator evidence\n'
                                     f'leave-one-out fidelity: {dev[1][1]:.2f}')
                    ax_non.set_xticks([])
                    ax_non.set_yticks([])
                    draw_image(ax_non, data['image_path'])
                    plot_node_importances_background(ax_non, graph, node_positions, ni[:, 1])
                    plot_edge_importances_background(ax_non, graph, node_positions, ei[:, 1])

                    pdf.savefig(fig)
                    plt.close(fig)


experiment.run_if_main()
