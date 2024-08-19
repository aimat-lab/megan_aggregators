import os
import csv
import random
import pathlib
import typing as t

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from graph_attention_student.visualization import truncate_colormap
from scipy.special import softmax
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace
from visual_graph_datasets.util import dynamic_import
from visual_graph_datasets.processing.base import ProcessingBase
from visual_graph_datasets.processing.molecules import MoleculeProcessing
from visual_graph_datasets.visualization.importances import create_combined_importances_pdf
from graph_attention_student.torch.megan import Megan

from megan_aggregators.utils import load_processing
from megan_aggregators.utils import get_protonations
from megan_aggregators.utils import EXPERIMENTS_PATH
from megan_aggregators.torch import load_model

# == INPUT PARAMETERS ==
# These parameters define the input of the experiment. This mainly includes the elements for which the 
# actual predictions should be made.

# :param PROCESSING_PATH:
#       The path to the processing module that is to be used for the counterfactual generation. This
#       has to be an absolute string path to an existing processing module that is supposed to be used
#       for the processing of the molecules.
PROCESSING_PATH: str = os.path.join(EXPERIMENTS_PATH, 'assets', 'process.py')
# :param MODEL_PATH:
#       The path to the model that is to be used for the counterfactual generation. This has to be an
#       absolute string path to an existing checkpoint file that represents a stored model.
MODEL_PATH: str = os.path.join(EXPERIMENTS_PATH, 'results', 'vgd_torch_chunked_megan__aggregators_binary', 'debug', 'model.ckpt')
# :param ELEMENTS:
#       This list defines the elements for which the model predictions should be generated. THis is a list of 
#       dictionaries where each dictionary contains the information about one element. This dictionary may contain 
#       any additional information about the element but should at least contain a "smiles" property which defines 
#       the smile string representation of the corresponding molecule from which the graph representation can be 
#       constructed.
ELEMENTS: list[str] = [
    {'name': 'A', 'smiles': 'Cc2nccc3c1ccccc1[nH]c23'},
    {'name': 'B', 'smiles': 'Oc2nccc3c1ccccc1[nH]c23'},
    {'name': 'C', 'smiles': 'Oc1cc3c(cn1)[nH]c2ccccc23'},
    {'name': 'D', 'smiles': 'Sc1cc3c(cn1)[nH]c2ccccc23'},
    {'name': 'E', 'smiles': 'c1ccc3c(c1)[nH]c2cnccc23'},
    {'name': 'F', 'smiles': 'Cc2cccc3c1ccncc1[nH]c23'},
    {'name': 'G', 'smiles': 'Cc1cc3c(cn1)[nH]c2ccccc23'},
    {'name': 'H', 'smiles': 'Sc2cccc3c1ccncc1[nH]c23'},
]
# :param PROTONATE_ELEMENTS:
#       This parameter defines whether the elements should be protonated before they are processed. The protonation
#       is done with the DimorphiteDL tool and is calculated based on the SMILES representation of the input 
#       elements.
PROTONATE_ELEMENTS: bool = False
# :param PREDICTION_TEMPERATURE:
#       This parameter defines the temperature at which the predictions are made. This is used to scale the logits
#       of the model before they are transformed into probabilities. The higher the temperature, the more the
#       probabilities are smoothed out.
PREDICTION_TEMPERATURE: float = 10
# :param CHANNEL_COLORS_MAP:
#       This dictionary structure can be used to define the color maps for the various explanation channels.
#       For the "combined" visualization, all the explanation masks are drawn into the same image. These 
#       color maps define how to color each of the different explanation channels to differentiate them.
#       It is color maps and not flat colors because the color will be scaled with the explanation channel's 
#       fidelity value. The keys of this dict have to be integer indices of the channels in the order as they
#       appear in the model. The values are matplotlib color maps which will be used to color the explanation
#       masks in the visualization.
CHANNEL_COLORS_MAP: dict[int, mcolors.Colormap] = {
    1: mcolors.LinearSegmentedColormap.from_list('red', ['white', '#FF7B2F']),
    0: mcolors.LinearSegmentedColormap.from_list('green', ['white', '#3EFFAF']),
}
# :param CHANNEL_INFOS:
#       This dictionary structure can be used to define the information about the different explanation channels.
#       This information will be displayed in the legend of the visualization. The keys of this dict have to be
#       integer indices of the channels in the order as they appear in the model. The values are dictionaries
#       which contain the information about the channel. The only necessary information is the "name" of the
#       channel which will be displayed in the legend.
CHANNEL_INFOS: dict[str, t.Any] = {
    0: {
        'name': 'Non-Aggregator',
    },
    1: {
        'name': 'Aggregator',
    }
}
# :param IMPORTANCE_THRESHOLD:
#       This float value determines the threshold for the binarization of the importance values. This binarization 
#       is only applied for the "combined" visualization type. Every importance value above the threshold will be
#       visualized with the solid color and values below will not be shown at all.
IMPORTANCE_THRESHOLD: float = None


# == EXPERIMENT PARAMETERS ==

__DEBUG__ = True

experiment = Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)

@experiment.hook('load_model', default=False, replace=False)
def load_model_from_disk(e: Experiment) -> Megan:
    """
    This hook is responsible for loading the persistent version of the model from the disk and 
    returning the functional model object instance.
    
    This default implementation will load the default model that is stored as part of the package.
    
    :returns: Megan model instance
    """
    model = Megan.load(e.MODEL_PATH)
    return model


@experiment.hook('load_elements', default=True, replace=False)
def load_elements(e: Experiment) -> list[dict]:
    """
    This hook should load all the elements for which a prediction should be generated. 
    
    This default implementation will just return the list of elements that is given by the 
    global parameters "ELEMENTS".
    """
    e.log('loading ELEMENTS list...')
    return e.ELEMENTS


@experiment.hook('predict_element', default=False, replace=False)
def predict_element(e: Experiment,
                    element: dict,
                    model: Megan,
                    processing: ProcessingBase,
                    ) -> tuple[dict, dict]:
    """
    This hook is being called to generate the actual model prediction given a single element dict.
    The hook receives the element dict, the model instance and the processing instance as input and
    is supposed to return the graph representation of the element and the information dict that is
    generated by the model.
    """
    # The SMILES string representation of the molecule is one attribute which all of the elements
    # are supposed to have!
    smiles = element['smiles']
    
    # 16.04.24
    # Since the model is trained with protonated molecules, it makes sense to also apply this protonations
    # to the elements before the prediction is made to stay in-distribution w.r.t. to the training data.
    if e.PROTONATE_ELEMENTS:
        # The "get_protonations" function returns a list of protonated SMILES strings for the given input 
        # SMILES string. All of these are different possible protonation states of the molecule and in this 
        # case we simply select the first one.
        #smiles = random.choice(get_protonations(smiles))
        smiles = get_protonations(smiles)[0]
    
    graph = processing.process(smiles)
    # This method actually runs the forward inference for the given element and returns a dictionary containing 
    # the various outputs that are generated from this forward pass. This includes most importantly the actual 
    # prediction of the model but also the generated explanation masks and the graph embedding vector for example.
    info = model.forward_graphs([graph])[0]
    # dev: (num_classes, num_channels)
    dev = model.leave_one_out_deviations([graph])[0]
    info['graph_deviation'] = dev
    
    # For a classification problem we define the fidelity as the diagonal elements of the devaition matrix minus 
    # all the non-diagonal elements.
    # fid: (num_channels, )
    fid = np.diag(dev) - np.sum(dev - np.diag(np.diag(dev)), axis=0)
    graph['graph_fidelity'] = fid
    
    graph['graph_prediction'] = softmax(info['graph_output'])
    graph['node_importances'] = info['node_importance']
    graph['edge_importances'] = info['edge_importance']
    
    return graph, info


@experiment.hook('process_element', default=False, replace=False)
def process_element(e: Experiment, 
                    element: dict,
                    graph: dict,
                    info: dict,
                    ) -> dict:
    """
    This hook receives the element dict, the graph representation dict of that element and the info 
    dict that was returned by the model during the prediction of that element. The function should return
    a dictionary which should define all the columns to be put into the final CSV file for the results.
    """
    out = softmax(info['graph_output'])
    label = np.argmax(out)
    
    out_temp = softmax(info['graph_output'] / e.PREDICTION_TEMPERATURE)
    
    name = element['name'] if 'name' in element else ''
    e.log(f' * {name} - {element["smiles"]} - label: {label}')
    data = {
        'name': element['name'] if 'name' in element else '',
        'smiles': element['smiles'],
        'aggregator': round(out[1], 5),
        'non_aggregator': round(out[0], 5),
        'label': 'aggregator' if label == 1 else 'non-aggregator',
        'confidence': round(out[label], 5),
    }
    
    return data


@experiment.hook('evaluate', default=False, replace=False)
def evaluate(e: Experiment,
             elements: list[dict],
             graphs: list[dict],
             infos: list[dict],
             processing: t.Optional[ProcessingBase] = None
             ) -> None:
    """
    The evaluation method is called at the end of the experiment and is supposed to implement all 
    possible additional evaluations that should be done for the elements
    """
    
    # ~ confidence distribution
    # We want to plot a histogram of the confidence distribution of the model predictions.
    # This is because there have been problems with an overconfidence of the model and this 
    # distribution should give us a good overview of the confidence of the model.
    
    e.log('plotting confidence distribution...')
    # This is a list that contains the likelihood values of the model to predict the "non-aggregator"
    # class which are values between [0, 1]
    likelihoods: list[float] = [softmax(info['graph_output'])[1] for info in infos]
    fig, ax = plt.subplots(
        ncols=1,
        nrows=1,
        figsize=(8, 8),
    )
    ax.hist(
        likelihoods,
    )
    ax.set_xlabel('Likelihood')
    ax.set_ylabel('Number of Elements')
    ax.set_title('Likelihood Distribution')
    fig_path = os.path.join(e.path, 'confidence_distribution.pdf')
    fig.savefig(fig_path)
    

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
    
    elements = e.apply_hook('load_elements')
    
    e.log('starting to predict elements...')
    graphs = []
    infos = []
    rows = []
    for element in elements:
        
        graph, info = e.apply_hook(
            'predict_element',
            element=element,
            model=model,
            processing=processing,
        )
        graphs.append(graph)
        infos.append(info)
        
        row = e.apply_hook(
            'process_element',
            element=element,
            graph=graph,
            info=info,
        )
        rows.append(row)
        
    e.log('writing the results to csv file...')
    csv_path = os.path.join(e.path, 'results.csv')
    with open(csv_path, 'w') as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            
    # ~ evaluation
    # At the very end we add a hook that leaves the opportunity to do some additional evaluations
    
    e.apply_hook(
        'evaluate',
        elements=elements,
        graphs=graphs,
        infos=infos,
    )
            
    # ~ creating the visualizations
    # Ultimately we want to create the visualizations of the explanations for the invidual elements.
    # However, for that to work we first have to create the graph visualizations for those elements
    # So here we iterate over all the elements and use the processing instance to create those 
    # visualization images and then save them into a folder in the experiment archive folder.
    
    e.log('creating graph visualizations...')
    vis_folder = os.path.join(e.path, 'images')
    os.mkdir(vis_folder)
    
    # In this list we will store the absolute paths to all the images that we create here
    # in the same order as all the other lists that contain information about the elements.
    image_paths: list[str] = []
    for index, (element, graph) in enumerate(zip(elements, graphs)):
        smiles = element['smiles']
        fig, node_positions = processing.visualize_as_figure(
            value=smiles,
            width=1000,
            height=1000,
        )
        graph['node_positions'] = node_positions
        fig_path = os.path.join(vis_folder, f'{index:03d}.png')
        fig.savefig(fig_path)
        plt.close(fig)
        
        image_paths.append(fig_path)

    # ~ visualization explanations
    # In this section we want to plot the individual explanations for the individual elements.
    # There is already the predefined "create_combined_importances_pdf" function to do that 
    # we just have to feed it all the necessary information about the 
    
    labels: list[str] = []
    for element, graph in zip(elements, graphs):
        name = element['name'] if 'name' in element else ''
        out = np.argmax(graph['graph_prediction'])
        pred = 'aggregator' if out == 1 else 'non-aggregator'
        
        label = (
            f'{name} - {element["smiles"]}\n'
            f'{pred} ({graph["graph_prediction"][out] * 100:.2f}%)\n'
            f'fidelity: {graph["graph_fidelity"]}'
        )
        labels.append(label)

    e.log('visualizing explanations...')
    pdf_path = os.path.join(e.path, 'combined_importances.pdf')
    create_combined_importances_pdf(
        graph_list=graphs,
        image_path_list=image_paths,
        node_positions_list=[graph['node_positions'] for graph in graphs],
        node_importances_list=[graph['node_importances'] for graph in graphs],
        edge_importances_list=[graph['edge_importances'] for graph in graphs],
        graph_fidelity_list=[graph['graph_fidelity'] for graph in graphs],
        label_list=labels,
        channel_infos=e.CHANNEL_INFOS,
        channel_colors_map=e.CHANNEL_COLORS_MAP,
        importance_threshold=e.IMPORTANCE_THRESHOLD,
        output_path=pdf_path,
        logger=e.logger,
        base_fig_size=12,
        log_step=100,
    )
        
        
experiment.run_if_main()    