"""
Module containing various utility methods.
"""
import os
import random
import shutil
import pathlib
import logging
import tempfile
import subprocess
import tempfile
from typing import List
import typing as t

import click
import torch
import jinja2 as j2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as ks
import seaborn as sns
from weasyprint import HTML, CSS
from dimorphite_dl import DimorphiteDL
from torch.utils.data import Dataset
from scipy.special import softmax
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from visual_graph_datasets.util import dynamic_import
from visual_graph_datasets.data import load_visual_graph_element
from visual_graph_datasets.processing.molecules import MoleculeProcessing
from visual_graph_datasets.visualization.base import draw_image
from visual_graph_datasets.visualization.importances import plot_node_importances_background
from visual_graph_datasets.visualization.importances import plot_edge_importances_background
from vgd_counterfactuals.base import CounterfactualGenerator
from vgd_counterfactuals.generate.molecules import get_neighborhood

from graph_attention_student.keras import CUSTOM_OBJECTS
from graph_attention_student.training import EpochCounterCallback
from graph_attention_student.models import load_model as load_model_raw
from graph_attention_student.torch.megan import Megan
from graph_attention_student.torch.data import data_from_graph


PATH = pathlib.Path(__file__).parent.absolute()
VERSION_PATH = os.path.join(PATH, 'VERSION')
EXPERIMENTS_PATH = os.path.join(PATH, 'experiments')
TEMPLATES_PATH = os.path.join(PATH, 'templates')
ASSETS_PATH = os.path.join(PATH, 'assets')

# 18.04.23 - Decided that it would make sense to ship one version of the fully trained model instead of
# always having to train one.
MODEL_PATH = os.path.join(PATH, 'model')
# 15.05.23 - If we save the model into the repository then we also need to save the processing file as well
# for it to make sense. We will also place that into the model folder.
PROCESSING_PATH = os.path.join(ASSETS_PATH, 'process.py')

# Use this jinja2 environment to conveniently load the jinja templates which are defined as files within the
# "templates" folder of the package!
TEMPLATE_ENV = j2.Environment(
    loader=j2.FileSystemLoader(TEMPLATES_PATH),
    autoescape=j2.select_autoescape(),
)
TEMPLATE_ENV.globals.update(**{
    'zip': zip,
    'enumerate': enumerate
})

# This logger can be conveniently used as the default argument for any function which optionally accepts
# a logger. This logger will simply delete all the messages passed to it.
NULL_LOGGER = logging.Logger('NULL')
NULL_LOGGER.addHandler(logging.NullHandler())


# == CLI RELATED ==

def get_version():
    """
    Returns the version of the software, as dictated by the "VERSION" file of the package.
    """
    with open(VERSION_PATH) as file:
        content = file.read()
        return content.replace(' ', '').replace('\n', '')


# https://click.palletsprojects.com/en/8.1.x/api/#click.ParamType
class CsvString(click.ParamType):

    name = 'csv_string'

    def convert(self, value, param, ctx) -> List[str]:
        if isinstance(value, list):
            return value

        else:
            return value.split(',')


# == LATEX RELATED ==
# These functions are meant to provide a starting point for custom latex rendering. That is rendering latex
# from python strings, which were (most likely) dynamically generated based on some kind of experiment data

def render_latex(kwargs: dict,
                 output_path: str,
                 template_name: str = 'article.tex.j2'
                 ) -> None:
    """
    Renders a latex template into a PDF file. The latex template to be rendered must be a valid jinja2
    template file within the "templates" folder of the package and is identified by the string file name
    `template_name`. The argument `kwargs` is a dictionary which will be passed to that template during the
    rendering process. The designated output path of the PDF is to be given as the string absolute path
    `output_path`.

    **Example**

    The default template for this function is "article.tex.j2" which defines all the necessary boilerplate
    for an article class document. It accepts only the "content" kwargs element which is a string that is
    used as the body of the latex document.

    .. code-block:: python

        import os
        output_path = os.path.join(os.getcwd(), "out.pdf")
        kwargs = {"content": "$\text{I am a math string! } \pi = 3.141$"
        render_latex(kwargs, output_path)

    :raises ChildProcessError: if there was ANY problem with the "pdflatex" command which is used in the
        background to actually render the latex

    :param kwargs:
    :param output_path:
    :param template_name:
    :return:
    """
    with tempfile.TemporaryDirectory() as temp_path:
        # First of all we need to create the latex file on which we can then later invoke "pdflatex"
        template = TEMPLATE_ENV.get_template(template_name)
        latex_string = template.render(**kwargs)
        latex_file_path = os.path.join(temp_path, 'main.tex')
        with open(latex_file_path, mode='w') as file:
            file.write(latex_string)

        # Now we invoke the system "pdflatex" command
        command = (f'pdflatex  '
                   f'-interaction=nonstopmode '
                   f'-output-format=pdf '
                   f'-output-directory={temp_path} '
                   f'{latex_file_path} ')
        proc = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            raise ChildProcessError(f'pdflatex command failed! Maybe pdflatex is not properly installed on '
                                    f'the system? Error: {proc.stdout.decode()}')

        # Now finally we copy the pdf file - currently in the temp folder - to the final destination
        pdf_file_path = os.path.join(temp_path, 'main.pdf')
        shutil.copy(pdf_file_path, output_path)


def plot_roc_curve(ax: plt.Axes,
                   y_true: np.ndarray,
                   y_pred: np.ndarray,
                   pos_label: t.Optional[int] = None,
                   label: t.Optional[str] = None,
                   show_label_auc: bool = True,
                   color: str = 'orange',
                   show_reference: bool = True,
                   reference_color: str = 'lightgray'
                   ) -> t.Tuple[np.ndarray, np.ndarray]:
    """
    Plots an ROC curve onto the given Axes ``ax`` based on the true binary labels ``y_true`` and the
    predicted labels ``y_pred``. Will also return the arrays for the false positive ratio and the true
    positive ratio, on which the ROC curve is based.

    :param ax: The matplotlib.Axes object on which to draw the ROC curve.
    :param y_true: A np array of the binary true labels
    :param y_pred: A np array of the corresponding predicted labels
    :param pos_label: Optional
    :param label: An optional string which will be displayed as a description of the ROC curve in the
        legend of the plot.
    :param show_label_auc: Whether to actually add labels to the plot
    :param color: A matplotlib color value for the main ROC curve
    :param show_reference: Boolean flag whether to display the reference curve. The reference curve will
        simply be a straight line between bottom left and top right corner showing the chance level of a
        completely random predictor (AUC 0.5)
    :param reference_color: A mpl color value for the reference curve

    :returns: A tuple of two elements: (1) The numpy array of the false positive rate and (2) the numpy
        array of the true positive rate
    """
    fpr, tpr, _ = roc_curve(
        y_true=y_true,
        y_score=y_pred,
        pos_label=pos_label,
    )
    auc = roc_auc_score(y_true, y_pred)

    if label is None:
        label = ''

    if show_label_auc:
        label = f'{label} (AUC = {auc:.2f})'

    ax.plot(
        fpr, tpr,
        color=color,
        label=label,
        ls='-',
    )

    ax.set_xlim(-0.05, 1.05)
    ax.set_xlabel('False Positive Rate')

    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel('True Positive Rate')

    if show_reference:
        reference_label = 'chance level'
        if show_label_auc:
            reference_label = f'{reference_label} (AUC = 0.5)'

        ax.plot(
            [0, 1], [0, 1],
            color=reference_color,
            ls='-',
            label=reference_label,
            zorder=-1,
        )

    return fpr, tpr


def plot_confusion_matrix(ax: plt.Axes,
                          labels_true: np.ndarray,
                          labels_pred: np.ndarray,
                          target_names: list[str],
                          cmap: str = 'viridis'
                          ):
    """
    Plots a confusion matrix basedo n the given ``labels_true`` and ``labels_pred`` arrays
    onto the given matplolib Axes ``ax``. The target names are used to label the axes of the
    confusion matrix.
    
    :param ax: The matplotlib Axes object on which to draw the confusion matrix
    :param labels_true: The true integer labels
    :param labels_pred: The predicted integer labels
    :param target_names: A list of strings which are the class names for the labels.
    
    :returns: The confusion matrix as a numpy array
    """
    # This will create the confusion matrix which is needed for the visualization
    cm = confusion_matrix(labels_true, labels_pred)

    # seaborn creates a nice heatmap of the confusion matrix
    ticklabels = target_names
    sns.heatmap(
        cm, 
        ax=ax, 
        annot=True, 
        fmt='02d',
        cmap=cmap,
        xticklabels=ticklabels,
        yticklabels=ticklabels,
        linewidths=0,
    )
    
    return cm
        
    


def load_processing(processing_path: str = PROCESSING_PATH):
    """
    Loads the Processing object which can be used to turn the domain specific representations of input
    elements (SMILES strings) into the Graph dicts which are needed to make predictions with the machine
    learning model.

    :param processing_path: The path to the "process.py" file which has been used for the visual graph
        dataset with which the model was trained.

    :returns: Processing instance
    """
    module = dynamic_import(processing_path)
    return module.processing


def visualize_explanations(smiles: str,
                           processing: MoleculeProcessing,
                           node_importances: np.ndarray,
                           edge_importances: np.ndarray,
                           ) -> plt.Figure:
    """
    Creates a visualization of the explanations given as the ``node_importances`` and ``edge_importances`` array 
    on top of the molecule visualization given by the ``smiles`` string representation. 
    
    This method will return a mpl Figure instance which consists of as many separate plots as there are explanation 
    channels in the given explanation arrays. Each plot will visualize a different explanation on top of the same 
    molecule.
    
    :param smiles: The string SMILES representation
    :param processing: The Processing instance
    :param node_importances: A numpy array containing the node attention values of the shape
        (number of nodes, number of explanations)
    :param edge_importances: A numpy array containing the edge attention values of the shape
        (number of edges, number of explanations)
    
    :returns: A mpl Figure instance
    """
    # We can extract the number of explanation channels easily from the shapes of the importance arrays
    num_channels = node_importances.shape[1]
    
    with tempfile.TemporaryDirectory() as path:
        
        name = '0'
        processing.create(
            smiles,
            index=name,
            output_path=path
        )
        
        # After the element has been created as a set of persistent files on the disk we need to load 
        # the relevant information from the disk
        data = load_visual_graph_element(path, name)
        graph = data['metadata']['graph']
        
        fig, rows = plt.subplots(
            ncols=num_channels,
            nrows=1,
            figsize=(num_channels * 10, 10),
            squeeze=False,
        )
        for index in range(num_channels):
            ax = rows[0][index]
            ax.set_title(f'channel {index}')
            
            # At first we need to put the visualization of the actual molecule onto the plot
            draw_image(ax, data['image_path'])
            
            # Then we can draw the explanations on top of that
            plot_node_importances_background(
                ax=ax,
                g=graph,
                node_positions=graph['node_positions'],
                node_importances=node_importances[:, index]
            )
            plot_edge_importances_background(
                ax=ax,
                g=graph,
                node_positions=graph['node_positions'],
                edge_importances=edge_importances[:, index]
            )
            
    return fig
            

def generate_counterfactuals_with_model(model: Megan,
                                        smiles: str,
                                        num: int = 15,
                                        k_neighborhood: int = 1,
                                        processing: MoleculeProcessing = load_processing(),
                                        fix_protonation: bool = False,
                                        min_ph: float = 6.4,
                                        max_ph: float = 6.4,
                                        ) -> List[tuple[str, np.ndarray, float]]:
    """
    Given a loaded ``model`` and a SMILES string ``smiles`` this function will generate a number of
    counterfactuals for the given input molecule. The number of counterfactuals to be generated is given by 
    the ``num`` argument. The ``k_neighborhood`` argument will determine the size of the neighborhood which 
    will be used to generate the counterfactuals. The ``processing`` argument is the instance of the
    MoleculeProcessing class which is used to convert the SMILES strings into the graph representation which
    the model can understand. The ``fix_protonation`` flag will determine whether the protonation state of
    the molecule should be fixed during the counterfactual generation. The ``min_ph`` and ``max_ph`` arguments
    will determine the pH range which will be used to for the protonation step.
    
    This function will return a list of tuples where each tuple consists of the SMILES string of the 
    generated counterfactual and the corresponding model prediction output array.
    
    :returns: List of tuples. Each tuple consists of the SMILES string and the model prediction output array.
    """
    
    # Since we are dealing with a classification problem here, we need to adjust the distance function
    # to return the difference in the probability of the original and the modified molecule being of the
    # same class.
    def distance_func(org, mod):
        label = np.argmax(org)
        # An important thing to consider here is that the model forward pass itself only returns the 
        # classification logits, which means that we will need to apply the softmax operation manually 
        # to get the probabilities.
        return softmax(org)[label] - softmax(mod)[label]
    
    # We need to customize the predict function to use the methods available for the torch version of the 
    # Megan model.
    def predict_func(model, graphs):
        infos: list[dict] = model.forward_graphs(graphs)
        return [info['graph_output'] for info in infos]
    
    generator = CounterfactualGenerator(
        model=model,
        processing=processing,
        distance_func=distance_func,
        predict_func=predict_func,
        neighborhood_func=lambda *args, **kwargs: get_neighborhood(
            *args, 
            **kwargs, 
            fix_protonation=fix_protonation, 
            min_ph=min_ph, 
            max_ph=max_ph,
        ),
    )
    
    # The counterfactual generator will actually directly create a "mini" visual graph dataset from all of the 
    # counterfactuals which will have to be saved to a folder on the disk. Therefore we create a temporary
    # directory for that purpose.
    with tempfile.TemporaryDirectory() as temp_path:
        index_data_map = generator.generate(
            original=smiles,
            path=temp_path,
            k_results=num,
            k_neighborhood=k_neighborhood,
        )
        # The generator will create the full graph representation, but for this wrapper we only want to return 
        # a list of elements that consists of the SMILES strings and the model predictions for those elements.
        results = [
            (
                data['metadata']['smiles'], 
                softmax(data['metadata']['prediction']), 
                data['metadata']['distance']
            ) 
            for data in index_data_map.values()
        ]
        results.sort(key=lambda tupl: tupl[2], reverse=True)
        return results


def get_protonations(smiles: str, 
                     min_ph: float = 6.4, 
                     max_ph: float = 6.4,
                     max_variants: int = 100,
                     ) -> List[str]:
    """
    Given a SMILES string ``smiles`` this function will return a list of SMILES strings which represent the
    different protonation states of the molecule at the given pH range. The ``min_ph`` and ``max_ph`` arguments
    will determine the pH range which will be used to generate the protonation states.

    :param smiles: The SMILES string of the molecule
    :param min_ph: The minimum pH value of the pH range
    :param max_ph: The maximum pH value of the pH range

    :returns: List of SMILES strings representing the different protonation states of the molecule
    """
    dmph = DimorphiteDL(
        min_ph=min_ph,
        max_ph=max_ph,
        pka_precision=0.1,
        label_states=False,
        max_variants=max_variants,
    )
    return dmph.protonate(smiles)


class VariableSchedulerCallback(EpochCounterCallback):

    def __init__(self,
                 property_name: str,
                 value_start: float,
                 value_end: float,
                 epoch_end: int,
                 epoch_start: int = 0):
        super(VariableSchedulerCallback, self).__init__()
        self.property_name = property_name
        self.epoch_start = epoch_start
        self.epoch_end = epoch_end
        self.value_start = value_start
        self.value_end = value_end

        self.step = (self.value_end - self.value_start) / (self.epoch_end - self.epoch_start)
        self.value = self.value_start

    def on_epoch_end(self, *args, **kwargs):
        variable = getattr(self.model, self.property_name)
        if self.epoch > self.epoch_start:
            self.value = self.value_start + (self.epoch / self.epoch_end) * self.step
            variable.assign(self.value)


class ChunkedDataset(Dataset):
    """
    Implements a PyTorch Dataset which is able to load a pre-chunked dataset. These chunked datasets 
    consist of multiple .PT files which actually contain the already processed PyG Data instances.
    These files can simply be loaded to directly access the data instances for the training.
    
    The constructor of this class accepts a folder path as an argument and will load all the .PT files 
    in that folder as individual chunks of the dataset.
    
    While iterating over a ChunkedDataset such as this, only one chunk at a time is actually loaded into 
    the memory, the datasets data generator function will yield all the Data instances contained in that 
    chunk until all of them are exhausted. Only then the current chunk will be unloaded and the next chunk 
    will be loaded.
    
    NOTE: Each chunk will be considered it's own dataset! When training a model, this means that one epoch 
          will only iterate the elements of a single chunk and not all the elements of the dataset. 
          This has been done because the implementation is much easier and from a user perspective, the 
          only necessary step to accomodate this is to multiple the number of the epochs by the number of
          chunks in the dataset.
    """
    
    def __init__(self, 
                 path: str,
                 ) -> None:
        # This is the thingy here
        self.path = path
        
        files = os.listdir(path)
        
        # In this list we will store the file paths to the actual PT files that represent the 
        # dataset chunks.
        self.file_paths: t.List[str] = []
        # We'll simply consider every file in the given folder path that has the PT file ending
        for file in files:
            if file.endswith('.pt'):
                self.file_paths.append(os.path.join(path, file))
                
        # The most common naming scheme for these kinds of chunk paths is to have the chunk index 
        # as the last part of the file name. We can use this to sort the file paths in the correct
        # order according to the numerical order of these chunk indices.
        self.file_paths.sort()
        self.num_files = len(self.file_paths)
        
        self.current_index = 0
        self.current_path = self.file_paths[self.current_index]
        
        # With this integer we will track how many items of the current chunk have been consumed already and 
        # once all the items (==chunk length) have been consumed we will use that as a signal to load the next
        # chunk.
        self.counter = 0
        
        # We will use this list to actually store the torch geometric Data instances that are loaded 
        # from the dataset
        self.data: t.List[t.Any] = []
        
        # To get started we load the first chunk here.
        self.load_chunk(self.current_index)
        
    def load_chunk(self, index: int) -> None:
        """
        Unloads the current chunk and instead loads the chunk with the given ``index``. After 
        calling this method, the internal ``self.data`` attribute will be populated as a list 
        of the Data instances of the new chunk.
        """
        print(f'loading chunk {index}...')
        self.current_path = self.file_paths[index]
        self.data = torch.load(self.current_path)
        
    def next_index(self):
        """
        Increments the ``self.current_index`` integer index of the chunk
        
        It is recommended to use this method instead of manually incrementing the index, because 
        this method will consider a index overflow and will cyclically reset the index to 0 if the
        end of the list of files is reached.
        """
        if self.current_index == self.num_files - 1:
            self.current_index = 0
        else:
            self.current_index += 1
        
    def __len__(self):
        return len(self.data)
        #return 10_000
        
    def __getitem__(self, idx: int):
        
        if self.counter >= len(self.data):
            self.next_index()
            self.load_chunk(self.current_index)
            self.counter = 0
            
        self.counter += 1
        return self.data[idx]
    
    
class GraphDataset(Dataset):
    
    def __init__(self, index_data_map: dict):
        self.index_data_map = index_data_map
        self.keys = list(index_data_map.keys())
    
    def __len__(self):
        return len(self.index_data_map)
        
    def __getitem__(self, idx: int):
        key = self.keys[idx]
        return data_from_graph(self.index_data_map[key]['metadata']['graph'])

    
class MultiChunkedDataset(Dataset):
    
    def __init__(self, 
                 path: str,
                 num_chunks: int = 1,
                 num_elements: int = 1000,
                 ) -> None:
        
        # This is the thingy here
        self.path = path
        self.num_chunks = num_chunks
        
        files = os.listdir(path)
        
        # In this list we will store the file paths to the actual PT files that represent the 
        # dataset chunks.
        self.file_paths: t.List[str] = []
        # We'll simply consider every file in the given folder path that has the PT file ending
        for file in files:
            if file.endswith('.pt'):
                self.file_paths.append(os.path.join(path, file))
                
        self.indices = list(range(len(self.file_paths)))
        
        # We will use this list to actually store the torch geometric Data instances that are loaded 
        # from the dataset
        self.data: t.List[t.Any] = []
        
        self.current_indices = []
        self.counter = 0
        
        # To get started we load the first chunk here.
        self.sample_chunks()
        
    def sample_chunks(self) -> None:
        """
        Unloads the current chunk and instead loads the chunk with the given ``index``. After 
        calling this method, the internal ``self.data`` attribute will be populated as a list 
        of the Data instances of the new chunk.
        """
        self.current_indices = random.sample(self.indices, self.num_chunks)
        
        print(f'loading chunks {self.current_indices}...')
        
        for index in self.current_indices:
            path = self.file_paths[index]
            self.data += torch.load(path)
        
    def __len__(self):
        #return 50_000
        return int(len(self.data) * 0.98)
        
    def __getitem__(self, idx: int):
        
        if self.counter >= len(self):
            #self.sample_chunks()
            self.counter = 0
            
        self.counter += 1
        return self.data[idx]

    
    
def create_report(pages: list[dict],
                  path: str,
                  template_name: str = 'report.html.j2'
                  ) -> None:
    """
    Generates a PDF report file at the given ``path`` from the given list of ``pages``.
    
    :param pages: A list of dict objects where each dict object represents one page of the report. Each page 
        should have the following keys: "title", "images", "texts". The "title" key should contain a string
        which will be used as the title of the page. The "images" key should contain a list of image file paths
        or matplotlib figure instances which will be displayed on the page. The "texts" key should contain a list
        of strings which will be displayed as text on the page.
    :param path: The absolute path to the PDF file which will be created. 
    
    :returns:
    """
    template: j2.Template = TEMPLATE_ENV.get_template(template_name)
        
    with tempfile.TemporaryDirectory() as temp_path:
        
        for page_index, page in enumerate(pages):
        
            # ~ possibly converting images into file
            # The images can be supplied in different formats but in the end we need to 
            # convert them such that they are unique image files in the temporary directory.
            image_paths: list[str] = []
            for image_index, image in enumerate(page['images']):
                
                image_name = f'{page_index}_{image_index}.png'
                image_path = os.path.join(temp_path, image_name)
                
                # If the image is a string then we assume it is the absolute path to an already 
                # existing image file and in this case we have to copy that file to the temporary
                # directory.
                if isinstance(image, str):
                    # We need to copy the file to the temporary directory
                    shutil.copy(image, image_path)
                    
                # Alternatively we can also pass the image as a matplotlib figure instance. In this case
                # we will save the figure to a file in the temporary directory.
                elif isinstance(image, plt.Figure):
                    fig.savefig(image_path)
                    
                image_paths.append(image_path)
                
                page['_images'] = page['images']
                page['images'] = image_paths
            
        # Finally we can render the template.
        context = {
            'pages': pages,
        }
        
        html_content: str = template.render(
            **context,
        )
        
        html = HTML(string=html_content)
        css = CSS(os.path.join(TEMPLATES_PATH, 'report.css'))
        html.write_pdf(path, stylesheets=[css])
    