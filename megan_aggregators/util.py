"""
Module containing various utility methods.
"""
import os
import shutil
import pathlib
import logging
import tempfile
import subprocess
from typing import List
import typing as t

import click
import jinja2 as j2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as ks
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from graph_attention_student.keras import CUSTOM_OBJECTS


PATH = pathlib.Path(__file__).parent.absolute()
VERSION_PATH = os.path.join(PATH, 'VERSION')
EXPERIMENTS_PATH = os.path.join(PATH, 'experiments')
TEMPLATES_PATH = os.path.join(PATH, 'templates')

# 18.04.2023 - Decided that it would make sense to ship one version of the fully trained model instead of
# always having to train one.
MODEL_PATH = os.path.join(PATH, 'model')

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


def load_model(model_path: str = MODEL_PATH) -> ks.models.Model:
    """
    Loads the keras model from memory given its ``model_path`` absolute folder path. By default, this
    function will load the default model which is shipped with this package.

    :param model_path: The absolute path to the folder which contains the models persistent representation.

    :returns: The MEGAN model which is saved at the given folder path
    """
    with ks.utils.custom_object_scope(CUSTOM_OBJECTS):
        model = ks.models.load_model(model_path)

    return model
