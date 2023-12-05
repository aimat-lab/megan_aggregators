"""
This experiment performs the chunked (memory efficient) training of the MEGAN model for the 
aggregators_binary dataset.
"""
import os
import pathlib
import typing as t

import tensorflow as tf
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

PATH = pathlib.Path(__file__).parent.absolute()


# == DATASET PARAMETERS ==
# These are the parameters related to the dataset and the processing thereof.

# :param CACHED_PATH:
#       This should define the path to the folder which contains the disk cached version of the dataset. This disk 
#       cache folder has to contain itself 2 sub folders "test" and "train".
#       The test folder should be a visual graph dataset represenation of the test split of the dataset. The train 
#       folder should contain multiple pickled files, that each contain the serialized representations of the Tensor 
#       instances that are required to directly train the model.
CACHED_PATH = os.path.join(PATH, 'cache', 'aggregators_binary_protonated')

# == TRAINING PARAMETERS ==
# These are the parameters that are relevant to the training process itself, so for example the number of epochs or 
# the batch size.

# :param LEARNING_RATE:
#       The learning rate for the optimizer
LEARNING_RATE: float = 1e-4
# :param BATCH_SIZE:
#       The number of elements to use in one batch during training
BATCH_SIZE: int = 256
# :param EPOCHS:
#       This is the number of epochs to train the model for
EPOCHS: int = 10

# == MODEL PARAMETERS ==
# These are the parameters that control the setup and configuration of the MEGAN model architecture to be trained.

# :param UNITS:
#       This list defines the message passing layers structure of the model. Each new entry in this list configues 
#       the model with an additional message passing (graph attention) layer and the integer value determines the 
#       size of the layer's hidden units.
UNITS: t.List[int] = [64, 64, 64]
# :param EMBEDDING_UNITS:
#       This list defines the message passing layers structure of the model. Each new entry in this list configues 
#       the model with an additional graph embedidng layer and the integer value determines the 
#       size of the layer's hidden units.
EMBEDDING_UNITS: t.List[int] = [64, 32]
# :param FINAL_UNITS:
#       This list defines the final prediction MLP layer structure of the model. Each new entry in this list configures 
#       the model with an additonal dense layer and the integer value determines the number of hidden units.
#       NOTE that the last value in this list should always be ==2 for the binary classification problem.
FINAL_UNITS: t.List[int] = [2, ]
# :param IMPORTANCE_FACTOR:
#       This determines the weighting factor of the explanation co-training objective of the network.
IMPORTANCE_FACTOR: float = 2.0
# :param IMPORTANCE_CHANNELS:
#       This determines the number of explanation channels the model should employ. Keep this as =2
IMPORTANCE_CHANNELS: int = 2
# :param IMPORTANCE_MULTIPLIER:
#       This is a hyperparameter of the additional explanation training procedure performed for MEGAN
#       models when (IMPORTANCE_FACTOR != 0). This parameter determines what the expected size of the
#       explanations is. If this parameter is reduced, the explanations will generally consist of less
#       elements (nodes, edges). Vice versa, a larger value will make explanations consist of more elements.
IMPORTANCE_MULTIPLIER = 0.3  # 2.0
# :param SPARSITY_FACTOR:
#       The coefficient for the sparsity regularization loss.
SPARSITY_FACTOR = 1.0
# :param FIDELITY_FACTOR:
#       This is the weighting factor for the "fidelity training" step. This fidelity training step will try
#       to directly train the fidelity of each explanation channel such that the contributions of these
#       channels to the final prediction result align as much as possible with their pre-determined
#       interpretations.
FIDELITY_FACTOR = 0.2
# :param FIDELITY_FUNCS:
#       There need to be as many functions as there are importance channels in this the model. Each function
#       receives the vectors of the original and the leave-one-in modified model predictions and is supposed
#       to reduce that to a vector of loss values which promote the channels to behave according to the
#       pre-determined interpretations. In the case of a classification problem we want the channels to
#       contain evidence which supports that class aka the leave-one-in modification should increase the
#       confidence for the corresponding class.
FIDELITY_FUNCS = [
    lambda org, mod: tf.nn.relu(-(org[:, 1] - mod[:, 1])) + tf.square(org[:, 0] - mod[:, 0]),
    lambda org, mod: tf.nn.relu(-(org[:, 0] - mod[:, 0])) + tf.square(org[:, 1] - mod[:, 1]),
]


__DEBUG__ = True

experiment = Experiment.extend(
    'train_megan_chunked.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)

experiment.run_if_main()