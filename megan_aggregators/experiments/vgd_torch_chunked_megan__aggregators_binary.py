import os
import typing as t

from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from megan_aggregators.utils import EXPERIMENTS_PATH

# == DATASET PARAMETERS ==
# These parameters determine where to load the dataset for the training.

# :param CHUNKED_DATASET_PATH:
#       This parameter is supposed to be a string path to the folder that contains the chunked version of the dataset.
#       This folder should contain two subfolders "train" and "test" which in turn contain the chunked version of the
#       training and test set in visual graph dataset format respectively.
CHUNKED_DATASET_PATH: str = os.path.join(EXPERIMENTS_PATH, 'assets', 'cache', 'aggregators_combined')
# :param DATASET_TYPE:
#       This is the parameter that determines the type of the dataset. It may be either "classification" or "regression".
DATASET_TYPE: str = 'classification'
# :param TARGET_NAMES:
#       This dictionary is supposed to contain the human readable names of the target values of the dataset. The keys
#       are the integer indices of the target values and the values are the human readable names of the target values.
TARGET_NAMES: dict = {
    0: 'aggregator',
    1: 'non-aggregator',
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
        'name': 'aggregator',
        'color': 'skyblue',
    },
    1: {
        'name': 'non-aggregator',
        'color': 'coral',
    }
}
# :param UNITS:
#       This list determines the layer structure of the model's graph encoder part. Each element in 
#       this list represents one layer, where the integer value determines the number of hidden units 
#       in that layer of the encoder network.
UNITS: t.List[int] = [64, 64, 64]
HIDDEN_UNITS = 128
# :param IMPORTANCE_UNITS:
#       This list determines the layer structure of the importance MLP which determines the node importance 
#       weights from the node embeddings of the graph. 
#       Each element in this list represents one layer where the integer value determines the number of hidden 
#       units in that layer.
IMPORTANCE_UNITS: t.List[int] = []
# :param PROJECTION_LAYERS:
#       This list determines the layer structure of the MLP's that act as the channel-specific projections.
#       Each element in this list represents one layer where the integer value determines the number of hidden
#       units in that layer.
PROJECTION_UNITS: t.List[int] = [64, 128, 256]
# :param FINAL_UNITS:
#       This list determines the layer structure of the model's final prediction MLP. Each element in 
#       this list represents one layer, where the integer value determines the number of hidden units 
#       in that layer of the prediction network.
#       Note that the last value of this list determines the output shape of the entire network and 
#       therefore has to match the number of target values given in the dataset.
FINAL_UNITS: t.List[int] = [64, 2]
# :param IMPORTANCE_FACTOR:
#       This is the coefficient that is used to scale the explanation co-training loss during training.
#       Roughly, the higher this value, the more the model will prioritize the explanations during training.
IMPORTANCE_FACTOR: float = 1.0
# :param IMPORTANCE_OFFSET:
#       This parameter more or less controls how expansive the explanations are - how much of the graph they
#       tend to cover. Higher values tend to lead to more expansive explanations while lower values tend to 
#       lead to sparser explanations. Typical value range 0.5 - 1.5
IMPORTANCE_OFFSET: float = 0.75
# :param SPARSITY_FACTOR:
#       This is the coefficient that is used to scale the explanation sparsity loss during training.
#       The higher this value the more explanation sparsity (less and more discrete explanation masks)
#       is promoted.
SPARSITY_FACTOR: float = 0.1
# :param FIDELITY_FACTOR:
#       This parameter controls the coefficient of the explanation fidelity loss during training. The higher
#       this value, the more the model will be trained to create explanations that actually influence the
#       model's behavior with a positive fidelity (according to their pre-defined interpretation).
#       If this value is set to 0.0, the explanation fidelity loss is completely disabled (==higher computational
#       efficiency).
FIDELITY_FACTOR: float = 0.1
# :param REGRESSION_REFERENCE:
#       When dealing with regression tasks, an important hyperparameter to set is this reference value in the 
#       range of possible target values, which will determine what part of the dataset is to be considered as 
#       negative / positive in regard to the negative and the positive explanation channel. A good first choice 
#       for this parameter is the average target value of the training dataset. Depending on the results for 
#       that choice it is possible to adjust the value to adjust the explanations.
REGRESSION_REFERENCE: t.Optional[float] = None
# :param REGRESSION_MARGIN:
#       When converting the regression problem into the negative/positive classification problem for the 
#       explanation co-training, this determines the margin for the thresholding. Instead of using the regression
#       reference as a hard threshold, values have to be at least this margin value lower/higher than the 
#       regression reference to be considered a class sample.
REGRESSION_MARGIN: t.Optional[float] = None
# :param ATTENTION_AGGREGATION:
#       This string literal determines the strategy which is used to aggregate the edge attention logits over 
#       the various message passing layers in the graph encoder part of the network. This may be one of the 
#       following values: 'sum', 'max', 'min'.
ATTENTION_AGGREGATION: str = 'max'
# :param NORMALIZE_EMBEDDING:
#       This boolean value determines whether the graph embeddings are normalized to a unit length or not.
#       If this is true, the embedding of each individual explanation channel will be L2 normalized such that 
#       it is projected onto the unit sphere.
NORMALIZE_EMBEDDING: bool = True
# :param CONTRASTIVE_FACTOR:
#       This is the factor of the contrastive representation learning loss of the network. If this value is 0 
#       the contrastive repr. learning is completely disabled (increases computational efficiency). The higher 
#       this value the more the contrastive learning will influence the network during training.
CONTRASTIVE_FACTOR: float = 1.0
# :param CONTRASTIVE_NOISE:
#       This float value determines the noise level that is applied when generating the positive augmentations 
#       during the contrastive learning process.
CONTRASTIVE_NOISE: float = 0.1
# :param CONTRASTIVE_TEMP:
#       This float value is a hyperparameter that controls the "temperature" of the contrastive learning loss.
#       The higher this value, the more the contrastive learning will be smoothed out. The lower this value,
#       the more the contrastive learning will be focused on the most similar pairs of embeddings.
CONTRASTIVE_TEMP: float = 1.0
# :param CONTRASTIVE_BETA:
#       This is the float value from the paper about the hard negative mining called the concentration 
#       parameter. It determines how much the contrastive loss is focused on the hardest negative samples.
CONTRASTIVE_BETA: float = 1.0
# :param CONTRASTIVE_TAU:
#       This float value is a hyperparameters of the de-biasing improvement of the contrastive learning loss. 
#       This value should be chosen as roughly the inverse of the number of expected concepts. So as an example 
#       if it is expected that each explanation consists of roughly 10 distinct concepts, this should be chosen 
#       as 1/10 = 0.1
CONTRASTIVE_TAU: float = 0.01
# :param PREDICTION_FACTOR:
#       This is a float value that determines the factor by which the main prediction loss is being scaled 
#       durign the model training. Changing this from 1.0 should usually not be necessary except for regression
#       tasks with a vastly different target value scale.
PREDICTION_FACTOR: float = 1.0
# :param LABEL_SMOOTHING:
#       This is a float value that determines the amount of label smoothing to be applied on the classification 
#       target values. This regularizes the model to not be too confident about the target values and can help
#       to prevent overfitting.
LABEL_SMOOTHING: float = 0.0
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
OUTPUT_NORM: t.Optional[float] = None


# == TRAINING PARAMETERS ==
# These parameters determine the training process itself.

# :param BATCH_SIZE:
#       The number of elements to be processed in one batch during the training process.
BATCH_SIZE: int = 100
# :param EPOCHS:
#       The number of epochs to train the model for.
EPOCHS: int = 10
# :param LEARNING_RATE:
#       The learning rate for the model training process.
LEARNING_RATE: float = 1e-4

__TESTING__ = False

WANDB_PROJECT = 'megan_aggregators'
WANDB_PROJECT = None

experiment = Experiment.extend(
    'vgd_torch_chunked_megan.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)

experiment.run_if_main()