"""
This experiment trains a MEGAN model to solve a binary classification problem based on a given visual graph 
dataset.
This experiment trains the MEGAN manner in a specific manner that allows to train for particularly large datasets 
even with limited memory (RAM) resources.

The main difference with this experiment is that it is specifically designed to support the training with a very large 
dataset. Using such a very large dataset creates various challanges. For example the dataset can be so large that 
it will not fit into RAM memory of most conventional computing systems. This means that the dataset will have to be loaded 
into memory only in subsets / chunks which have to be processed sequentially. This of course complicates the implementation 
of the training procedure a bit.

Here, this is implemented in the following manner. The dataset still has to be provided in the visual graph dataset format. 
This dataset will be initially loaded into the memory in small chunks and then aggregated until the given caching chunk 
size is reached. At that point, all collected elements will be converted into tensorflow Tensor instances which could directly 
be used for the model training. These tensors are then directly saved to a disk file in the cache folder using pickle. Then 
later on during the training only one of these tensor chunks is loaded at the same time to perform the training. If all the 
batches in one chunk were completely processed, the next chunk is loaded until an epoch ends when all chunks have been 
processed once. 
"""
import os
import random
import pathlib
import logging
import pickle
import typing as t
from collections import Counter

import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from kgcnn.data.utils import ragged_tensor_from_nested_numpy
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path
from graph_attention_student.models.megan import Megan2
from graph_attention_student.training import NoLoss
from graph_attention_student.data import tensors_from_graphs
from graph_attention_student.models import load_model
from graph_attention_student.fidelity import leave_one_out_analysis
from graph_attention_student.visualization import plot_leave_one_out_analysis
from visual_graph_datasets.data import VisualGraphDatasetReader
from visual_graph_datasets.data import VisualGraphDatasetWriter
from visual_graph_datasets.visualization.importances import plot_node_importances_background
from visual_graph_datasets.visualization.importances import plot_edge_importances_background
from visual_graph_datasets.visualization.importances import create_importances_pdf

from megan_aggregators.utils import NULL_LOGGER

PATH = pathlib.Path(__file__).parent.absolute()
CACHE_PATH = os.path.join(PATH, 'cache')

# == DATASET PARAMETERS ==
# These are the parameters related to the dataset and the processing thereof.

# :param VISUAL GRAH_DATASET:
#       This has to be the absolute string path pointing towards the visual graph dataset to be used for the training 
#       of the model.
VISUAL_GRAPH_DATASET: str = '/media/ssd/.visual_graph_datasets/datasets/rb_adv_motifs'
# :param NUM_TEST:
#       This is the number of elements to sample for the test set on which the performance of the model will be 
#       evaluated
NUM_TEST: int = 1000
# :param CHUNK_SIZE:
#       This is the size of the chunks into which the dataset will be split during the pre-processing. So this is 
#       the number of elements that will be loaded into the actual memory at the same time. Therefore this number 
#       has to be chosen appropriately for the available hardware.
CHUNK_SIZE: int = 1000
# :param OVERSAMPLING_FACTORS:
#       This is a dictionary where the keys are the integer identifiers for the possible classes of the dataset
#       and the associated values are the corresponding oversampling factors to be applied to the elements of that 
#       class. An oversampling factor of 5 for example means that every element of that class will be duplicated 5 
#       times as part of the dataset. In this specific case, the oversampling is generally necessary because the 
#       underlying dataset is highly imbalanced (meaning that one class appears significantly more often than the other)
OVERSAMPLING_FACTORS: dict = {
    0: 30,   # non-aggregator
    1: 1,  # aggregator
}

# == TRAINING PARAMETERS ==
# These are the parameters that are relevant to the training process itself, so for example the number of epochs or 
# the batch size.

# :param BATCH_SIZE:
#       The number of elements to use in one batch during training
BATCH_SIZE: int = 32
# :param EPOCHS:
#       This is the number of epochs to train the model for
EPOCHS: int = 10



# == MODEL PARAMETERS ==
# These are the parameters that control the setup and configuration of the MEGAN model architecture to be trained.

# :param UNITS:
#       This list defines the message passing layers structure of the model. Each new entry in this list configues 
#       the model with an additional message passing (graph attention) layer and the integer value determines the 
#       size of the layer's hidden units.
UNITS: t.List[int] = [16, 16, 16]
# :param FINAL_UNITS:
#       This list defines the final prediction MLP layer structure of the model. Each new entry in this list configures 
#       the model with an additonal dense layer and the integer value determines the number of hidden units.
#       NOTE that the last value in this list should always be ==2 for the binary classification problem.
FINAL_UNITS: t.List[int] = [16, 2]
# :param IMPORTANCE_FACTOR:
#       This determines the weighting factor of the explanation co-training objective of the network.
IMPORTANCE_FACTOR: float = 1.0
# :param IMPORTANCE_CHANNELS:
#       This determines the number of explanation channels the model should employ. Keep this as =2
IMPORTANCE_CHANNELS: int = 2
# :param IMPORTANCE_MULTIPLIER:
#       This is a hyperparameter of the additional explanation training procedure performed for MEGAN
#       models when (IMPORTANCE_FACTOR != 0). This parameter determines what the expected size of the
#       explanations is. If this parameter is reduced, the explanations will generally consist of less
#       elements (nodes, edges). Vice versa, a larger value will make explanations consist of more elements.
IMPORTANCE_MULTIPLIER = 0.5  # 2.0
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

# == EVALUATION PARAMETERS ==

NUM_EXAMPLES: int = 100


__DEBUG__ = True


@Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)
def experiment(e: Experiment):
    e.log('starting experiment...')
    
    # ~ pre-processing the dataset
    # ---
    
    assert os.path.exists(e.VISUAL_GRAPH_DATASET), 'Given dataset path does not exist on the local system!'
    
    e.log('setting up the dataser reader...')
    reader = VisualGraphDatasetReader(
        path=e.VISUAL_GRAPH_DATASET,
        logger=e.logger,    
    )
    dataset_length = len(reader)
    e.log(f'processing a dataset with {dataset_length} elements')
    
    @e.hook('save_cache_graphs')
    def save_cache_graphs(e: Experiment,
                          graphs: t.List[dict],
                          path: str,
                          ):

        # Then we serialize and save the input. The network input consists of three different tensors which provide the 
        # information about the graph nodes, edges and edge index lists.
        # We need to save these three tensors into distinct files.
        x = tensors_from_graphs(graphs)
        with open(os.path.join(path, 'x.pkl'), mode='wb') as file:
            pickle.dump(x, file)
            del x
    
        # The corresponding target labels for those graphs we actually have to save as a numpy array and not a tensor!
        targets = np.array([graph['graph_labels'] for graph in graphs], dtype=np.float32)
        ni = ragged_tensor_from_nested_numpy([np.zeros((graph['node_attributes'].shape[1], e.IMPORTANCE_CHANNELS)) for graph in graphs])
        ei = ragged_tensor_from_nested_numpy([np.zeros((graph['edge_attributes'].shape[1], e.IMPORTANCE_CHANNELS)) for graph in graphs])
        y = (targets, ni, ei)
        with open(os.path.join(path, 'y.pkl'), mode='wb') as file:
            pickle.dump(y, file)
        
    @e.hook('load_cache_graphs')
    def load_cache_graphs(e: Experiment,
                          path: str,
                          ):
        
        with open(os.path.join(path, 'x.pkl'), mode='rb') as file:
            x = pickle.load(file)
    
        with open(os.path.join(path, 'y.pkl'), mode='rb') as file:
            y = pickle.load(file)
    
        return x, y
    
    # We save all the raw tensor data in a custom folder in the overall experiment cache folder. However, to make this more 
    # efficient we only actually perform the whole dataset pre-processing if that folder does not exist yet. If it does exist 
    # we are simply going to re-use that.
    cache_path = os.path.join(CACHE_PATH, f'train_megan_chunked__{e.name}__cache')
    # If ...
    test_path = os.path.join(CACHE_PATH, f'train_megan_chunked__{e.name}__test')
    e['test_path'] = test_path
    
    e.log(f'cache path: {cache_path} - test path: {test_path}')
    if not os.path.exists(cache_path):
        
        os.mkdir(cache_path)
        
        # the test set
        # sampling the test set is actually a bit of a problem here. Certainly the easiest option would be to simple take the 
        # first or last N elements of the dataset in general as the test set. However, in this dataset we can expect that there 
        # is an ordering bias - which means that it is not inherently shuffled enough to get a representative sample like this.
        # Now the alternative we are going to do instead: We are going to sample a certain number of elements from each of the 
        # dataset chunks randomly to make up the test set.
        test_data_map: t.Dict[int, dict] = {}
        num_chunks = len(reader.chunk_map)
        num_test_per_chunk = int(e.NUM_TEST / num_chunks)
        
        index_class_map = {}
        # This will be a temporary dict structure where the keys are the global integer indices of the elements and the the values 
        # are the actual data elements.
        train_data_map: t.Dict[int, dict] = {}
        chunk_index = 0
        for i, data_map in enumerate(reader.chunk_iterator(randomize=True)):
            e.log(f'processing dataset chunk {i} with {len(data_map)} elements')
            if len(data_map) == 0:
                continue
            
            # As the first step we extract a number of the elements for the test set.
            for index in random.sample(list(data_map.keys()), k=num_test_per_chunk):
                test_data_map[index] = data_map[index]
                del data_map[index]
            
            # Only afterwards we aggregate the remaining elements to the training dict 
            train_data_map.update(data_map)
            index_class_map.update({index: np.argmax(data['metadata']['graph']['graph_labels']) for index, data in data_map.items()})
            
            # If the collected data is now bigger than the selected chunk size then we actually process all of that data into 
            # tensors and then save all of those tensors as one big chunk to the cache folder.
            if len(train_data_map) >= e.CHUNK_SIZE:
                
                while len(train_data_map) >= e.CHUNK_SIZE:
                    
                    graphs = []
                    for index, data in list(train_data_map.items())[:e.CHUNK_SIZE]:
                        graph = data['metadata']['graph']
                        target = np.argmax(graph['graph_labels'])
                        graphs += [graph] * e.OVERSAMPLING_FACTORS[target]
                        
                        del train_data_map[index]
                                        
                    e.log(f'saving tensor cache {chunk_index} with {len(graphs)} elements...')
                    path = os.path.join(cache_path, f'{chunk_index:02d}')
                    os.mkdir(path)
                    
                    e.apply_hook(
                        'save_cache_graphs',
                        graphs=graphs,
                        path=path,
                    )
                    chunk_index += 1
                    graphs = []
                
        class_counter = Counter(index_class_map.values())
        e.log(f'overall class distribution in the dataset: {class_counter}')
        
        # ~ writing test set
        e.log('saving the test set...')
        os.mkdir(test_path)
        writer = VisualGraphDatasetWriter(test_path, chunk_size=10_000)
        for index, data in test_data_map.items():
            data['index'] = index
            data['metadata']['image_path'] = data['image_path']
            writer.write(
                name=index,
                metadata=data['metadata'],
                figure=None,
            )
            
    # IF the cache folder does already exist we actually have to do the thing where the
    else:
        e.log('loading the test set...')
        reader = VisualGraphDatasetReader(
            path=test_path,
            logger=e.logger,
            log_step=1000
        )
        test_data_map = reader.read()
        e.log(f'loaded test dataset with {len(test_data_map)} elements')
        
    indices = list(range(dataset_length))
    test_indices = list(test_data_map.keys())
    train_indices = list(set(indices).difference(set(test_indices)))
    e['test_indices'] = test_indices
    e['train_indices'] = train_indices
    
    # ~ creating the model
    # At first we need to set up the model so that we can then use it for the training.

    @e.hook('create_model')
    def create_model(e: Experiment):
        
        e.log('creating MEGAN model...')
        model = Megan2(
            units=e.UNITS,
            importance_factor=e.IMPORTANCE_FACTOR,
            importance_channels=e.IMPORTANCE_CHANNELS,
            importance_multiplier=e.IMPORTANCE_MULTIPLIER,
            final_units=e.FINAL_UNITS,
            final_activation='linear',
            sparsity_factor=e.SPARSITY_FACTOR,
            fidelity_factor=e.FIDELITY_FACTOR,
            fidelity_funcs=e.FIDELITY_FUNCS,
            concat_heads=False,
            use_bias=True,
            use_edge_features=True,
        )
        model.compile(
            optimizer=ks.optimizers.Adam(learning_rate=1e-3),
            # metrics=[ks.metrics.CategoricalAccuracy()],
            loss=[
                ks.losses.CategoricalCrossentropy(from_logits=True),
                NoLoss(),
                NoLoss(),
            ],
            loss_weights=[1, 0, 0],
            run_eagerly=False,
        )
        
        return model

    e.log('creating model...')
    model = e.apply_hook(
        'create_model',
    )

    # ~ training the model
    
    def training_generator(epochs: int,
                           batch_size: int,
                           ):
        
        chunk_paths = [path for name in os.listdir(cache_path) if (path := os.path.join(cache_path, name)) and os.path.isdir(path)]
        
        epoch_index = 0
        try:
            while epoch_index < epochs:
                
                e.log(f'starting epoch {epoch_index}...')
                for chunk_index, path in enumerate(chunk_paths):
                    
                    e.log(f'loading chunk {chunk_index} for training')
                    x_chunk, y_chunk = e.apply_hook(
                        'load_cache_graphs',
                        path=path
                    )
                    chunk_size = len(y_chunk[0])
                    indices_chunk = list(range(chunk_size))
                    random.shuffle(indices_chunk)

                    batch_index = 0
                    while batch_index < chunk_size:
                        num_batch = min(batch_size, chunk_size - batch_index)
                        indices_batch = indices_chunk[batch_index:batch_index+num_batch]
                        x_batch = [v[indices_batch] if isinstance(v, np.ndarray) else tf.gather(v, indices_batch) for v in x_chunk]
                        y_batch = [v[indices_batch] if isinstance(v, np.ndarray) else tf.gather(v, indices_batch) for v in y_chunk]
                                            
                        batch_index += num_batch
                        # print(batch_index, x_batch, y_batch)
                        yield x_batch, y_batch
                        
                epoch_index += 1
        
        # We want to be able to externally stop the training without killing the entire program...
        except KeyboardInterrupt:
            pass
                    
    
    e.log('setting up the dataset generator...')
    generator = training_generator(e.EPOCHS, e.BATCH_SIZE)
    model.fit(
        generator,
    )
        
    # ~ saving the model
    # Now we also want to save the model so that we can use it later on.
    model_path = os.path.join(e.path, 'model')
    model.save(model_path)
        
        
@experiment.analysis
def analysis(e: Experiment):
    
    e.log('starting the analysis...')
    
    # ~ loading the model and the test set
    # Since we are in the analysis runtime env here, we need to load the model and the test dataset again 
    # here.
    e.log('loading the model...')
    model_path = os.path.join(e.path, 'model')
    model = load_model(model_path)

    e.log('loading the test set...')
    reader = VisualGraphDatasetReader(
        path=e['test_path'],
        logger=e.logger,
        log_step=1000,
    )
    test_data_map = reader.read()
    
    # ~ evaluating the model
    # after the training is done we can then evaluate the model on the test set.
    e.log(f'evaluating the model on {len(test_data_map)} elements...')
    indices_test = list(test_data_map.keys())
    graphs_test = [data['metadata']['graph'] for data in test_data_map.values()]
    
    predictions = model.predict_graphs(graphs_test)
    for index, graph, (out, ni, ei) in zip(indices_test, graphs_test, predictions):
        e[f'out/pred/{index}'] = out
        e[f'out/true/{index}'] = graph['graph_labels']
        
        e[f'ni/{index}'] = ni
        e[f'ei/{index}'] = ei
    
    e.log('leave-one-out fidelity analysis...')
    loo_results = leave_one_out_analysis(
        model=model,
        graphs=graphs_test,
        num_targets=2,
        num_channels=2,
    )
    fig = plot_leave_one_out_analysis(
        loo_results,
        num_targets=2,
        num_channels=2,
        num_bins=50,
    )
    fig_path = os.path.join(e.path, 'leave_one_out.pdf')
    plt.savefig(fig_path)
    
    e.log('evaluating test metrics...')
    values_true = np.array([v[1] for v in e[f'out/true'].values()])
    values_pred = np.array([v[1] for v in e[f'out/pred'].values()])
    labels_true = np.array([np.argmax(v) for v in e[f'out/true'].values()])
    labels_pred = np.array([np.argmax(v) for v in e[f'out/true'].values()])

    acc_value = accuracy_score(labels_true, labels_pred)
    e['acc'] = acc_value
    e.log(f' * acc: {acc_value:.2f}')
    
    f1_value = f1_score(labels_true, labels_pred)
    e['f1'] = f1_value
    e.log(f' * f1: {f1_value:.2f}')
    
    fpr, tpr, _ = roc_curve(values_true, values_pred)
    auc_value = roc_auc_score(values_true, values_pred)
    e['auc'] = auc_value
    e.log(f' * auc: {auc_value:.2f}')
    
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
    ax.plot(fpr, tpr, color='darkorange', label=f'AUC: {auc_value:.2f}')
    ax.plot([0, 1], [0, 1], color='gray')
    ax.set_title('receiver operating curve (ROC)')
    ax.set_xlabel('false positive rate')
    ax.set_ylabel('true positive rate')
    ax.legend()
    fig_path = os.path.join(e.path, 'auc.pdf')
    fig.savefig(fig_path)
    
    e.log('creating confusion matrix...')
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
    cm = confusion_matrix(labels_true, labels_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
    )
    disp.plot(
        ax=ax
    )
    fig_path = os.path.join(e.path, 'confusion_matrix.pdf')
    fig.savefig(fig_path)
    
    # ~ plotting some examples
    e.log('plotting examples...')
    indices_example = random.sample(indices_test, k=e.NUM_EXAMPLES)
    graphs_example = [test_data_map[i]['metadata']['graph'] for i in indices_example]
    image_paths_example = [test_data_map[i]['metadata']['image_path'] for i in indices_example]
    pdf_path = os.path.join(e.path, 'examples.pdf')
    create_importances_pdf(
        graph_list=graphs_example,
        image_path_list=image_paths_example,
        node_positions_list=[graph['node_positions'] for graph in graphs_example],
        importances_map={
            'test': (
                [e[f'ni/{index}'] for index in indices_example],
                [e[f'ei/{index}'] for index in indices_example],
            )
        },
        plot_node_importances_cb=plot_node_importances_background,
        plot_edge_importances_cb=plot_edge_importances_background,
        output_path=pdf_path,
        logger=e.logger,
        log_step=10,
    )
    
    
    
experiment.run_if_main()
    
    
    