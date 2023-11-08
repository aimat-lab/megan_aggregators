|made-with-python| |python-version| |version|

.. |made-with-python| image:: https://img.shields.io/badge/Made%20with-Python-1f425f.svg
   :target: https://www.python.org/

.. |python-version| image:: https://img.shields.io/badge/Python-3.8.0-green.svg
   :target: https://www.python.org/

.. |version| image:: https://img.shields.io/badge/version-0.1.0-orange.svg
   :target: https://www.python.org/

==========================
MEGAN: Aggregators Dataset
==========================

This repository implements the training of a self-explaining MEGAN_ graph neural network model for the
``aggregators_binary`` dataset. The primary task is to classify molecular graphs into the two classes
"aggregator" and "non-aggregator".
Aside from that, The MEGAN model additionally creates node and edge attributional
explanations for each individual prediction.

=======
🔔 News
=======

- **May 2023** Added the aggregation model to the MeganExplains web interface: `MeganExplains Aggregation <https://megan.aimat.science/predict/megan_aggregator>`_.
  So you can test out the model without having to install it!
- **August 2023** - Check out the arxiv preprint of the `paper`_ here: https://arxiv.org/abs/2306.02206

=========================
📦 Installation by Source
=========================

first clone the repository:

.. code-block:: shell

    git clone https://github.com/aimat-lab/megan_aggregators

Then in the main folder run a ``pip install``:

.. code-block:: shell

    cd megan_aggregators
    python3 -m pip -e install .

On Linux it might be necessary to install Tk if not already installed

.. code-block:: shell

    sudo apt install python3-tk

Afterwards, you can check the install by invoking the CLI:

.. code-block:: shell

    python3 -m megan_aggregators.cli --version
    python3 -m megan_aggregators.cli --help

=============
🚀 Quickstart
=============

The easiest way to get started is to use the saved model instance that comes shipped with the code. This model 
can locally be loaded and is ready to make aggregation predictions within a few lines of code:

.. code-block:: python

    import tensorflow.keras as ks
    from megan_aggregators.util import load_model

    # This will load the MEGAN keras model which can be used to make predictions.
    model: ks.models.Model = load_model()
    
    smiles = 'CCC(CCN)CCC'
    # This model can now make predictions about given molecules. However, these molecules first have to be 
    # converted into the appropriate graph representation such that the model can understand them.
    # This can be done with a "processing" instance.
    processing = load_processing()
    graph = processing.process(smiles)

    # "prediction" is a numpy array with the shape (2, ) where the first of the two elements is the 
    # classifiation logits for the "non-aggregator" class and the second value is the classification 
    # logits for the "aggregator" class. 
    predition = model.predict_graphs([graph])[0]

    # The predicted label can be applying the argmax function.
    # 0 - non-aggregator
    # 1 - aggregator
    result = np.argmax(prediction)

================
💡 Documentation
================

Currently, the documentation is given the in form of 

==============
🧪 Experiments
==============

=================
🤖 Model Training
=================

Downloading the Dataset
=======================

The ``aggregators_binary`` dataset can be downloaded from the following URL:
https://bwsyncandshare.kit.edu/s/pGExzNEkjbadKHw
It is in the format of a VisualGraphDataset_, which means that the dataset is represented as a folder
where each element is represented by two files: One JSON file which contains the entire pre-processed graph
representation of the corresponding element, and one PNG file which depicts a visualization of the molecule
that is later used to visualize the attributional explanations.

Since this dataset is rather large with ~400.000 molecules, the dataset is about 20GB. Thus, availability
of a high-speed internet connection and an SSD storage device are highly recommended.

Model Training
==============

The model training can be performed by executing the python module
``megan_aggregators/experiments/train_megan.py``. **Before executing**, however, the value of the global
variable ``VISUAL_GRAPH_DATASET_PATH`` has to be set to wherever the dataset was downloaded to on the local
system. Additionally, there are several other global variables which can be used to configure the model and
the training process.

Due to the large dataset size, the training will take a considerable amount of time. Also note that the
execution of the training process will require **at least 32GB of RAM**.

After the experiment is finished, the results and several visualizations and artifacts can be found in the
``megan_aggregators/experiments/results`` folder. These artifacts for example include a confusion matrix
for the classification results on the test set and example visualizations of the generated explanations on
a subset of the test set.

==============
📖 Referencing
==============

If you use, extend or otherwise mention or work, please cite `the paper <https://arxiv.org/abs/2306.02206>`_ as follows:

.. code-block:: bibtex

    @article{sturm2023mitgating
        title={Mitigating Molecular Aggregation in Drug Discovery with Predictive Insights from Explainable AI},
        author={Sturm, Hunter and Teufel, Jonas and Kaitlin A., Isfeld and Friederich, Pascal and Davis, Rebecca L.},
        journal={arxiv.org},
        year={2023}
    }

==========
🫱🏻‍🫲🏾 Credits
==========

* PyComex_ is a micro framework which simplifies the setup, processing and management of computational
  experiments. It is also used to auto-generate the command line interface that can be used to interact
  with these experiments.
* VisualGraphDataset_ is a library which aims to establish a special dataset format specifically for graph
  XAI applications with the aim of streamlining the visualization of graph explanations and to make them
  more comparable by packaging canonical graph visualizations directly with the dataset.
* MEGAN_ Multi-Explanation Graph Attention Network: Is a self-explaining GNN variant, which generates
  attributional explanations along multiple independent channels alongside the primary predictions.
* KGCNN_ Is a library for the creation of graph neural networks based on the RaggedTensor feature of the
  Tensorflow/Keras machine learning framework.

.. _PyComex: https://github.com/the16thpythonist/pycomex
.. _VisualGraphDataset: https://github.com/awa59kst120df/visual_graph_datasets
.. _MEGAN: https://github.com/awa59kst120df/graph_attention_student
.. _KGCNN: https://github.com/aimat-lab/gcnn_keras

.. _`paper`: https://arxiv.org/abs/2306.02206