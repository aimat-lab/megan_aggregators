|made-with-python| |python-version| |version|

.. |made-with-python| image:: https://img.shields.io/badge/Made%20with-Python-1f425f.svg
   :target: https://www.python.org/

.. |python-version| image:: https://img.shields.io/badge/Python-3.8.0-green.svg
   :target: https://www.python.org/

.. |version| image:: https://img.shields.io/badge/version-0.1.0-orange.svg
   :target: https://www.python.org/


👩‍🏫 MEGAN: Aggregators Dataset
==============================

This repository implements the training of a self-explaining MEGAN_ graph neural network model for the
``aggregators_binary`` dataset. The primary task is to classify molecular graphs into the two classes
"aggregator" and "non-aggregator".
Aside from that, The MEGAN model additionally creates node and edge attributional
explanations for each individual prediction.


🔔 News
-------

- **May 2023** Added the aggregation model to the MeganExplains web interface: `MeganExplains Aggregation <https://megan.aimat.science/predict/megan_aggregator>`_.
  So you can test out the model without having to install it!
- **August 2023** - Check out the arxiv preprint of the `paper`_ here: https://arxiv.org/abs/2306.02206


📦 Installation by Source
-------------------------

To install the code, one has to first clone the repository from GitHub:

.. code-block:: shell

    git clone https://github.com/aimat-lab/megan_aggregators

The package should best be installed into a Python 3.10 environment.

To get started, it is recommended to use ``conda`` and create a new environment to install the package. Note that 
pytorch needs to be explicitly installed before installing the ``megan_aggregators`` package.

.. code-block:: shell

    conda create -m agg python=3.10
    conda activate agg
    pip install torch==2.3.1
    pip -e install megan_aggregators/

**Optional.** On Linux it might be necessary to install Tk if not already installed

.. code-block:: shell

    sudo apt install python3-tk

**Checking the installation.** Afterwards, you can check the installation by running the quickstart example script:

.. code-block:: shell

    python -m megan_aggregators.examples.00_quickstart

**⚠️ Installation on Virtual Machines.** On virtual machines, ``cuda`` is usually not available. However, installing pytorch with 
the above mentioned method will result in segmentation faults.
Instead, you can replace the ``torch`` installation with the cpu-only version like this:

.. code-block:: shell

    pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
    pip install torch_scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.3.1+cpu.html

🚀 Quickstart
-------------

Using the Model
~~~~~~~~~~~~~~~

The easiest way to get started is to use the pre-trained model instance that is packaged with the code. 

This model can locally be loaded and is ready to make aggregation predictions within a few lines of code:

.. code-block:: python

    from megan_aggregators import predict_aggregator
    from megan_aggregators import get_protonations
    from megan_aggregators import generate_counterfactuals

    SMILES: str = 'CCC(CCN)CCC'

    # ~ Aggregation Prediction
    # The "predict_aggregator" function performs an aggregation prediction for the given SMILES 
    # string using the default model and returns the probability of the molecule being an aggregator.
    probability: float = predict_aggregator(SMILES)
    label = 'aggregator' if probability > 0.5 else 'non-aggregator'
    print(f'The molecule {SMILES} is classified as {label} ({probability*100:.2f}% aggregator)')

    # ~ Protonation States
    # The "get_protonations" function generates all possible protonation states for the given SMILES
    # string within the given pH range. The output of the function will be a list of multiple SMILES 
    # strings which represent the different protonation states.
    print('Protonation states:')
    protonated_smiles = get_protonations(SMILES, min_ph=6.4, max_ph=6.4)
    for smiles in protonated_smiles:
        _probability: float = predict_aggregator(smiles)
        print(f' * {smiles} is classified as ({_probability*100:.2f}% aggregator)')

    # ~ Counterfactual Generation
    # The "generate_counterfactuals" fucntion generates the counterfactuals for the given SMILES 
    # string representation of a molecule. These counterfactuals are molecules which are structurally 
    # similar to the original molecule but cause a strongly different prediction by the model. 
    # The function returns a list of tuples where the first value of the tuple is the counterfactual 
    # SMILES string and the second value is the models prediction array and the third value is the 
    # difference in the predicted probabilities.
    counterfactuals: list[tuple[str, list, float]] = generate_counterfactuals(SMILES, 10)
    print(f'Counterfactuals for {SMILES}')
    for smiles, array, distance in counterfactuals:
        print(f' * {smiles:20} ({array[0] * 100:.2f}% aggregator) - distance: {distance:.2f}')


Explaining Predictions
~~~~~~~~~~~~~~~~~~~~~~

The MEGAN model is a *self-explaining graph neural network* which means that it is able to produce explanations 
in addition to the target class predictions. These explanations are supposed to illustrate the structure-property 
relationships that were influential for each of the model's decisions. These explanations come in the format of 
attetion maps. For each prediction, the explanation consists of a set of values between 0 and 1 that are associated 
with each node and each edge of a molecule. Higher attention values indicate that a higher importance of certain 
substructurs for the outcome of the prediction.

The MEGAN model employs a multi-explanation scheme whereby multiple different explanations are created - one for 
each possible output class. In the case of the aggregation prediction, the model will therefore always produce 
2 explanations: One which illustrates the structural evidence in favor of the "aggregator" class and another 
for the evidence for the "non-aggregator" class.

.. code-block:: python

    from megan_aggregators import load_processing
    from megan_aggregators import load_model
    from megan_aggregators import visualize_explanations

    # We can create the model and the input graph as before
    model = load_model()
    processing = load_processing()

    smiles = 'CCC(CCN)CCC'
    graph = processing.process(smiles)

    # The model's method "explain_graphs" can be used to create these explanations masks
    # for the input graph.
    # The result of this operation will be the combined node and edge explanation arrays
    # with the following shapes:
    # node_importances: (number of atoms, 2)
    # edge_importances: (number of bonds, 2)
    info = model.forward_graphs([graph])[0]
    node_importances = info['node_importance']
    edge_importances = info['edge_importance']

    # ~ visualizing the explanation
    # This utility function will visualize the different explanations channels into
    # separate axes within the same figure.
    fig = visualize_explanations(
        smiles,
        processing,
        node_importances,
        edge_importances,
    )

    # Finally we can save the figure as a file to look at it
    fig.savefig('explanations.png')


🧪 Experiments
--------------

All the computational experiments performed in the context of this project are implemented in the PyComex_ micro framework for 
computation experimentation. In this framework, each experiment is implemented as an individual python module ``.py`` file. 

All the experiment modules can be found in the ``megan_aggregators/experiments`` folder. The most important subset of experiments 
will be described below:

- ``train_megan.py`` - This experiment will train a MEGAN model, if provided a valid path to a binary classification visual 
  graph dataset.

📖 Referencing
--------------

If you use, extend or otherwise mention or work, please cite `the paper <https://arxiv.org/abs/2306.02206>`_ as follows:

.. code-block:: bibtex

    @article{sturm2023mitgating
        title={Mitigating Molecular Aggregation in Drug Discovery with Predictive Insights from Explainable AI},
        author={Sturm, Hunter and Teufel, Jonas and Kaitlin A., Isfeld and Friederich, Pascal and Davis, Rebecca L.},
        journal={arxiv.org},
        year={2023}
    }


🫱🏻‍🫲🏾 Credits
-----------

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