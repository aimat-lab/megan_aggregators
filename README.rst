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

- **May 2025** - Check out the the published Version of our Paper in *Angewandte Chemie*: https://onlinelibrary.wiley.com/doi/full/10.1002/anie.202503259 
- **January 2025** - We've released a new and refactored vesion of the web interface!
- **August 2023** - Check out the arxiv preprint of the `paper`_ here: https://arxiv.org/abs/2306.02206
- **May 2023** Added the aggregation model to the MeganExplains web interface: `MeganExplains Aggregation <https://megan.aimat.science/predict/megan_aggregator>`_.
  So you can test out the model without having to install it!

📦 Installation by Source
-------------------------

To install the code, one has to first clone the repository from GitHub:

.. code-block:: shell

    git clone https://github.com/aimat-lab/megan_aggregators

The package should best be installed into a Python 3.10 environment.

To get started, it is recommended to use ``uv`` and create a new environment to install the package. Note that 
pytorch needs to be explicitly installed before installing the ``megan_aggregators`` package due to its dependency on 
``torch_scatter`` which needs to be compiled against the installed torch version.

.. code-block:: shell

    cd megan_aggregators
    # Create and activate virtual environment
    uv venv --seed --python=3.10 .venv
    source .venv/bin/activate
    # Installation
    uv pip install torch==2.3.1
    uv pip install -e .

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
    from megan_aggregators import generate_counterfactuals

    SMILES: str = 'Oc1c(I)cc(Cl)c2cccnc12'

    ## --- Aggregation Prediction ---
    # The "predict_aggregator" function performs an aggregation prediction for the given SMILES 
    # string using the default model and returns the probability of the molecule being an aggregator.
    probability: float = predict_aggregator(SMILES)
    label = 'aggregator' if probability > 0.5 else 'non-aggregator'
    print(f'\nThe molecule {SMILES} is classified as {label} ({probability*100:.2f}% aggregator)')

    ## --- Counterfactual Generation ---
    # The "generate_counterfactuals" fucntion generates the counterfactuals for the given SMILES 
    # string representation of a molecule. These counterfactuals are molecules which are structurally 
    # similar to the original molecule but cause a strongly different prediction by the model. 
    # The function returns a list of tuples where the first value of the tuple is the counterfactual 
    # SMILES string and the second value is the models prediction array and the third value is the 
    # difference in the predicted probabilities.
    counterfactuals: list[tuple[str, list, float]] = generate_counterfactuals(SMILES, 10)
    print(f'\nCounterfactuals for {SMILES}')
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

    ## --- Getting Explanations ---
    # The model's method "forward_graphs" can be used to get the full model output, which 
    # includes not only the predictions but also the explanation masks.
    # node_importances: (number of atoms, 2)
    # edge_importances: (number of bonds, 2)
    info = model.forward_graphs([graph])[0]
    node_importances = info['node_importance']
    edge_importances = info['edge_importance']

    ## --- Visualizing Explanations ---
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

- ``predict_elements.py`` - Uses the shipped model to predict the aggregation class of a set of molecules and creates visualizations of 
  the explanations for each prediction. The elements to be predicted can be defined as an experiment parameter.
- ``generate_counterfactuals.py`` - Uses the shipped model to predict the counterfactuals for a given molecule which can be defined 
  as an experiment parameter.
- ``yang_baseline.py`` - Implements the Yang et al. baseline model for the aggregation prediction task. To execute this experiment, you'll 
  first have to download the ``aggregators_new.csv`` dataset from the file share: https://bwsyncandshare.kit.edu/s/4r9kgyCFQL6PTcF 
  and place it into the ``megan_aggregators/experiments/assets`` 
  folder.


📖 Referencing
--------------

If you use, extend or otherwise mention or work, please cite `the paper <https://arxiv.org/abs/2306.02206>`_ as follows:

.. code-block:: bibtex

    @article{sturm2025mitigating,
      author = {Sturm, Hunter and Teufel, Jonas and Isfeld, Kaitlin A. and Friederich, Pascal and Davis, Rebecca L.},
      title = {Mitigating Molecular Aggregation in Drug Discovery With Predictive Insights From Explainable AI},
      journal = {Angewandte Chemie International Edition},
      year = {2025},
      volume = {64},
      number = {29},
      doi = {10.1002/anie.202503259},
      url = {https://onlinelibrary.wiley.com/doi/full/10.1002/anie.202503259}
    }


📝 Changelog
-------------

**02.03.2023 - 0.1.0**

- initial version

**19.03.2023 - 0.2.0**

- Added the possibility to ship and load a pre-trained model with the package so that not training is 
  necessary to obtain the predictions.

**03.09.2025 - 0.3.0** 

- Fixed a bug which still used the old class indices in the `predict_aggregator` function which 
  caused the predictions to appear inverted.
- Updated the `dimorphite_dl` dependency
- Updated the example scripts

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

.. _PyComex: https://github.com/the16thpythonist/pycomex
.. _VisualGraphDataset: https://github.com/awa59kst120df/visual_graph_datasets
.. _MEGAN: https://github.com/awa59kst120df/graph_attention_student

.. _`paper`: https://arxiv.org/abs/2306.02206
.. _`angewandte`: https://onlinelibrary.wiley.com/doi/full/10.1002/anie.202503259