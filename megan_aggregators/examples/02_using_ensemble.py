import numpy as np

from megan_aggregators.utils import load_processing
from megan_aggregators.models import load_ensemble

# This will load the default ensemble consisting of a selection of the best models.
ensemble = load_ensemble()

# Constructing the graph represention from the SMILES string
smiles = 'CCC(CCN)CCC'
processing = load_processing()
graph = processing.process(smiles)

# Since the ensemble class implements the same interface as a single model instance, it is possible
# to use the same methods to make predictions
prediction = ensemble.predict_graphs([graph])[0]

# The predicted label can be applying the argmax function.
# 0 - non-aggregator
# 1 - aggregator
result = np.argmax(prediction)
print(prediction, result)