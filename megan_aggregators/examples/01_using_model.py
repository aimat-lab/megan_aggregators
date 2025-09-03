import numpy as np
from rich.pretty import pprint
from megan_aggregators import load_model
from megan_aggregators import load_processing

## --- Loading the Model ---
# This will load the MEGAN PyTorch model which can be used to make predictions.
model = load_model()

## --- Preparing Input Molecule ---
smiles = 'Oc1c(I)cc(Cl)c2cccnc12'
# This model can now make predictions about given molecules. However, these molecules first have to be
# converted into the appropriate graph representation such that the model can understand them.
# This can be done with a "processing" instance.
processing = load_processing()
graph: dict = processing.process(smiles)

## --- Model Prediction ---
# "prediction" is a numpy array with the shape (2, ) where the first of the two elements is the
# classifiation logits for the "non-aggregator" class and the second value is the classification
# logits for the "aggregator" class.
prediction: list = model.predict_graphs([graph])[0]

# The predicted label can be applying the argmax function.
# 0 - non-aggregator
# 1 - aggregator
class_prediction = np.argmax(prediction)
print(f'raw prediction: {prediction}')
print(f'predicted class: {class_prediction}')