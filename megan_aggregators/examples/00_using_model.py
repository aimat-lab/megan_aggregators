import numpy as np
import tensorflow.keras as ks
from megan_aggregators.models import load_model
from megan_aggregators.utils import load_processing

# This will load the MEGAN keras model which can be used to make predictions.
model: ks.models.Model = load_model()
# The package comes with different pre-trained models. load_model will select the best one by default,
# but different ones can be loaded by providing their string names as an argument.
# model = load_model("model_2")
# model = load_model("model_3")
# ...

smiles = 'CCC(CCN)CCC'
# This model can now make predictions about given molecules. However, these molecules first have to be
# converted into the appropriate graph representation such that the model can understand them.
# This can be done with a "processing" instance.
processing = load_processing()
graph = processing.process(smiles)

# "prediction" is a numpy array with the shape (2, ) where the first of the two elements is the
# classifiation logits for the "non-aggregator" class and the second value is the classification
# logits for the "aggregator" class.
prediction = model.predict_graphs([graph])[0]

# The predicted label can be applying the argmax function.
# 0 - non-aggregator
# 1 - aggregator
result = np.argmax(prediction)
print(prediction, result)