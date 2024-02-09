import numpy as np
import tensorflow.keras as ks

from megan_aggregators.models import load_model
from megan_aggregators.utils import load_processing
from megan_aggregators.utils import generate_counterfactuals

np.set_printoptions(precision=2)

# This will load the MEGAN keras model which can be used to make predictions.
model: ks.models.Model = load_model()

smiles = 'Oc1c(I)cc(Cl)c2cccnc12'
processing = load_processing()
graph = processing.process(smiles)

prediction = model.predict_graphs([graph])[0]
# 0 - non-aggregator
# 1 - aggregator
result = np.argmax(prediction)
print(f'original smiles: {smiles} - label: {result}')

# "generate_counterfactuals" is a utility function that can be used to generate counterfactuals for a given 
# input molecule and model. The result of this operation will be a list of tuples, where each tuple contains
# the following elements:
# 1. The SMILES string of the counterfactual molecule
# 2. The model's prediction for this counterfactual molecule
# 3. The distance between the original and the counterfactual molecule
# The "num" parameter can be used to specify the number of counterfactuals that should be generated.
# The "k_neighborhood" parameter can be used to specify the number of edits that should be made to the original
# molecule in order to generate the counterfactuals.
results: tuple[str, np.ndarray, float] = generate_counterfactuals(
    model=model,
    smiles=smiles,
    num=10,
    k_neighborhood=1,
)

print('counterfactuals:')
for result in results:
    print(f' * smiles: {result[0]}'
          f' - label: ({np.argmax(result[1])})'
          f' - distance {result[2]:.2f}')