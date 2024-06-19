"""
This module shows an example of how to use the default API to interact with the MEGAN 
Aggregation prediction. This default API uses the pre-trained model which is shipped 
with the model. This model can be used to make predictions for new input molecules and 
can be used to generate counterfactuals for a given molecule.
"""
from megan_aggregators import predict_aggregator
from megan_aggregators import generate_counterfactuals
from megan_aggregators import get_protonations

SMILES: str = 'CCC(CCN)CCC'

# ~ Aggregation Prediction
# The "predict_aggregator" function performs an aggregation prediction for the given SMILES 
# string using the default model and returns the probability of the molecule being an aggregator.
probability: float = predict_aggregator(SMILES)
label = 'aggregator' if probability > 0.5 else 'non-aggregator'
print(f'\nThe molecule {SMILES} is classified as {label} ({probability*100:.2f}% aggregator)')

# ~ Protonation States
# The "get_protonations" function generates all possible protonation states for the given SMILES
# string within the given pH range. The output of the function will be a list of multiple SMILES 
# strings which represent the different protonation states.
print('\nProtonation states:')
protonated_smiles = get_protonations(SMILES, min_ph=6.4, max_ph=6.4)
for smiles in protonated_smiles:
    _probability: float = predict_aggregator(smiles)
    print(f' * {smiles:20} ({_probability*100:.2f}% aggregator)')

# ~ Counterfactual Generation
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
    