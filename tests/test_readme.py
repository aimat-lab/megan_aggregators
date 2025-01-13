"""
Tests the code snippets from the README file.
"""
from .util import ARTIFACTS_PATH


def test_quickstart():
    
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
        
        
def test_explaining():
    
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
    fig.savefig(f'{ARTIFACTS_PATH}/explanations.png')