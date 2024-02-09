import matplotlib.pyplot as plt
from visual_graph_datasets.visualization.base import draw_image
from visual_graph_datasets.visualization.importances import plot_node_importances_background
from visual_graph_datasets.visualization.importances import plot_edge_importances_background

from megan_aggregators.utils import load_processing
from megan_aggregators.utils import visualize_explanations
from megan_aggregators.models import load_model

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
node_importances, edge_importances = model.explain_graphs([graph])[0]

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