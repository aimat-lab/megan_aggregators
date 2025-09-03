import numpy as np

from rich.pretty import pprint
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint
from megan_aggregators import load_model
from megan_aggregators import load_processing
from megan_aggregators.utils import generate_counterfactuals_with_model

np.set_printoptions(precision=2)

console = Console()

## --- Loading the Model ---
model = load_model()

## --- Preparing Input Molecule ---
smiles = 'Oc1c(I)cc(Cl)c2cccnc12'
processing = load_processing()
graph = processing.process(smiles)

## --- Original Prediction ---
prediction = model.predict_graphs([graph])[0]
result = np.argmax(prediction)
print(f'original smiles: {smiles} - label: {result}')

## --- Generating Counterfactuals ---
# "generate_counterfactuals" is a utility function that can be used to generate counterfactuals for a given 
# input molecule and model. The result of this operation will be a list of tuples, where each tuple contains
# the following elements:
# 1. The SMILES string of the counterfactual molecule
# 2. The model's prediction for this counterfactual molecule
# 3. The distance between the original and the counterfactual molecule
# The "num" parameter can be used to specify the number of counterfactuals that should be generated.
# The "k_neighborhood" parameter can be used to specify the number of edits that should be made to the original
# molecule in order to generate the counterfactuals.
results: tuple[str, np.ndarray, float] = generate_counterfactuals_with_model(
    model=model,
    smiles=smiles,
    num=10,
    k_neighborhood=1,
)
print('raw results:')
pprint(results, max_length=3)

# Create a rich table for counterfactuals
table = Table(title="🧪 Generated Counterfactuals", show_header=True, header_style="bold magenta")
table.add_column("Index", style="dim", width=6)
table.add_column("SMILES", style="cyan", no_wrap=False)
table.add_column("Predicted Label", style="green", justify="center")
table.add_column("Distance", style="yellow", justify="right")

for i, result in enumerate(results, 1):
    smiles_str = result[0]
    label = np.argmax(result[1])
    if np.argmax(result[1]) == 0:
        label = 'non-aggregator (0)'
    else:
        label = 'aggregator (1)'
    distance = f"{result[2]:.2f}"
    
    table.add_row(str(i), smiles_str, str(label), distance)

console.print(table)