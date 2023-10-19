import typing as t

import numpy as np
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace
from graph_attention_student.models.megan import Megan

from megan_aggregators.util import load_model


experiment = Experiment.extend(
    'train_megan.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)


@experiment.hook('modify_dataset')
def modify_dataset(e: Experiment, graph_list: t.List[dict]):
    e.log('loading existing aggregator model...')
    model: Megan = load_model()

    e.log('starting to add explanation descriptors to the dataset...')
    batch_size = 10_000
    current_index = 0
    while current_index < len(graph_list):
        num_elements = min(batch_size, len(graph_list) - current_index)
        indices = list(range(current_index, current_index+num_elements))
        graphs = [graph_list[i] for i in indices]
        predictions = model.predict_graphs(graphs)

        for graph, (out, ni, ei) in zip(graphs, predictions):
            graph['node_attributes'] = np.concatenate(
                [graph['node_attributes'], ni],
                axis=-1,
            )
            graph['edge_attributes'] = np.concatenate(
                [graph['edge_attributes'], ei],
                axis=-1,
            )

        current_index += batch_size
        e.log(f' * ({current_index}/{len(graph_list)})')

    return graph_list


experiment.run_if_main()
