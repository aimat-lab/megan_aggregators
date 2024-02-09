import os
import typing as t

import numpy as np
import tensorflow.keras as ks
import visual_graph_datasets.typing as tv

from graph_attention_student.models import CUSTOM_OBJECTS
from graph_attention_student.models import load_model as _load_model
from graph_attention_student.models.megan import Megan2 as _Megan2
from megan_aggregators.utils import ASSETS_PATH

# == GLOBAL VARIABLES ==
# Here we define some global constant variables that define the default behavior.

DEFAULT_MODEL_PATH = os.path.join(ASSETS_PATH, 'models', 'model_3')
DEFAULT_ENSEMBLE_PATHS = [
    os.path.join(ASSETS_PATH, 'models', 'model_0'),
    os.path.join(ASSETS_PATH, 'models', 'model_1'),
    os.path.join(ASSETS_PATH, 'models', 'model_2'),
    os.path.join(ASSETS_PATH, 'models', 'model_3'),
    os.path.join(ASSETS_PATH, 'models', 'model_4')
]


class PredictGraphsMixin:
    """
    This mixin defines an interface that can be implemented by a model to unify the interface of querying the model 
    for predictions.
    """
    def predict_graphs(graphs: tv.GraphDict) -> t.List[np.ndarray]:
        raise NotImplementedError()
    
    def explain_graphs(graphs: tv.GraphDict) -> t.List[tuple]:
        raise NotImplementedError()    

    
class Megan2(_Megan2, PredictGraphsMixin):
    """
    This is a wrapper around the original Megan model which implements the PredictGraphsMixin.
    
    In fact the original Megan model does already implement a ``predict_graphs`` method but the behavior of that 
    method is not consistent with the behavior that is expected here. The original method returns not only the
    actual prediction, but also the explanations all at once. To avoid confusion the PredictGraphMixin splits this 
    behavior into two seperate methods ``predict_graphs`` and ``explain_graphs``.
    """
    def predict_graphs_raw(self, graphs: t.List[tv.GraphDict]):
        return _Megan2.predict_graphs(self, graphs)

    def predict_graphs(self, graphs: t.List[tv.GraphDict]):
        outputs = self.predict_graphs_raw(graphs)
        
        predictions = [pred for pred, _, _ in outputs]
        return predictions
    
    def explain_graphs(self, graphs: t.List[tv.GraphDict]):
        outputs = self.predict_graphs_raw(graphs)
        
        explanations = [(ni, ei) for _, ni, ei in outputs]
        return explanations


class ModelEnsemble:
    
    def __init__(self,
                 models: t.List[PredictGraphsMixin],
                 aggregate_prediction_cb: t.Callable = lambda arr: np.mean(arr, axis=-1),
                 aggregate_explanation_cb: t.Callable = lambda arr: np.median(arr, axis=-1),
                 ):
        
        self.models = models
        self.aggregate_prediction = aggregate_explanation_cb
        self.aggregate_explanation = aggregate_prediction_cb
        
    def predict_graphs(self, graphs: t.List[tv.GraphDict]) -> t.List[np.ndarray]:
        predictions = [self.aggregate_prediction(arr) for arr in self.predict_graphs_all(graphs)]
        return predictions
    
    def explain_graphs(self, graphs: t.List[tv.GraphDict]) -> t.List[np.ndarray]:
        explanations = [(self.aggregate_explanation(ni), self.aggregate_explanation(ei))
                        for ni, ei in self.explain_graphs_all(graphs)]
        return explanations
        
    def predict_graphs_all(self, graphs: t.List[dict]) -> t.List[np.ndarray]:
            
        results = []
        for predictions in zip(*[model.predict_graphs(graphs) for model in self.models]):
            # "predictions" is a tuple with as many elements as there are models in the ensemble
            # every element of this tuple is the model output for one single 
            # graph of each corresponding model.
            
            outs: t.List[np.ndarray] = []
            for prediction in predictions:
                
                out: np.ndarray = prediction[0]
                out = np.expand_dims(out, axis=-1)
                outs.append(out)
            
            outs = np.concatenate(outs, axis=-1)            
            results.append(outs)
            
        return results
    
    def explain_graphs_all(self, graphs: t.List[dict]) -> t.List[t.Tuple[np.ndarray, np.ndarray]]:
        
        results = []
        for predictions in zip(*[model.predict_graphs(graphs) for model in self.models]):
            
            node_importances: t.List[np.ndarray] = []
            edge_importances: t.List[np.ndarray] = []
            for prediction in predictions:
                ni: np.ndarray = np.expand_dims(prediction[1], axis=-1)
                ei: np.ndarray = np.expand_dims(prediction[2], axis=-1)
                
                node_importances.append(ni)
                edge_importances.append(ei)
                
                
            node_importances = np.concatenate(node_importances, axis=-1)
            edge_importances = np.concatenate(edge_importances, axis=-1)
            results.append((node_importances, edge_importances))
            
        return results
                
                
                
def load_model(model_path: str = DEFAULT_MODEL_PATH) -> ks.models.Model:
    """
    Loads the keras model from memory given its ``model_path`` absolute folder path. By default, this
    function will load the default model which is shipped with this package.

    :param model_path: The absolute path to the folder which contains the models persistent representation.

    :returns: The MEGAN model which is saved at the given folder path
    """
    scope = {
        **CUSTOM_OBJECTS,
        'Megan2': Megan2,
    }
    
    with ks.utils.custom_object_scope(scope):
        return ks.models.load_model(model_path)


def load_ensemble(model_paths: str = DEFAULT_ENSEMBLE_PATHS) -> t.Any:
    """
    Loads the keras
    """
    models = []
    for path in model_paths:
        model = _load_model(path)
        models.append(model)
        
    ensemble = ModelEnsemble(models)
    return ensemble