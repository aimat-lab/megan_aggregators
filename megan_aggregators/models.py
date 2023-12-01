import typing as t


class PredictGraphsMixin:
    
    def predict_graphs():
        raise NotImplementedError()


class ModelEnsemble:
    
    def __init__(self,
                 models: t.List[PredictGraphsMixin],
                 ):
        self.models = models
        
    def predict_graphs(self, graphs: t.List[dict]) -> t.Tuple[list, list]:
        predictions_list: t.List[dict] = []
            
        for predictions in zip(*[model.predict_graph for model in self.models]):
            # "values" is a tuple with as many elements as there are models in the ensemble
            # every element of this tuple is the model output for one single 
            # graph of each corresponding model.
            
            arrays = []
            for prediction in predictions:
                pass
