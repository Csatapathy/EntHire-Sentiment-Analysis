from model import NLPModel
from build_Model import build



class Inference:
    def __init__(self):
        """
        Initialising the class by loading the pipeline
        """
        self.data={}
        self.model=NLPModel()
        self.load_model=build()
        self.model.pipeline=self.load_model.load_pipeline()

    def get_predictions(self):
        """
        Makes predictions from the data using the pipeline
        """
        result={}
        for key in self.data.keys():
            pred=self.model.predict(self.data[key])
            result["predictions"]=self.make_inference(pred)
        return result         

    def make_inference(self, arr):
            """
            returns the inferences made on the data:
            IF 0 -> returns "Negative"
            IF 1 -> returns "Positive"
            """
            res = []
            for val in arr:
                if val == 0:
                    res.append("Negative")
                else:
                    res.append("Positive")
            return res

    