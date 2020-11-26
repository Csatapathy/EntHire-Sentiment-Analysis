class Inference:
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