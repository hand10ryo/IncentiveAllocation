import pandas as pd
import numpy as np

class BaseBinaryTreatmentOptimizer:
    def __init__(self):
        raise NotImplementedError("This method should be implemented in a subclass")
    
    def run(self, pred_dict, budget):
        raise NotImplementedError("This method should be implemented in a subclass")
    

class GreedyOptimizer(BaseBinaryTreatmentOptimizer):
    def __init__(self):
        return

    def run(self, pred_dict, budget, cost = None):
        if cost is None:
            cost = pred_dict["cost"]

        pred_df = pd.DataFrame(pred_dict)
        pred_df.loc[:,"cost"] = cost
        pred_df = pred_df.sort_values("score", ascending=False)
        pred_df["cum_cost"] = pred_df["cost"].cumsum()
        pred_df["treatmet_opt"] = (pred_df["cum_cost"] <= budget).astype(int)
        pred_df = pred_df.sort_index()
        return pred_df["treatmet_opt"]
    
    def sort(self, pred_dict):
        pred_df = pd.DataFrame(pred_dict)
        pred_df = pred_df.sort_values("score")
        return pred_df.index.values

