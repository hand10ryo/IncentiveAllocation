import numpy as np
import pandas as pd

class BaseBinaryTreatmentEvaluator:
    def __init__(self, policy, X_test, t_test, y_test, z_test, p_test = None):
        self.policy = policy
        self.X_test = X_test
        self.t_test = t_test
        self.y_test = y_test
        self.z_test = z_test
        self.p_test = p_test

    def score(self, budget: int = None):
        raise NotImplementedError("This method should be implemented in a subclass")

    def score_curve(self, budget_array: np.ndarray = None, budget_rate_array: np.ndarray = None, n_split: int = None):
        raise NotImplementedError("This method should be implemented in a subclass")
    

class AddonPerCost(BaseBinaryTreatmentEvaluator):
    def score(self, budget, actual_cost=True):
        if self.p_test is None:
            p_test = self.t_test.mean() * np.ones_like(self.t_test)
        else:
            p_test = self.p_test

        if actual_cost:
            cost = self.z_test * self.t_test / p_test
            t_opt = self.policy.optimize(self.X_test, budget, cost=cost)
        else:
            t_opt = self.policy.optimize(self.X_test, budget)

        y_tg_mean = self.y_test[(t_opt == 1) & (self.t_test == 1)].mean()
        y_cg_mean = self.y_test[(t_opt == 1) & (self.t_test == 0)].mean()
        addon = y_tg_mean - y_cg_mean 
        total_cost = self.z_test[(t_opt == 1).values].mean()

        if total_cost != 0:
            return addon / total_cost
        else:
            return 0
        
    def score_curve(
            self, 
            budget_array: np.ndarray = None, 
            budget_rate_array: np.ndarray = None, 
            n_split: int = None,
            actual_cost: bool = True,
        ):
        if self.p_test is None:
            p_test = self.t_test.mean() * np.ones_like(self.t_test)
        else:
            p_test = self.p_test

        pred_dict = self.policy.predict(self.X_test)

        if actual_cost:
            cost = self.z_test * self.t_test / p_test
        else:
            cost = pd.Series(pred_dict["cost"])

        sorted_index = self.policy.sort(self.X_test)
        cost_cumsum = cost.iloc[sorted_index].cumsum()

        if budget_array is None:
            if budget_rate_array is None:
                if n_split is None:
                    raise ValueError
                else:
                    budget_rate_array = np.arange(1, n_split+1) / n_split
                
            budget_array =  budget_rate_array * cost_cumsum.max() 


        addon_revenue_list = []
        for budget in budget_array:
            give_flg = cost_cumsum < budget
            tg_revenue = self.y_test.loc[give_flg & (self.t_test == 1)].mean()
            cg_revenue = self.y_test.loc[give_flg & (self.t_test == 0)].mean()
            tg_cost = self.z_test.loc[give_flg & (self.t_test == 1)].mean()
            addon_revenue = (tg_revenue - cg_revenue) / tg_cost
            addon_revenue_list.append(addon_revenue)

        return budget_array, budget_rate_array, np.array(addon_revenue_list)



class Addon(BaseBinaryTreatmentEvaluator):
    def score(self, budget, actual_cost=True):
        if actual_cost:
            t_opt = self.policy.optimize(self.X_test, budget, cost=self.z_test)
        else:
            t_opt = self.policy.optimize(self.X_test, budget)

        y_tg_mean = self.y_test[(t_opt == 1) & (self.t_test == 1)].mean()
        y_cg_mean = self.y_test[(t_opt == 1) & (self.t_test == 0)].mean()
        addon = y_tg_mean - y_cg_mean 
        return addon