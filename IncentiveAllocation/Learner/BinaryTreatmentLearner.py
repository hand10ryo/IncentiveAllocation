from copy import deepcopy
from sklearn.base import clone
import numpy as np

class BaseBinaryTreatmentLearner:
    def __init__(self, model):
        raise NotImplementedError("This method should be implemented in a subclass")

    def fit(self, X_train, y_train, t_train, p_train):
        raise NotImplementedError("This method should be implemented in a subclass")

    def predict(self, X_test):
        raise NotImplementedError("This method should be implemented in a subclass")
    
    @staticmethod
    def model_predict(model, X_test):
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X_test)[:, 1]
        else:
            return model.predict(X_test)


class TLearner(BaseBinaryTreatmentLearner):
    def __init__(self, revenue_model, cost_model):
        self.revenue_model = revenue_model
        self.cost_model = cost_model

    def fit(self, X_train, y_train, z_train, t_train, p_train=None):
        if p_train is None:
            p_train = t_train.mean() * np.ones_like(y_train)

        w_train = 1 / np.clip(p_train, 1e-6, None)
        self.tg_revenue_model = clone(self.revenue_model)
        self.cg_revenue_model = clone(self.revenue_model)
        self.tg_revenue_model.fit(
            X_train[t_train == 1], 
            y_train[t_train == 1],
            sample_weight = w_train[t_train == 1]
        )
        self.cg_revenue_model.fit(
            X_train[t_train == 0], 
            y_train[t_train == 0],
            sample_weight = w_train[t_train == 0]
        )
        self.cost_model.fit(
            X_train[t_train == 1], 
            z_train[t_train == 1],
            sample_weight = w_train[t_train == 1]
        )

    def predict(self, X_test):
        y_tg_pred = self.model_predict(self.tg_revenue_model, X_test)
        y_cg_pred = self.model_predict(self.cg_revenue_model, X_test)
        z_tg_pred = self.model_predict(self.cost_model, X_test)
        return {
            "tg_revenue": y_tg_pred, 
            "cg_revenue": y_cg_pred, 
            "addon_revenue": y_tg_pred - y_cg_pred,
            "cost": z_tg_pred,
            "score": (y_tg_pred - y_cg_pred) / np.clip(z_tg_pred, 1e-6, None)
        }


class SLearner(BaseBinaryTreatmentLearner):
    def __init__(self, revenue_model, cost_model):
        self.revenue_model = revenue_model
        self.cost_model = cost_model

    def fit(self, X_train, y_train, z_train, t_train, p_train=None):
        if p_train is None:
            p_train = t_train.mean() * np.ones_like(y_train)

        w_train = 1 / np.clip(p_train, 1e-6, None)
        self.revenue_model.fit(
            X_train.assign(t=t_train),
            y_train,
            sample_weight = w_train
        )
        self.cost_model.fit(
            X_train[t_train == 1], 
            z_train[t_train == 1],
            sample_weight = w_train[t_train == 1]
        )

    def predict(self, X_test):
        y_tg_pred = self.model_predict(self.revenue_model, X_test.assign(t=1))
        y_cg_pred = self.model_predict(self.revenue_model, X_test.assign(t=0))
        z_tg_pred = self.model_predict(self.cost_model, X_test)
        return {
            "tg_revenue": y_tg_pred, 
            "cg_revenue": y_cg_pred, 
            "addon_revenue": y_tg_pred - y_cg_pred,
            "cost": z_tg_pred,
            "score": (y_tg_pred - y_cg_pred) / np.clip(z_tg_pred, 1e-6, None)
        }

class TOTLearner(BaseBinaryTreatmentLearner):
    def __init__(self, revenue_model, cost_model):
        self.revenue_model = revenue_model
        self.cost_model = cost_model

    def fit(self, X_train, y_train, z_train, t_train, p_train=None):
        if p_train is None:
            p_train = t_train.mean() * np.ones_like(y_train)

        y_tot_train = y_train * (y_train / p_train - (1 - t_train) / (1 - p_train))
        self.revenue_model.fit(X_train, y_tot_train)
        self.cost_model.fit(
            X_train[t_train == 1], 
            z_train[t_train == 1],
            sample_weight = 1 / p_train[t_train == 1]
        )

    def predict(self, X_test):
        y_addon_pred = self.model_predict(self.revenue_model, X_test)
        z_tg_pred = self.model_predict(self.cost_model, X_test)
        return {
            "addon_revenue": y_addon_pred,
            "cost": z_tg_pred,
            "score": y_addon_pred / np.clip(z_tg_pred, 1e-6, None)
        }


