


class BinaryTreatmentPolicy:
    def __init__(self, learner, optimizer):
        self.learner = learner
        self.optimizer = optimizer
        self.pred_dict = None

    def fit(self, X_train, y_train, z_train, t_train, p_train = None):
        self.learner.fit(X_train, y_train, z_train, t_train, p_train)

    def predict(self, X_test):
        if self.pred_dict is None:
            self.pred_dict = self.learner.predict(X_test) 
        return self.pred_dict

    def optimize(self, X_test, Budget, cost = None):
        pred_dict = self.predict(X_test) 
        return self.optimizer.run(pred_dict, Budget, cost = cost)

    def sort(self, X_test):
        pred_dict = self.predict(X_test)
        return self.optimizer.sort(pred_dict)