import numpy as np
class Perceptron:
    def __int__(self,learning_rate = 0.01,n_iters = 1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = self._unit_step_func
        self.weight = None
        self.bias = None
    def fit(self,X,y):
        n_sample, n_feature = X.shape

        self.weight
        pass
    def predict(self,X):
        linear_output = np.dot(X,self.weight) + self.bias
        y_predict = self.activation_func(linear_output)
        return y_predict


    def _unit_step_func(self,x):
        return np.where(x>=0,1,0)