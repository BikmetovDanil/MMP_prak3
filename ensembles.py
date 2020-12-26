import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize_scalar


class RandomForestMSE:
    def __init__(self, n_estimators, max_depth=5, feature_subsample_size=None,
                 **trees_parameters):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.trees_parameters = trees_parameters
        
    def fit(self, X, y, X_val=None, y_val=None):
        K = self.feature_subsample_size
        if K is None:
            K = X.shape[1] // 3
        
        bootstrap = []
        bootstrap_index = []
        feat_sample = []
        out_of_bag = []
        out_of_bag_index = []
        
        for j in range(self.n_estimators):
            bootstrap_index += [np.random.randint(0, X.shape[0], X.shape[0])]
            feat_sample += [np.random.permutation(np.arange(X.shape[1]))[:K]]
            bootstrap += [X[bootstrap_index[j]][:, feat_sample[j]]]
            out_of_bag_index += [np.array(list(set(np.arange(X.shape[0])).difference(set(bootstrap_index[j]))))]
            out_of_bag += [X[out_of_bag_index[j]][:, feat_sample[j]]]
            
        self.feat_sample = feat_sample
        self.forest = []
        p = []
        loss = []
        for j in range(self.n_estimators):
            self.forest += [DecisionTreeRegressor(max_depth=self.max_depth)]
            self.forest[j].fit(bootstrap[j], y[bootstrap_index[j]])
            p += [self.forest[j].predict(X[:, self.feat_sample[j]])]
            ar_p = np.array(p)
            loss += [np.sqrt(mean_squared_error(y, ar_p.mean(axis = 0)))]
        return np.array(loss)

    def predict(self, X):    
        p = []
        for j in range(self.n_estimators):
            p += [self.forest[j].predict(X[:, self.feat_sample[j]])]
        p = np.array(p)
        return p.mean(axis=0)
    
    def predict_vect(self, X, start, step):
        vect = []
        p = []
        for j in range(self.n_estimators):
            p += [self.forest[j].predict(X[:, self.feat_sample[j]])]
            if j >= start and (j - start) % step == 0:
                a = np.array(p)
                vect += [a.mean(axis = 0)]
        return np.array(vect)


class GradientBoostingMSE:
    def __init__(self, n_estimators, learning_rate=0.1, max_depth=5, feature_subsample_size=None,
                 **trees_parameters):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.trees_parameters = trees_parameters
        
    def fit(self, X, y, X_val=None, y_val=None):
        self.X = X
        self.y = y
        
        K = self.feature_subsample_size
        if K is None:
            K = X.shape[1] // 3
        
        b = np.zeros_like(y) + np.mean(y)
        prediction = b.copy()        
        self.trees = []
        feat_sample = []
        loss = []

        for j in range(self.n_estimators):
            if j == 0:
                resid = y
            else:
                resid = y - prediction
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            feat_sample += [np.random.permutation(np.arange(X.shape[1]))[:K]]
            tree.fit(X[:, feat_sample[j]], resid)
            b = tree.predict(X[:, feat_sample[j]]).reshape((X.shape[0], 1))
            self.trees += [tree]
            prediction += self.learning_rate * b
            loss += [np.sqrt(mean_squared_error(y, prediction))] 
        self.feat_sample = feat_sample
        return np.array(loss)
        
    def predict(self, X):
        predictions = np.mean(self.y) + np.zeros(X.shape[0])
        for j in range(self.n_estimators):
            predictions += self.learning_rate * self.trees[j].predict(X[:, self.feat_sample[j]])
        return predictions

