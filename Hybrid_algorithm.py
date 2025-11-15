import pandas as pd
import numpy as np
from collections import Counter
class treeNode:
    def __init__(self, threshold=None, feature_index=None, value=None):
        self.threshold = threshold
        self.feature_index = feature_index
        self.value = value
        self.left = None
        self.right = None

    def is_leaf_Node(self):
        return self.value is not None
class XGBoostClassifier:
    def __init__(self, n_estimators=500, learning_rate=0.5, max_depth=6,
                 lamda=3.0, subsample_features=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.lamda = lamda
        self.subsample_features = subsample_features
        self.trees = []

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _g(self, y_true, y_pred):
        return self._sigmoid(y_pred) - y_true

    def _h(self, y_true, y_pred):
        sig = self._sigmoid(y_pred)
        return sig * (1 - sig)

    def _exact_greedysplit_vectorized(self, X_col, y_true, y_pred):
        """Sparse-aware, vectorized split finder."""
        g = self._g(y_true, y_pred)
        h = self._h(y_true, y_pred)

        # skip all-zero columns
        nonzero_mask = X_col != 0
        X_nz = X_col[nonzero_mask]
        g_nz = g[nonzero_mask]
        h_nz = h[nonzero_mask]

        if X_nz.size < 2:
            return -np.inf, None

        # Total sums
        G_total, H_total = np.sum(g), np.sum(h)

        # Zero-entry contribution
        G_zero = np.sum(g[~nonzero_mask])
        H_zero = np.sum(h[~nonzero_mask])

        # Sort once
        sorted_idx = np.argsort(X_nz)
        X_sorted = X_nz[sorted_idx]
        g_sorted = g_nz[sorted_idx]
        h_sorted = h_nz[sorted_idx]

        # Vectorized prefix sums (no explicit loop)
        G_L = G_zero + np.cumsum(g_sorted)
        H_L = H_zero + np.cumsum(h_sorted)
        G_R = G_total - G_L
        H_R = H_total - H_L

        # Gain for all splits at once
        gain = (G_L**2) / (H_L + self.lamda + 1e-6) + \
               (G_R**2) / (H_R + self.lamda + 1e-6) - \
               (G_total**2) / (H_total + self.lamda + 1e-6)

        best_idx = np.argmax(gain)
        best_gain = gain[best_idx]
        best_threshold = X_sorted[best_idx]

        return best_gain, best_threshold

    def _build_tree(self, X, y_true, y_pred, depth):
        n_samples, n_features = X.shape
        if (n_samples < 3) or (depth >= self.max_depth):
            G = np.sum(self._g(y_true, y_pred))
            H = np.sum(self._h(y_true, y_pred))
            leaf_value = -G / (H + self.lamda + 1e-6)
            return treeNode(value=leaf_value)

        # Random feature subsampling
        feature_indices = np.random.choice(
            n_features,
            int(max(1, self.subsample_features * n_features)),
            replace=False
        )

        best_gain, best_threshold, best_feature = -np.inf, None, None

        for feature_index in feature_indices:
            gain, threshold = self._exact_greedysplit_vectorized(X[:, feature_index], y_true, y_pred)
            if gain > best_gain:
                best_gain, best_threshold, best_feature = gain, threshold, feature_index

        if best_gain < 1e-6:
            G = np.sum(self._g(y_true, y_pred))
            H = np.sum(self._h(y_true, y_pred))
            leaf_value = -G / (H + self.lamda + 1e-6)
            return treeNode(value=leaf_value)

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = X[:, best_feature] > best_threshold

        left_subtree = self._build_tree(X[left_mask], y_true[left_mask], y_pred[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y_true[right_mask], y_pred[right_mask], depth + 1)

        node = treeNode(threshold=best_threshold, feature_index=best_feature)
        node.left = left_subtree
        node.right = right_subtree
        return node

    def _predict_tree(self, X, tree):
        y_pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            node = tree
            while not node.is_leaf_Node():
                if X[i, node.feature_index] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            y_pred[i] = node.value
        return y_pred

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        y_mean = np.mean(y)
        y_pred = np.full(y.shape, np.log(y_mean / (1 - y_mean + 1e-6)))

        for _ in range(self.n_estimators):
            tree = self._build_tree(X, y, y_pred, 0)
            self.trees.append(tree)
            update = self._predict_tree(X, tree)
            y_pred += self.learning_rate * update

    def predict(self, X):
        X = np.asarray(X)
        y_pred = np.zeros(X.shape[0])
        for tree in self.trees:
            y_pred += self.learning_rate * self._predict_tree(X, tree)
        y_pred = self._sigmoid(y_pred)
        return y_pred 

class ovr_variant:
    def __init__(self,n_estimators = 80,lamda = 3,learning_rate=0.3,max_depth=6,subsample_features=1/28):
        self.ovr_models = []
        self.n_estimators = n_estimators
        self.lamda = lamda
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample_features = subsample_features
        self.epsilon = None
        self.train = None
        self.target = None
    def fit(self,X,y):
        self.train = X
        self.target = y
        classes = np.unique(y)
        for c in classes:
            print(f"Training class {c} vs Rest")
            y_binary = (y == c).astype(int)
            model = XGBoostClassifier(n_estimators=self.n_estimators,
                                      lamda=self.lamda,learning_rate=self.learning_rate,
                                      max_depth=self.max_depth,subsample_features=self.subsample_features)
            model.fit(X,y_binary)
            self.ovr_models.append(model)
    def predict(self,X,epsilon=0.3):
        self.epsilon = epsilon
        n_samples = X.shape[0]
        n_classes = len(self.ovr_models)
        scores = np.zeros((n_samples,n_classes))
        for idx,model in enumerate(self.ovr_models):
            probs = model.predict(X)
            scores[:,idx] = probs

        predictions =self.refine_knn_ovr_fast(scores,X,self.train,self.target,k=5,epsilon=self.epsilon)
        return predictions


    def refine_knn_ovr_fast(self,probs,x_val,x_train, y_train, k=5, epsilon=0.1):
        
        prob_val = probs  

        x_val = np.asarray(x_val, dtype=float)
        x_train = np.asarray(x_train, dtype=float)
        y_train = np.asarray(y_train)

        N, C = prob_val.shape

        final_preds = np.empty(N, dtype=y_train.dtype)
        epsilon_dist = 1e-6

        max_prob = np.max(prob_val, axis=1, keepdims=True)

        candidate_mask_all = (max_prob - prob_val) <= epsilon
        
        candidate_counts = np.sum(candidate_mask_all, axis=1)

        no_refine_mask = candidate_counts == 1
        
        refine_mask = candidate_counts > 1
        
        final_preds[no_refine_mask] = np.argmax(candidate_mask_all[no_refine_mask], axis=1)
        
        x_val_refine = x_val[refine_mask]
        prob_val_refine = prob_val[refine_mask]
        indices_refine = np.where(refine_mask)[0]

        for j, (x_val_i, p_i) in enumerate(zip(x_val_refine, prob_val_refine)):
            
            candidates = np.where(candidate_mask_all[indices_refine[j]])[0]

            mask_sub = np.isin(y_train, candidates)
            X_sub = x_train[mask_sub]
            y_sub = y_train[mask_sub]
            
            if X_sub.shape[0] == 0:

                final_preds[indices_refine[j]] = np.argmax(p_i)
                continue

            dists_sq = np.sum((X_sub - x_val_i)**2, axis=1)
            
            k_eff = min(k, len(dists_sq))

            k_indices = np.argpartition(dists_sq, k_eff)[:k_eff]
    
            neighbor_dists_sq = dists_sq[k_indices]
            neighbor_labels = y_sub[k_indices]
            
            neighbor_dists = np.sqrt(neighbor_dists_sq)
            
            weights = 1.0 / (neighbor_dists + epsilon_dist)
            
            vote = Counter()
            for label, weight in zip(neighbor_labels, weights):
                vote[label] += weight
                
            if vote:
                best_label = max(vote.items(), key=lambda x: x[1])[0]
            else:

                best_label = np.argmax(p_i)
                
            final_preds[indices_refine[j]] = best_label

        return final_preds