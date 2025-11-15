import pandas as pd
import numpy as np
from collections import Counter

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # Convert to float32 for speed
        X = np.asarray(X, dtype=np.float32)

        # Mean center
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # SVD decomposition (fastest for PCA)
        # X = U * S * Vt  --> principal axes = Vt[:n_components]
        _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # Keep only top components
        self.components = Vt[:self.n_components]

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        X_centered = X - self.mean
        return np.dot(X_centered, self.components.T)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class KNNClassifier:
    def __init__(self, k=5):
        self.k = k
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int32)

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        n_samples = X.shape[0]
        predictions = np.empty(n_samples, dtype=np.int32)

        for i in range(n_samples):
            # squared Euclidean distances (no sqrt needed for nearest)
            dists = np.sum((self.X - X[i]) ** 2, axis=1)

            # get indices of k smallest distances (faster than argsort)
            k_idx = np.argpartition(dists, self.k)[:self.k]

            # take votes from neighbors
            nearest_labels = self.y[k_idx]

            # majority vote
            neighbor_dists = np.sqrt(dists[k_idx])
            weights = 1.0 / (neighbor_dists + 1e-6)

            vote = Counter()
            for lbl, w in zip(nearest_labels, weights):
                vote[lbl] += w

            predictions[i] = max(vote.items(), key=lambda x: x[1])[0]


        return predictions
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
    def __init__(self,n_estimators = 40,lamda = 3,learning_rate=0.3,max_depth=6,subsample_features=1/28,knn_k=5):
        self.ovr_models = []
        self.n_estimators = n_estimators
        self.lamda = lamda
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample_features = subsample_features
        self.knn_k = knn_k
        self.train = None
        self.target = None
        self.epsilon = None

    def fit(self, X, y):
        self.train = X
        self.target = y

        classes = np.unique(y)
        for c in classes:
            print(f"Training class {c} vs Rest")
            y_binary = (y == c).astype(int)
            model = XGBoostClassifier(
                n_estimators=self.n_estimators,
                lamda=self.lamda,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                subsample_features=self.subsample_features
            )
            model.fit(X, y_binary)
            self.ovr_models.append(model)

        # -------- GLOBAL PCA FOR OPTIMIZED REFINEMENT --------
        print("Precomputing PCA for refinement...")
        self.pca_global = PCA(n_components=100)   # or 50 or 80 tuned
        self.train_pca = self.pca_global.fit_transform(X)

        print("Fit complete.")

    def predict(self, X, epsilon=0.25):
        self.epsilon = epsilon
        n_samples = X.shape[0]
        scores = np.zeros((n_samples,len(self.ovr_models)))

        for idx,model in enumerate(self.ovr_models):
            scores[:,idx] = model.predict(X)

        refined = self.refine_knn_ovr_fast(scores, X, self.train, self.target, k=self.knn_k, epsilon=self.epsilon)
        return refined


    def refine_knn_ovr_fast(self, probs, x_val, x_train, y_train, k=5, epsilon=0.25):
        prob_val = probs
        N, C = prob_val.shape

        x_val = np.asarray(x_val, dtype=float)
        y_train = np.asarray(y_train)

        final_preds = np.empty(N, dtype=y_train.dtype)
        epsilon_dist = 1e-6

        max_prob = np.max(prob_val, axis=1, keepdims=True)
        candidate_mask_all = (max_prob - prob_val) <= epsilon

        candidate_counts = np.sum(candidate_mask_all, axis=1)
        no_refine_mask = candidate_counts == 1
        refine_mask = candidate_counts > 1

        final_preds[no_refine_mask] = np.argmax(candidate_mask_all[no_refine_mask], axis=1)

        # ---- Only PCA transform once for val ----
        x_val_pca = self.pca_global.transform(x_val)

        idx_refine = np.where(refine_mask)[0]

        for idx in idx_refine:
            p_i = prob_val[idx]
            candidates = np.where(candidate_mask_all[idx])[0]

            mask_sub = np.isin(y_train, candidates)
            X_sub_pca = self.train_pca[mask_sub]
            y_sub = y_train[mask_sub]

            if X_sub_pca.shape[0] == 0:
                final_preds[idx] = np.argmax(p_i)
                continue

            dists_sq = np.sum((X_sub_pca - x_val_pca[idx])**2, axis=1)

            k_eff = min(k, len(dists_sq))
            k_idx = np.argpartition(dists_sq, k_eff)[:k_eff]

            neighbor_dists = np.sqrt(dists_sq[k_idx])
            neighbor_labels = y_sub[k_idx]
            weights = 1.0 / (neighbor_dists + epsilon_dist)

            vote = {}
            for lbl, w in zip(neighbor_labels, weights):
                vote[lbl] = vote.get(lbl,0)+w

            final_preds[idx] = max(vote.items(), key=lambda x: x[1])[0]

        return final_preds

