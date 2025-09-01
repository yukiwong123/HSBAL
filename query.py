import time

import numpy as np

from sklearn.neighbors import BallTree

from sklearn.base import BaseEstimator

from modAL.utils.data import modALinput
from modAL.utils.selection import multi_argmax, shuffled_argmax

from joblib import Parallel, delayed
def max_std_sampling(regressor: BaseEstimator, X: modALinput,
                     n_instances: int = 1,  random_tie_break=False,
                     **predict_kwargs) -> np.ndarray:
    _, std = regressor.predict(X, return_std=True, **predict_kwargs)
    std = std.reshape(X.shape[0], )

    if not random_tie_break:
        return multi_argmax(std, n_instances=n_instances), std

    return shuffled_argmax(std, n_instances=n_instances), std


def query_batch(tree, X_batch):

    dists, _ = tree.query(X_batch, k=1)
    return dists.flatten()

def parallel_balltree_query(tree, A, batch_size=1000, n_jobs=-1):
    batches = [A[i:i+batch_size] for i in range(0, len(A), batch_size)]
    results = Parallel(n_jobs=n_jobs)(
        delayed(query_batch)(tree, batch) for batch in batches
    )
    return np.concatenate(results)
def flexible_weight(weight, add_idx, config_features, norm_dists1, query_instance=1):
    j=[]
    if query_instance == 1:

        tree = BallTree(config_features[add_idx], metric='manhattan')
        avg_dists = parallel_balltree_query(tree, config_features)
        avg_dists = np.array(avg_dists)

        tree1 = BallTree(np.delete(config_features, add_idx, axis=0), metric='manhattan')
        dists1, _ = tree1.query(config_features, k=int(len(np.delete(config_features, add_idx, axis=0))/4))
        avg_dists1 = np.mean(dists1, axis=1)
        avg_dists1 = np.array(avg_dists1)

        alpha = 0.4
        beta = 0.4

        weight = np.array(weight)
        add_idx = set(add_idx)

        norm_dists = (avg_dists - np.min(avg_dists)) / (np.ptp(avg_dists) + 1e-8)
        norm_dists1 = (avg_dists1 - np.min(avg_dists1)) / (np.ptp(avg_dists1) + 1e-8)
        norm_weight = (weight - np.min(weight)) / (np.ptp(weight) + 1e-8)


        score = alpha * norm_weight + beta * norm_dists - (1-alpha-beta) * norm_dists1

        candidates = [i for i in range(len(score)) if i not in add_idx]

        sorted_candidates = sorted(candidates, key=lambda x: score[x], reverse=True)
        j=sorted_candidates[0]
    else:
        pass

    return [j], query_instance
