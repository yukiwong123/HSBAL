import time
from sklearn.neighbors import BallTree
import numpy as np
from modAL.models import ActiveLearner, CommitteeRegressor
from xgboost import XGBRegressor
import query
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from functools import partial
def mre_error(result,pred):
    sum=0.0
    for i in range(len(result)):
        sum = sum + (abs(result[i]-pred[i])/result[i])
    return (sum/(len(result))*100)

def calculate_median(y):
    mid = np.median(y, axis=0).tolist()
    return mid

def hsbal(N_train_all, config_option, initial_idx, all_data, count, stop_point, mre_result):

    sampled_config_ids_copy = initial_idx.copy()
    result = [row[-1] for row in all_data]
    config_features = np.asarray(config_option)

    result = np.asarray(result)


    regressors = [
        partial(GradientBoostingRegressor, random_state=7),
        partial(DecisionTreeRegressor, random_state=7),
        partial(RandomForestRegressor, n_jobs=-1, random_state=7),
        partial(ExtraTreesRegressor, n_jobs=-1, random_state=7),
        partial(XGBRegressor, random_state=7),
    ]

    learner_list = [
        ActiveLearner(
            estimator=regressor(),
            X_training=config_features[initial_idx],
            y_training=result[initial_idx]
        )
        for regressor in regressors
    ]
    # initializing the Committee
    committee = CommitteeRegressor(
        learner_list =learner_list,
        query_strategy=query.max_std_sampling
    )
    models = [reg() for reg in regressors]

    idx = 0
    n_samples = config_features.shape[0]
    sample_size = max(1, n_samples//10 )
    sample_indices = np.random.choice(n_samples, size=sample_size, replace=False)
    approx_features = config_features[sample_indices]
    tree1 = BallTree(approx_features, metric='manhattan')
    dists1, _ = tree1.query(config_features, k=int(n_samples / 40))

    avg_dists1 = np.mean(dists1, axis=1)
    avg_dists1 = np.array(avg_dists1)
    norm_dists1 = (avg_dists1 - np.min(avg_dists1)) / (np.ptp(avg_dists1) + 1e-8)

    for i in range(stop_point):

        max_std_query_idx, std = committee.query(config_features)
        query_ids, query_instance = query.flexible_weight( std, sampled_config_ids_copy, config_features,norm_dists1)
        query_id = query_ids[0]

        sampled_config_ids_copy.append(query_id)

        query_result = result[query_id]
        committee.teach([config_features[query_id]], [query_result])

        if(i==N_train_all[idx]-len(initial_idx)-1):
            X_train = config_features[sampled_config_ids_copy]
            y_train = result[sampled_config_ids_copy]
            y_train = np.asarray(y_train, dtype=np.float64)

            test_id = np.setdiff1d(np.array(range(len(config_features))), sampled_config_ids_copy)
            x_test = config_features[test_id]
            y_test = result[test_id]
            print("index: " + str(idx))

            for model in models:
                model.fit(X_train, y_train)

            y_pred = [[] for _ in range(len(models))]
            for id, model in enumerate(models):
                y_pred[id] = model.predict(x_test)

            y_median = calculate_median(y_pred)

            mre_result[idx][count] = mre_error(y_test, y_median)
            print("mre_result:", mre_result[idx][count])
            idx=idx+1


    return sampled_config_ids_copy, mre_result