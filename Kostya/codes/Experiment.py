"""
Reproduce experiment from node2vec article. section 4.3
"""

import os
import pickle
import warnings

from codes.Transformers import *
# from multiprocessing import Pool
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from codes.load_data import *

# print(theano.config)

# Quality measures. First is used by grid search.
qual_measures = {
    'f1-macro': lambda y_true, y_pred: metrics.f1_score(y_true, y_pred, average='macro'),
    'f1-micro': lambda y_true, y_pred: metrics.f1_score(y_true, y_pred, average='micro'),
    'accuracy': lambda y_true, y_pred: metrics.accuracy_score(y_true, y_pred),
}


def scorer(estimator, X, y):
    return qual_measures['accuracy'](y, estimator.predict(X))


def run_experiment(nx_G, Y, dim, data_name, model_name, load_dumped_res=True, verbose=1):
    """
    :param nx_G: 
    :param Y: 
    :param data_name: 
    :param model_name: one from node2vec, hist, opt, sdv, nmf
    :return: 
    """
    print("name: {}, method: {}, dim: {}, load_dumped_res={}".format(data_name, model_name, dim, load_dumped_res))

    res_templ = {
        'data name': data_name,
        'embedding model': model_name,
        'dim': dim,
    }

    best_params_filename = './dumps/best_params/{}_{}_d{}.dump'.format(data_name, model_name, dim)
    result_filename = './dumps/results/{}_{}_d{}.dump'.format(data_name, model_name, dim)

    # Parameters
    n_jobs = -1
    random_state = 42
    rand_splits_num = 10
    grid_search_data_part = 0.3
    rand_splits_test_size = 0.5
    repeat_qual_estim = 10

    params_grid = {
        'classifier__C': [
            # 1e-4,
            # 1e-3,
            # 1e-2,
            # 1e-1,
            1,
            # 10,
            # 1e2,
            # 1e3,
            # 1e4
        ],
    }

    # Prepare data
    m = Y.shape[1]
    X = Y.index.values[:, None]
    X_gs, X2, Y_gs, Y2 = \
        train_test_split(X, Y, train_size=grid_search_data_part, random_state=random_state)

    # Prepare pipeline
    model_name = model_name.lower()
    emb_transform = transformers[model_name](nx_G, data_name, dim, load_dumped_model=load_dumped_res)
    params_grid.update(emb_transform.cvparams)
    clf = LogisticRegression(n_jobs=n_jobs)
    ppl = Pipeline([('embedding', emb_transform), ('classifier', clf)])

    # Grid search for good p and q on grid_search_data_part fraction of data
    if load_dumped_res and os.path.exists(best_params_filename):
        best_params = pickle.load(open(best_params_filename, 'rb'))
    else:
        gscv = GridSearchCV(ppl, params_grid, n_jobs=n_jobs, verbose=1, scoring=scorer)
        best_params = []
        for i in range(m):  # go it for all target variables
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gscv.fit(X_gs, Y_gs[i])
            best_params.append(gscv.best_params_)
        pickle.dump(best_params, open(best_params_filename, 'wb'))

    # quality estimation
    results = []
    for j in range(repeat_qual_estim):
        Y2_train_predict = []
        Y2_test_predict = []
        Y2_train = []
        Y2_test = []
        for i, best_param in zip(range(m), best_params):
            y2 = Y2[i]
            ppl.set_params(**best_param)
            X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=rand_splits_test_size, random_state=random_state+j)
            ppl.fit(X2_train, y2_train)
            y_train2_predict = ppl.predict(X2_train)
            y_test2_predict = ppl.predict(X2_test)

            Y2_train_predict.append(y_train2_predict)
            Y2_test_predict.append(y_test2_predict)

            Y2_train.append(y2_train)
            Y2_test.append(y2_test)

        Y2_train_predict = np.array(Y2_train_predict)
        Y2_test_predict = np.array(Y2_test_predict)
        Y2_train = np.array(Y2_train)
        Y2_test = np.array(Y2_test)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if m > 1:
                train_qual = {name: qual_measures[name](Y2_train, Y2_train_predict) for name in qual_measures}
                test_qual = {name: qual_measures[name](Y2_test, Y2_test_predict) for name in qual_measures}
            else:
                train_qual = {name: qual_measures[name](Y2_train[0], Y2_train_predict[0]) for name in qual_measures}
                test_qual = {name: qual_measures[name](Y2_test[0], Y2_test_predict[0]) for name in qual_measures}

        for data_name in train_qual:
            results.append(dict(res_templ))
            results[-1]['scorer'] = data_name
            results[-1]['train|test'] = 'train'
            results[-1]['value'] = train_qual[data_name]

        for data_name in test_qual:
            results.append(dict(res_templ))
            results[-1]['scorer'] = data_name
            results[-1]['train|test'] = 'test'
            results[-1]['value'] = test_qual[data_name]

    save = {
        'results': pd.DataFrame.from_dict(results),
        'best_params': best_params,
    }
    pickle.dump(save, open(result_filename, 'wb'))

    return save

if __name__ == '__main__':
    # D = [32, 64, 128, 192, 256]
    D = [64]
    # methods = ['stoc_hist', 'node2vec', 'stoc_opt', 'stocsk_opt', 'bigclam', 'gamma', 'deepwalk', 'svd', 'nmf']
    methods = ['deepwalk']
    for dataset in get_hard_datasets():
        nx_G, Y, name = dataset
        for method in methods:
            for d in D:
                try:
                    run_experiment(nx_G, Y, d, name, method, load_dumped_res=True)
                except Exception as e:
                    print('!!!!')
                    print('errororororo')
                    print(e)
