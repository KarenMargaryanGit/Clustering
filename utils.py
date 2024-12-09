import pandas as pd
import numpy as np
from tqdm import tqdm
import random
from warnings import simplefilter

import optuna
from optuna.samplers import TPESampler, RandomSampler
from functools import partial

import pickle
import logging
import json
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler as SklearnStandardScaler
from sklearn.preprocessing import MinMaxScaler as SklearnMinMaxScaler

from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from sklearn.model_selection import ParameterGrid

import cupy as cp
import cudf

from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list

from cuml.decomposition import PCA
from cuml.manifold import TSNE
from cuml.cluster import HDBSCAN, DBSCAN, KMeans, AgglomerativeClustering
from cuml.preprocessing import StandardScaler, MinMaxScaler
from cuml.metrics.cluster.silhouette_score import cython_silhouette_score

from catboost import CatBoostClassifier
import shap

from kmodes.kmodes import KModes

from shapely.geometry import MultiPoint
from shapely.ops import voronoi_diagram


def drop_outliers(data, features):
    for feature in features:
        Q1 = data[feature].quantile(0.25)
        Q3 = data[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 5 * IQR
        upper_bound = Q3 + 5 * IQR
        data = data[(data[feature] >= lower_bound) & (data[feature] <= upper_bound)]

    return data
    
def drop_sparse_cols(data, mode_share_thresh=.95):
    sparse_cols = []
    for col in data.columns:
        if 'mortg' not in col.lower():
            mode_value = data[col].mode()[0]
            mode_share = data[col].value_counts(normalize=True, dropna=False).loc[mode_value]
            if mode_share > mode_share_thresh:
                sparse_cols.append(col)

    return data.drop(columns=sparse_cols), sparse_cols
    
def train_optuna(df,
                 sampler,
                 metric,
                 direction,
                 n_trials,
                 obj_func,
                 name,
                 major_label_share,
                 minor_label_cnt,
                 clusters_cnt_max,
                 clusters_cnt_min,
                 pca, 
                 drop_hdbscan,
                 timeout):
    
    study_name = f'study_{name}_{metric}'

    objective_partial = partial(obj_func,
                                df=df,
                                metric=metric,
                                name=name,
                                major_label_share=major_label_share,
                                minor_label_cnt=minor_label_cnt,
                                clusters_cnt_max=clusters_cnt_max,
                                clusters_cnt_min=clusters_cnt_min,
                                pca=pca,
                                drop_hdbscan=drop_hdbscan)

    storage = optuna.storages.InMemoryStorage()

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=direction,
        sampler=sampler
        )

    study.optimize(objective_partial,
                  n_trials=n_trials,
                  show_progress_bar=True,
                  gc_after_trial=True,
                #   n_jobs=1,
                  timeout=timeout)

    print(f"{study_name} Best trial:")
    best_trial = study.best_trial
    print("  Value: {}".format(best_trial.value))
    print("  Params: ")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))

    return study, best_trial

def objective(trial, df, metric, name, major_label_share, minor_label_cnt, clusters_cnt_max, clusters_cnt_min, pca, drop_hdbscan=False):

    suggest_pca = [False, .7, .5, .3] if pca else [False]
    pca_threshold = trial.suggest_categorical('pca_threshold', suggest_pca)
    methods = ['KMeans'] if name in ['trans'] else ['HDBSCAN', 'KMeans', 'DBSCAN']
    method = trial.suggest_categorical('method', methods)
    max_minsamples = 600 if name in ['cred_hist', 'products'] else 1000
    max_minclustersize = 1500 if name in ['cred_hist', 'products'] else 2000
    

    if method == 'HDBSCAN':
        min_cluster_size = trial.suggest_int('min_cluster_size', 100, max_minclustersize, step=100)
        min_samples = trial.suggest_int('min_samples', 100, max_minsamples, step=100)
        cluster_selection_method = trial.suggest_categorical('cluster_selection_method', ['eom', 'leaf'])
        cluster_selection_epsilon = trial.suggest_float('cluster_selection_epsilon', 0, 1.0, step=0.1)
        X_transformed, cluster_labels = clustering(
            df=df,
            method=method,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method=cluster_selection_method,
            cluster_selection_epsilon=cluster_selection_epsilon,
            pca_threshold=pca_threshold
            )
    elif method == 'DBSCAN':
        eps = trial.suggest_float('eps', 0, 1, step=0.1)
        min_samples = trial.suggest_int('min_samples', 100, max_minsamples, step=100)
        X_transformed, cluster_labels = clustering(
            df=df,
            method=method,
            eps=eps,
            min_samples=min_samples,
            pca_threshold=pca_threshold
            )
    elif method == 'KMeans':
        n_clusters = trial.suggest_int('n_clusters', 4, 12)
        init = trial.suggest_categorical('init', ['scalable-k-means++', 'k-means++'])
        max_iter = trial.suggest_categorical('max_iter', [300])
        
        X_transformed, cluster_labels = clustering(
            df=df,
            method=method,
            n_clusters=n_clusters,
            init=init,
            max_iter=max_iter,
            pca_threshold=pca_threshold
            )

    labels_len = cp.unique(cluster_labels).size 

    if labels_len > 2 and clusters_cnt_min <= labels_len <= clusters_cnt_max:
        major_label_vc = cudf.Series(cluster_labels).value_counts(normalize=True).iloc[0]
        minor_label_vc = cudf.Series(cluster_labels).value_counts().iloc[-1]
        if major_label_vc < major_label_share and minor_label_vc > minor_label_cnt:
            try:
                if metric == 'silhouette':
                    return cython_silhouette_score(X_transformed, cluster_labels, chunksize=1e4)
                elif metric == 'davies_bouldin':
                    return davies_bouldin_score(X_transformed.get(), cluster_labels.get())
            except Exception as e:
                print(e)
                pass

    
    return -.5 if metric == 'silhouette' else 5.0

def clustering(df, **kwargs):
    X = StandardScaler().fit_transform(df.values)
    method = kwargs.pop('method', None)
    pca_threshold = kwargs.pop('pca_threshold', None)
    kwargs.update({'output_type': 'cupy'})

    if pca_threshold:
        n_components = choose_n_components(X, pca_threshold)
        pca = PCA(n_components=n_components)
        X = pca.fit_transform(X)
        # print('pca done')

    if method == 'HDBSCAN':
        model = HDBSCAN(**kwargs)
    elif method == 'DBSCAN':
        model = DBSCAN(**kwargs)
    elif method == 'KMeans':
        model = KMeans(**kwargs, random_state=11)
    elif method == 'AgglomerativeClustering':
        model = AgglomerativeClustering(**kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")
    

    predicted_labels = model.fit_predict(X, out_dtype='int64') if method == 'DBSCAN' else model.fit_predict(X)
    min_label = predicted_labels.min()
    shifted_labels = predicted_labels - min_label if min_label < 0 else predicted_labels
    cluster_volumes = cp.bincount(shifted_labels)
    sorted_cluster_indices = cp.argsort(-cluster_volumes)
    label_mapping = cp.zeros_like(sorted_cluster_indices)
    for new_label, old_label in enumerate(sorted_cluster_indices):
        label_mapping[old_label] = new_label
    
    cluster_labels = label_mapping[shifted_labels]
    
    return X, cluster_labels

def choose_n_components(X, var_threshold):
    pca = PCA(random_state=11, n_components=len(X[0])).fit(X)
    cum_var = np.cumsum(pca.explained_variance_ratio_)

    return np.argmax(cum_var >= var_threshold) + 1



def calc_metrics(X, cluster_labels, exclude_noise=False, silh_score=True):
    if exclude_noise:
        core_samples_mask = cluster_labels != -1
        X = X[core_samples_mask]
        cluster_labels = cluster_labels[core_samples_mask]

    davies_bouldin = davies_bouldin_score(X.get(), cluster_labels.get())
    calinski_harabasz = calinski_harabasz_score(X.get(), cluster_labels.get())
    silhouette = cython_silhouette_score(X, cluster_labels) if silh_score else None

    return davies_bouldin, calinski_harabasz, silhouette


def save_train_results(filename, cuda_results_tuple):
    
    with open (f"{filename}.pkl", 'wb') as file:
        pickle.dump(cuda_results_tuple, file)

    best_trials_dict_pd, df_dict_pd, labels_dict_pd, best_params_dict_pd, labels_series_dict_pd = cuda_results_tuple

    for key in labels_dict_pd:
        labels_dict_pd[key] = labels_dict_pd[key].get()
        df_dict_pd[key] = df_dict_pd[key].to_pandas()
        labels_series_dict_pd[key] = labels_series_dict_pd[key].to_pandas()

    pd_results_tuple = best_trials_dict_pd, df_dict_pd, labels_dict_pd, best_params_dict_pd, labels_series_dict_pd
    with open (f"{filename}_pd.pkl", 'wb') as file:
        pickle.dump(pd_results_tuple, file)

def train_and_save(filename, selected_feats, trials_dict, pca_dict, clusters_cnt_max_dict, clusters_cnt_min_dict, data, trans_ix, **params):
    best_trials_dict = {}
    df_dict = {}
    labels_dict = {}
    best_params_dict = {}
    labels_series_dict = {}

    metric = params.get('metric')

    for name, features in selected_feats.items():
        print('\n\nstart ', name, '\n')
        if name == 'trans':
            df = data[features].dropna().loc[trans_ix, :]
        else:
            df = data[features].dropna()
        print('df shape: ', df.shape)
        params.update({
            'n_trials': trials_dict.get(name, 50),
            'pca': pca_dict.get(name, False), 
            'clusters_cnt_max': clusters_cnt_max_dict.get(name, 12),
            'clusters_cnt_min': clusters_cnt_min_dict.get(name, 4)
        })
        study, best_trial = train_optuna(df=df,
                                            name=name,
                                            **params)

        if best_trial.values[0] > 0:
            best_trials_dict[name] = best_trial
            df_dict[name] = df
            best_params = best_trial.params
            X, labels = clustering(df, **best_params)
            labels_series = cudf.Series(labels, index=df.index)
            labels_series_dict[name] = labels_series
            labels_dict[name] = labels
            best_params_dict[name] = best_params

    cuda_results_tuple = best_trials_dict, df_dict, labels_dict, best_params_dict, labels_series_dict

    save_train_results(filename, cuda_results_tuple)

    return cuda_results_tuple
    
def get_shap_values(data_dict_pd, labels_series_dict_pd, local=False):
    all_fi = pd.DataFrame(columns=['Feature', 'Importance', 'data', 'cluster'])
    data = pd.concat(data_dict_pd.values(), axis=1, join='inner')
    data = data.loc[:, ~data.columns.duplicated()]    
    for name in labels_series_dict_pd.keys():
        print(f"\nCLUSTERING {name}\n")
        df = data_dict_pd[name]
        df = df if local else data.drop(columns=df.columns)
        df.columns = [str(col) for col in df.columns]
        labels = labels_series_dict_pd[name].loc[df.index]
        
        for i_cluster in np.unique(labels):
            print(f"\nCLUSTER {i_cluster}\n")
            X_train = df
            y_train = (labels==i_cluster)*1
            model = CatBoostClassifier(
                iterations=200,
                task_type="GPU",
                devices='0',
                silent=True
                )

            model.fit(X_train, y_train)
            explainer = shap.Explainer(model)
            shap_values = explainer(X_train)
            shap.plots.bar(shap_values)
            mean_shap_values = np.mean(np.abs(shap_values.values), axis=0)
            feature_importance = pd.DataFrame(list(zip(df.columns, mean_shap_values)), columns=['Feature', 'Importance'])
            feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
            feature_importance['data'] = name
            feature_importance['cluster'] = i_cluster
            all_fi = pd.concat([all_fi, feature_importance])

    return all_fi
    

mcc_groups = {
    'Супермаркеты': ['5411', '5499'],
    'Рестораны': ['5811', '5812', '5813', '5814', '5462'],
    'Одежда': ['5651', '5691'],
    'Аптеки': ['5912'],
    'Красота': ['5977', '7230'],
    'Финансы': ['6011', '6012', '6051', '6536', '4829'],
    'Такси': ['4121'],
    'Курьерская доставка': ['4215'],
    'Топливо и авто': ['5172', '5533', '5541', '5983'],
    'Розничные магазины': ['5200', '5311', '5331', '5399', '5999', '5039', '5722'],
    'Коммунальные услуги': ['4900'],
    'Азартные игры': ['7995']
}


col_rename_dict = {
    'lifr_time': 'life_time',
    'prConsumer Finance': 'crd_consumer_loan_cnt',
    'prFinancial leasing': 'crd_leasing_loan_cnt',
    'prGold Collatoralised Loans': 'crd_gold_loan_cnt',
    'prGuaranties': 'crd_guarantee_cnt',
    'prInvestment loans': 'crd_invest_loan_cnt',
    'prMortgage Loans': 'crd_mortgage_loan_cnt',
    'prPersonal secured loans': 'crd_secured_loan_cnt',
    'prSME Loans / Manual': 'crd_sme_loan_cnt',
    'prBond': 'dbt_bond_cnt',
    'prCard': 'dbt_paycard_cnt',
    'prDeposit box': 'dbt_depositbox_cnt',
    'prDeposit': 'dbt_deposit_cnt',
    'prMetal account': 'dbt_metal_cnt',
    'prOther Assets': 'dbt_other_cnt',
    'prSaving account': 'dbt_savings_cnt',
    'prInternet banking': 'ch_ibank_cnt',
    'prMy Ameria': 'ch_mobapp_cnt',
    'prPhone banking': 'ch_smsbank_cnt'
    }
    
selected_domain_features = {
    'behavioral': [
        'age',
        'ch_mobapp_flag',
        'ch_types_cnt',
        'life_time'
        ],
    'product': [
        'Balance_avg',
        'Consumer Finance',
        'Personal unsecured loans',
        'Plastic Cards',
        'Time Deposits',
        'avg_total_income',
        'bank_income_comissions',
        'crd_consumer_loan_flag',
        'crd_overdraft_loan_flag',
        'crd_types_cnt',
        'crd_unsecured_loan_flag',
        'dbt_savings_flag',
        'dbt_types_cnt',
        'prd_types_cnt'
        ],
     'finance': [
         'ActiveAllLoansCount',
         'ActiveCardLoansCount',
         'Balance_avg',
         'Cr_turnover_avg',
         'PaidLoanKindCount',
         'RequestedBanksCount365',
         'TotalActiveLoansBalance',
         'crd_types_cnt',
         'fico_nonzero'
         ],
     'personal': [
         'WorkExperience',
         'age',
         'gender_type',
         'marital_status_2',
         'salary'
         ], 
     'trans': [
         'Супермаркеты',
         'Рестораны',
         'Одежда',
         'Аптеки',
         'Красота',
         'Финансы',
         'Такси',
         'Курьерская доставка',
         'Топливо и авто',
         'Розничные магазины',
         'Коммунальные услуги',
         'Азартные игры'
         ]
}