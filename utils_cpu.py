import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics.cluster import pair_confusion_matrix
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler
from sklearn.preprocessing import MinMaxScaler as SklearnMinMaxScaler
from sklearn.preprocessing import QuantileTransformer as SklearnQuantileTransformer
from sklearn.preprocessing import RobustScaler as SklearnRobustScaler
from sklearn.preprocessing import PowerTransformer

from tqdm import tqdm
import pickle
import logging
import json
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list

from shapely.ops import voronoi_diagram

from sklearn.model_selection import ParameterGrid

import random

from catboost import CatBoostClassifier
import shap

from kmodes.kmodes import KModes

from warnings import simplefilter

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from shapely.geometry import MultiPoint
from shapely.ops import voronoi_diagram

from scipy.spatial import Voronoi
from shapely.geometry import Polygon, MultiPoint

import faiss

cols_dict = {
    'gender_type': 'Пол (доля мужчин)',
    'marital_status_2': 'В браке (доля)',
    'age': 'Возраст (лет в среднем)',
    'salary': 'Зарплата (драм в среднем)',
    'WorkExperience': 'Стаж работы (месяцев, в среднем)',
    'salary_nonzero': 'Есть данные о зарплате',
    'life_time': 'Срок жизни в банке (месяцев в среднем)',
    'ch_types_cnt': 'Число каналов дбо',
    'ch_smsbank_flag': 'Использование смс-банка',
    'ch_ibank_flag': 'Использование интернет-банка',
    'ch_mobapp_flag': 'Использование MyAmeria',
    'unsecured_loan_flag': 'Есть потребительский кредит',
    'crd_overdraft_loan_flag': 'Есть овердрафт на карте',
    'avg_total_income': 'Доход банка (драм в среднем)',
    'Demand Deposits': 'Доход банка от несрочных депозитов (драм в среднем)',
    'Commissions from Cards': 'Доход банка от карточных комиссий (драм в среднем)',
    'Other Commissions': 'Доход банка от иных комиссий (драм в среднем)',
    'Commission from transfers': 'Доход банка от переводов (драм в среднем)',
    'FX Income': 'Доход FX (драм в среднем)',
    'Consumer Finance': 'Доход банка от товарных кредитов (драм в среднем)',
    'Personal unsecured loans': 'Доход банка от потреб кредитов (драм в среднем)',
    'Plastic Cards': 'Доход банка от овердрафтов (драм в среднем)',
    'Mortgage Loans': 'Доход банка от ипотеки (драм в среднем)',
    'Time Deposits': 'Доход банка от срочных депозитов (драм в среднем)',
    'Cash-non cash commission': 'Доход банка от cash-non cash (драм в среднем)',
    'bank_income_comissions': 'Доход банка комиссионный (драм в среднем)',
    'dbt_savings_flag': 'Есть накопительный счёт',
    'dbt_deposit_flag': 'Есть депозит',
    'crd_unsecured_loan_flag': 'Есть потребительский кредит',
    'crd_consumer_loan_flag': 'Есть товарный кредит',
    'crd_mortgage_loan_flag': 'Есть ипотечный кредит',
    'dbt_types_cnt': 'Число дебетовых продуктов (в среднем)',
    'crd_types_cnt': 'Число кредитных продуктов (в среднем)',
    'prd_types_cnt': 'Число всех продуктов (в среднем)',
    'dbt_paycard_above2': 'Более двух платежных карт',
    'Balance_avg': 'Баланс (драм в среднем)',
    'Cr_turnover_avg': 'Поступления (драм в среднем)',
    'Average_Income': 'Доход клиента (драм в среднем)',
    'fico_nonzero': 'Есть данные Fico (БКИ) (доля)',
    'ActiveCardLoansCount': 'Число кредитных карт (БКИ) (в среднем)',
    'ActiveAllLoansCount': 'Число активных кредитов (БКИ) (в среднем)',
    'ActiveLoanKindCount': 'Число типов активных кредитов (БКИ) (в среднем)',
    'SwitchisClassQuantity': 'Число изменений статусов (БКИ) (в среднем)',
    'MaxLoanAmount': 'Максимальная выданная сумма кредита (БКИ) (в среднем)',
    'DelayedPaymentsForPaidCredits': 'Число пропущенных платежей по закрытым кредитам (БКИ) (в среднем)',
    'DelayedPaymentsForActiveCredits': 'Число пропущенных платежей по активным кредитам (БКИ) (в среднем)',
    'TheWorstClassLoan': 'Худший статус (БКИ) (в среднем)',
    'PaidLoanKindCount': 'Число типов закрытых кредитов (БКИ) (в среднем)',
    'PaidLoansCount': 'Число закрытых кредитов (БКИ) (в среднем)',
    'TotalActiveLoansBalance': 'Суммарный баланс активных кредитов (БКИ) (в среднем)',
    'RequestedBanksCount365': 'Число банков с запросами КИ за год (БКИ) (в среднем)',
    'RequestQuantity365': 'Число запросов КИ за год (БКИ) (в среднем)',
    }

cols_dict_similarity = {'gender_type': 'Пол',
 'marital_status_2': 'В браке',
 'age': 'Возраст',
 'salary': 'Зарплата',
 'WorkExperience': 'Стаж работы',
#  'salary_nonzero': 'Есть данные о зарплате',
 'life_time': 'Срок жизни в банке (мес)',
 'ch_types_cnt': 'Число каналов дбо',
 'ch_smsbank_flag': 'Использование смс-банка',
 'ch_ibank_flag': 'Использование интернет-банка',
 'ch_mobapp_flag': 'Использование MyAmeria',
#  'unsecured_loan_flag': 'Есть потребительский кредит',
 'crd_overdraft_loan_flag': 'Есть овердрафт на карте',
 'avg_total_income': 'Средний доход банка',
 'Demand Deposits': 'Доход банка от несрочных депозитов',
 'Commissions from Cards': 'Доход банка от карточных комиссий',
 'Other Commissions': 'Доход банка от иных комиссий',
 'Commission from transfers': 'Доход банка от переводов',
 'FX Income': 'Доход FX',
 'Consumer Finance': 'Доход банка от товарных кредитов',
 'Personal unsecured loans': 'Доход банка от потреб кредитов',
 'Plastic Cards': 'Доход банка от овердрафтов',
 'Mortgage Loans': 'Доход банка от ипотеки',
 'Time Deposits': 'Доход банка от срочных депозитов',
 'Cash-non cash commission': 'Доход банка от cash-non cash',
 'bank_income_comissions': 'Доход банка комиссионный',
 'dbt_savings_flag': 'Есть накопительный счёт',
 'dbt_deposit_flag': 'Есть депозит',
 'crd_unsecured_loan_flag': 'Есть потребительский кредит',
 'crd_consumer_loan_flag': 'Есть товарный кредит',
 'crd_mortgage_loan_flag': 'Есть ипотечный кредит',
 'dbt_types_cnt': 'Число дебетовых продуктов',
 'crd_types_cnt': 'Число кредитных продуктов',
 'prd_types_cnt': 'Число всех продуктов',
 'dbt_paycard_above2': 'Более двух платежных карт',
 'Balance_avg': 'Средний баланс',
 'Cr_turnover_avg': 'Оборот по кредитам',
 'Average_Income': 'Средний доход клиента',
 'fico_nonzero': 'Есть данные Fico (БКИ)',
 'ActiveCardLoansCount': 'Число кредитных карт (БКИ)',
 'ActiveAllLoansCount': 'Число активных кредитов (БКИ)',
 'ActiveLoanKindCount': 'Число типов активных кредитов (БКИ)',
 'SwitchisClassQuantity': 'Число изменений статусов (БКИ)',
 'MaxLoanAmount': 'Максимальная выданная сумма кредита (БКИ)',
#  'DelayedPaymentsForPaidCredits': 'Число пропущенных платежей по закрытым кредитам (БКИ)',
#  'DelayedPaymentsForActiveCredits': 'Число пропущенных платежей по активным кредитам (БКИ)',
 'TheWorstClassLoan': 'Худший статус (БКИ)',
 'PaidLoanKindCount': 'Число типов закрытых кредитов (БКИ)',
 'PaidLoansCount': 'Число закрытых кредитов (БКИ)',
 'TotalActiveLoansBalance': 'Суммарный баланс активных кредитов (БКИ)',
 'RequestedBanksCount365': 'Число банков с запросами КИ за год (БКИ)',
 'RequestQuantity365': 'Число запросов КИ за год (БКИ)'
}
    
def normalize_centres(col, centroids, data, pmin=10, pmax=90):
    p_min, p_max = np.percentile(data[col], [pmin, pmax])
    if p_min == p_max:
        p_min, p_max = data[col].min(), data[col].max()

    return (centroids - p_min)/(p_max - p_min)

def show_heatmap(df_pd, clusters_pd, scaler, cmap, cols_dict=cols_dict, cols2drop=[], reorder_features=True, center=0):
    
    data = df_pd.loc[df_pd.index, df_pd.columns].rename(columns=cols_dict)
    
    df_pd_scaled = pd.DataFrame(scaler.transform(df_pd), index=df_pd.index, columns=df_pd.columns)
    df_pd = df_pd.drop(columns=cols2drop, errors='ignore').rename(columns=cols_dict)
    df_pd_scaled = df_pd_scaled.drop(columns=cols2drop, errors='ignore').rename(columns=cols_dict)

    sns.set_theme(style="white")
    
    centers = data.groupby(clusters_pd).mean()
    
    centers_scaled = df_pd_scaled.groupby(clusters_pd).mean()
    cluster_counts = clusters_pd.value_counts(normalize=True).sort_index()
    centers_scaled_t = centers_scaled.T
    centers_t = centers.T
    
    corr_matrix = centers_scaled.corr()
    Z = linkage(corr_matrix, method='average')
    dendro = dendrogram(Z, no_plot=True)
    reordered_index = leaves_list(Z) if reorder_features else range(df_pd.shape[1])
    reorded_clusters = centers_scaled_t.mean().sort_values(ascending=False).index.tolist()
    reordered_centers = centers_scaled_t.iloc[reordered_index,:][reorded_clusters]
    # annot_data = centers.T.iloc[reordered_index, :][reorded_clusters]
    annot_data = centers.T.iloc[reordered_index, :][reorded_clusters].copy()
    annot_data = annot_data.apply(lambda col: col.map(lambda x: f'{x:,.0f}' if x >= 1e3 else f'{x:.1f}'))

    cluster_counts = cluster_counts[reorded_clusters].values
    
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=(1, 4), hspace=0)
    
    sns.set_color_codes("pastel")
    
    ax_histx = fig.add_subplot(gs[0, 0])
    ax_histx.bar(np.arange(len(cluster_counts)),
                    cluster_counts, color='lightgrey', alpha=1, width=1)
    ax_histx.bar_label(ax_histx.containers[0],
                        fontsize=10, family='Open Sans', fmt=lambda x: f'{int(round(x * 100))}%')
    ax_histx.set_xlim(-.5, len(cluster_counts) -.5)
    ax_histx.set_yticks([])
    ax_histx.set_xticks([])
    
    for spine in ax_histx.spines.values():
        spine.set_visible(False)
    
    ax_heatmap = fig.add_subplot(gs[1, 0])
    sns.heatmap(reordered_centers,
                cmap=cmap,
                center=center,
                annot=annot_data.values,
                yticklabels=reordered_centers.index,
                annot_kws={"size": 8, "fontfamily": "Open Sans"},
                fmt="",
                linewidths=.5,
                cbar=False,
                ax=ax_heatmap)
                
    plt.yticks(fontsize=8, rotation=0, family='Open Sans')
    plt.subplots_adjust(hspace=0)
    
    plt.show()
        
def get_centers(df, clusters):
    
    scaler = SklearnMinMaxScaler()
    data_scaled = scaler.fit_transform(df)

    df_scaled = pd.DataFrame(data_scaled, index=df.index, columns=df.columns)

    return df_scaled.groupby(clusters).mean()
    


def transform_labels(predicted_labels, ix):

    min_label = predicted_labels.min()
    shifted_labels = predicted_labels - min_label if min_label < 0 else predicted_labels
    cluster_volumes = np.bincount(shifted_labels)
    sorted_cluster_indices = np.argsort(-cluster_volumes)
    label_mapping = np.zeros_like(sorted_cluster_indices)

    for new_label, old_label in enumerate(sorted_cluster_indices):
        label_mapping[old_label] = new_label

    cluster_labels = label_mapping[shifted_labels]
    return pd.Series(cluster_labels, index=ix, name='cluster')

def get_shap_values_pd(data_dict, labels_series_dict, local=False):
    all_fi = pd.DataFrame(columns=['Feature', 'Importance', 'data', 'cluster'])
    data = pd.concat(data_dict.values(), axis=1, join='inner')
    data = data.loc[:, ~data.columns.duplicated()]    
    for name in labels_series_dict.keys():
        print(f"\nCLUSTERING {name}\n")
        df = data_dict[name]
        df = df if local else data.drop(columns=df.columns)
        df.columns = [str(col) for col in df.columns]
        labels = labels_series_dict[name].loc[df.index]
        
        for i_cluster in np.unique(labels):
            print(f"\nCLUSTER {i_cluster}\n")
            X_train = df
            y_train = (labels==i_cluster)*1
            model = CatBoostClassifier(iterations=200,
                                # task_type="GPU", devices='0',
                                silent=True)

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

def get_outliers_ix(data, features):
    outlier_indices = set()
    for feature in features:
        Q1 = data[feature].quantile(0.25)
        Q3 = data[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 5 * IQR
        upper_bound = Q3 + 5 * IQR
        outlier_mask = (data[feature] < lower_bound) | (data[feature] > upper_bound)
        outlier_indices.update(data[outlier_mask].index)

    return pd.Index(outlier_indices)
    
def get_feature_types(df):
    continuous_features = []
    discrete_features = []
    binary_features = []

    threshold_unique_discrete = 20

    for column in df.columns:
        unique_values = df[column].nunique()

        if unique_values == 2:
            binary_features.append(column)
        elif unique_values <= threshold_unique_discrete:
            discrete_features.append(column)
        else:
            continuous_features.append(column)

    feature_types = {
        'continuous': continuous_features,
        'discrete': discrete_features,
        'binary': binary_features
    }
    return feature_types

def power_transform_features(df, cols):

    ptyj = PowerTransformer(method='yeo-johnson')
    ptbc = PowerTransformer(method='box-cox')
    transformed_df = pd.DataFrame(index=df.index)

    for col in cols:
        if df[col].min() > 0:
            transformed_df[col] = ptbc.fit_transform(df[[col]])
        elif df[col].min() == 0:
            transformed_df[col] = ptbc.fit_transform(df[[col]]+1)
        else:
            transformed_df[col] = ptyj.fit_transform(df[[col]])

    return transformed_df

def log_features(df, cols):
    df_logged = pd.DataFrame(index=df.index)

    for col in cols:
        df_logged[col] = np.sign(df[col]) * np.log1p(np.abs(df[col]))

    return df_logged
    
def get_faiss_labels(df_labeled, df_unlabeled, cols_dict, labels_dict, name, k=5):
    scaler = SklearnStandardScaler()
    df = pd.concat([df_labeled, df_unlabeled]).loc[:, cols_dict[name]]
    df_scaled = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)
    _, d = df_scaled.shape
    xb = df_scaled.loc[df_labeled.index].values.astype('float32')
    xq = df_scaled.loc[df_unlabeled.index].values.astype('float32')
    col_means = np.nanmean(xq, axis=0)
    inds = np.where(np.isnan(xq))
    xq[inds] = np.take(col_means, inds[1])
    xq = xq.astype('float32')

    index = faiss.IndexFlatL2(d)
    index.add(xb)

    D, I = index.search(xq, k)

    indice_dict = dict(enumerate(df_labeled.index))

    labels = labels_dict[name].iloc[I.flatten()].values.reshape(I.shape)
    indices = df_labeled.iloc[I.flatten()].index.values.reshape(I.shape)

    return labels, indices
    

def voronoi_finite_polygons_2d(vor, radius=None):
    if vor.points.shape[1] != 2:
        raise ValueError("Только 2D поддерживается")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max() * 2

    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]
        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue

        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >=0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >=0 and v2 >=0:
                continue
            tangent = vor.points[p2] - vor.points[p1]
            tangent /= np.linalg.norm(tangent)
            normal = np.array([-tangent[1], tangent[0]])

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, normal)) * normal
            far_point = vor.vertices[v2] + direction * radius

            new_vertices.append(far_point.tolist())
            new_region.append(len(new_vertices) -1)

        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)
    
def plot_voronoi(centers: pd.DataFrame, sizes: pd.Series):
    cluster_labels = centers.index.values
    cluster_centers = centers.values

    vor = Voronoi(cluster_centers)

    regions, vertices = voronoi_finite_polygons_2d(vor)

    fig, ax = plt.subplots(figsize=(14, 8))

    sizes_normalized = (sizes - sizes.min()) / (sizes.max() - sizes.min())

    cmap = plt.cm.Blues

    min_x, min_y = cluster_centers.min(axis=0) - 0.1
    max_x, max_y = cluster_centers.max(axis=0) + 0.1

    for idx, region in enumerate(regions):
        polygon = vertices[region]
        poly = Polygon(polygon)
        cluster_id = cluster_labels[idx]
        size = sizes.loc[cluster_id]
        intensity = sizes_normalized.loc[cluster_id]
        color = cmap(intensity)
        color = list(color)
        color[-1] = 0.6
        ax.fill(*zip(*polygon), color=color, edgecolor='gray', linewidth=0.5)

    for i, (x, y) in enumerate(cluster_centers):
        ax.text(x, y, str(cluster_labels[i]),
                fontsize=10,
                ha='center',
                va='center',
                color='blue')

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_xticks([])
    ax.set_yticks([])


    plt.show()

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

feats = [
    'age',
    'gender_type',
    'marital_status_2',
    'WorkExperience',
    'life_time',
    'salary',
    'ch_mobapp_flag',
    'fico_nonzero',
    'RequestedBanksCount365',
    'ActiveAllLoansCount',
    'TotalActiveLoansBalance',
    'Balance_avg',
    'dbt_types_cnt',
    'crd_types_cnt',
    'prd_types_cnt',
    'Consumer Finance',
    'Personal unsecured loans',
    'Plastic Cards',
    'bank_income_comissions',
    'Time Deposits',
    'avg_total_income'
    ]
    
featcols = [
    'Возраст', 'Пол', 'В браке', 'Стаж работы', 'Срок жизни в банке (мес)',
    'Зарплата', 'Использование MyAmeria', 'Есть данные Fico (БКИ)',
    'Число банков с запросами КИ за год (БКИ)', 'Число активных кредитов (БКИ)',
    'Суммарный баланс активных кредитов (БКИ)', 'Средний баланс',
    'Азартные', 'Экономное потребление', 'Среднее потребление', 'Высокое потребление', 'Премиальное потребление',
    'Число дебетовых продуктов', 'Число кредитных продуктов',
    'Число всех продуктов', 'Доход банка от товарных кредитов',
    'Доход банка от потреб кредитов', 'Доход банка от овердрафтов',
    'Доход банка комиссионный', 'Доход банка от срочных депозитов', 'Средний доход банка'
    ]