# import optuna
# import numpy as cp
# import cupy as cp
# from cuml.cluster import DBSCAN, KMeans, AgglomerativeClustering
# import hdbscan
# from cuml.metrics import silhouette_score
# import warnings
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# from kneed import KneeLocator
# from cuml.preprocessing import StandardScaler
# import logging
# import json
# import gc

# warnings.filterwarnings("ignore")

# class ClusteringOptimizer:
#     def __init__(self, X, category = '', n_clusters = (7,13), n_trials=100, verbose=False):
#         self.X = cp.asarray(X)
#         self.X_scaled = StandardScaler().fit_transform(self.X)
#         self.n_clusters = n_clusters
#         self.n_trials = n_trials
#         self.best_scores = {}
#         self.best_params = {}
#         self.best_labels = {}
#         self.cluster_distribusion = {}
#         self.best_wcss_scores = {}
#         self.optimal_k = {}
#         self.verbose = verbose
#         self.category = category
        
#         if not self.verbose:
#             optuna.logging.set_verbosity(optuna.logging.WARNING)

#     def save_model_parametrs(self):
        
#         data = {
#             'parametrs' : self.best_params,
#             'scores' : self.best_scores,
#         }
#         cluster_counts = {}
#         for x in self.best_labels:
#             # Count the number of data points in each cluster
#             label = self.best_labels[x]
#             label[label == -1] = self.best_labels[x].max() + 1
#             cluster_counts[x] = cp.bincount(label)
#         print(data)
#         print(cluster_counts)

#         # with open(f"data/data_{self.category}.json", "w") as json_file:
#         #     json.dump(data, json_file, indent = 4)
            

#     def plot_wcss_elbow(self, n_clusters_, wcss):
#         import matplotlib.pyplot as plt
#         import numpy as cp

#         kl = KneeLocator(n_clusters_, wcss, curve='convex', direction='decreasing')
#         optimal_k = kl.elbow

#         plt.figure(figsize=(8, 5))
#         plt.plot(n_clusters_, wcss, marker='o', linestyle='-', color='blue', label='WCSS')
#         plt.axvline(x=optimal_k, color='red', linestyle='--', label=f'Optimal K: {optimal_k}')
#         plt.title('Elbow Method for Optimal Number of Clusters')
#         plt.xlabel('Number of Clusters (K)')
#         plt.ylabel('Score')
#         plt.xticks(n_clusters_)
#         plt.legend()
#         plt.grid(True)
#         plt.show()

#         return optimal_k

        
#     def optimize_kmeans(self):
#         arr = dict()
#         wcss =[]

#         for n in tqdm(range(self.n_clusters[0], self.n_clusters[1]+1)):
#             def objective(trial):
#                 params = {
#                     # 'n_clusters': trial.suggest_int('n_clusters', 5, 20),
#                     'init': trial.suggest_categorical('init', ['k-means++', 'random']),
#                     'n_init': trial.suggest_int('n_init', 5, 15),
#                     'max_iter': trial.suggest_int('max_iter', 100, 500),
#                     'tol': trial.suggest_float('tol', 1e-5, 1e-3, log=True)
#                 }
                
#                 clusterer = KMeans(n_clusters = n, **params, random_state=42)
#                 clusterer.fit(self.X)
                
#                 if len(cp.unique(clusterer.labels_)) < 2:
#                     return -1
                    
#                 score =  float(clusterer.inertia_)
#                 return -score


#             study = optuna.create_study(direction='maximize')
#             study.optimize(objective, n_trials=self.n_trials)
            
#             arr[n] = (-study.best_value, study.best_params)
#             wcss.append(-study.best_value)


#         n_clusters_ = range(self.n_clusters[0], self.n_clusters[1]+1)

#         n = self.plot_wcss_elbow(n_clusters_, wcss)
#         self.best_wcss_scores['kmeans'] = arr[n][0]
#         self.best_params['kmeans'] = arr[n][1]
#         self.best_params['kmeans']['n_clusters'] = n

#         best_kmeans = KMeans(**self.best_params['kmeans'], random_state=42)
#         self.best_labels['kmeans'] = best_kmeans.fit_predict(self.X)

#         self.best_scores['kmeans'] =  float(silhouette_score(self.X, self.best_labels['kmeans']))

#         self.optimal_k['kmeans'] = n

#         label = cp.asnumpy(self.best_labels['kmeans'])
#         label[label == -1] = label.max() + 1
#         cluster_counts = np.bincount(label)


#         print(f"Optimal number of clusters: {self.optimal_k['kmeans']}")
#         print(f"Best parameters: {self.best_params['kmeans']}")
#         print(f"Best WCSS Inertia: {self.best_wcss_scores['kmeans']}")
#         print(f"Silhouette Score: {self.best_scores['kmeans']}")
#         print(f"Cluster counts: {cluster_counts}")

#         gc.collect()
#         cp.get_default_memory_pool().free_all_blocks()

    
#     def optimize_dbscan(self):
#         def objective(trial):
#             params = {
#                 'eps': trial.suggest_float('eps', 0.1, 5.0),
#                 'min_samples': trial.suggest_int('min_samples', 2, 20),
#                 'metric': trial.suggest_categorical('metric', ['euclidean', 'manhattan'])
#             }

#             clusterer = DBSCAN(**params)
#             labels = clusterer.fit_predict(self.X_scaled)

#             n_clusters_ = len(set(labels) - {-1})
#             if n_clusters_ < 2 or cp.all(labels == -1):
#                 return -1  
            
#             if n_clusters_ < self.n_clusters[0] or n_clusters_ > self.n_clusters[1]:
#                 return -1  

#             valid_labels = labels[labels != -1]
#             valid_points = self.X[labels != -1]
#             score = float(silhouette_score(valid_points, valid_labels, metric=params['metric']))
#             return score

#         study = optuna.create_study(direction='maximize')
#         study.optimize(objective, n_trials=self.n_trials)

#         # self.best_scores['dbscan'] = study.best_value
#         self.best_params['dbscan'] = study.best_params

#         best_dbscan = DBSCAN(**study.best_params)
#         labels = best_dbscan.fit_predict(self.X_scaled)
#         self.best_labels['dbscan'] = labels

#         # n_clusters_ = len(set(labels) - {-1})
#         n_clusters_ = len(set(labels))

#         label = cp.asnumpy(self.best_labels['dbscan'])
#         label[label == -1] = label.max() + 1
#         cluster_counts = np.bincount(label)

#         self.best_scores['dbscan'] = float(silhouette_score(self.X, labels))
#         self.optimal_k['dbscan'] = n_clusters_
#         # Print results
#         print(f"Best Silhouette Score: {self.best_scores['dbscan']}")
#         print(f"Best Parameters: {self.best_params['dbscan']}")
#         print(f"Number of Clusters (excluding noise): {self.optimal_k['dbscan']}")
#         print(f"Cluster counts: {cluster_counts}")

#         gc.collect()        
#         cp.get_default_memory_pool().free_all_blocks()
        
#     def optimize_hdbscan(self) -> None:
#         def objective(trial):
#             params = {
#                 'min_cluster_size': trial.suggest_int('min_cluster_size', 5, 15),
#                 'min_samples': trial.suggest_int('min_samples', 1, 10),
#                 'cluster_selection_epsilon': trial.suggest_float('cluster_selection_epsilon', 0.0, 1.0),
#                 'metric': trial.suggest_categorical('metric', ['euclidean', 'manhattan'])
#             }
            
#             clusterer = hdbscan.HDBSCAN(**params)
#             labels = clusterer.fit_predict(self.X_scaled)
            
#             if len(cp.unique(labels)) < 2 or cp.all(labels == -1):
#                 return -1
#             if len(cp.unique(labels)) < self.n_clusters[0] or len(cp.unique(labels)) > self.n_clusters[1]:
#                 return -1

                
#             score = float(silhouette_score(self.X, labels))
#             return score
            
#         study = optuna.create_study(direction='maximize')
#         study.optimize(objective, n_trials=self.n_trials)
        
#         self.best_scores['hdbscan'] = study.best_value
#         self.best_params['hdbscan'] = study.best_params
        
#         best_hdbscan = hdbscan.HDBSCAN(**study.best_params)
#         self.best_labels['hdbscan'] = best_hdbscan.fit_predict(self.X_scaled)

#         # n_clusters_ = len(set(self.best_labels['hdbscan']) - {-1})
#         n_clusters_ = len(set(self.best_labels['hdbscan']))
#         self.optimal_k['hdbscan'] = n_clusters_

#         label = cp.asnumpy(self.best_labels['hdbscan'])
#         label[label == -1] = label.max() + 1
#         cluster_counts = np.bincount(label)
        
#         print(f"Best Silhouette Score: {self.best_scores['hdbscan']}")
#         print(f"Best Parameters: {self.best_params['hdbscan']}")
#         print(f"Number of Clusters (excluding noise): {self.optimal_k['hdbscan']}")
#         print(f"Cluster counts: {cluster_counts}")

#         gc.collect()        
#         cp.get_default_memory_pool().free_all_blocks()

        
#     def optimize_agglomerative(self):
#         arr = dict()
#         silhouette_scores = []

#         for n in tqdm(range(self.n_clusters[0], self.n_clusters[1] + 1)):

#             def objective(trial):
#                 params = {
#                     'linkage': trial.suggest_categorical('linkage', ['ward', 'complete', 'average']),
#                 }

#                 if params['linkage'] == 'ward':
#                     params['metric'] = 'euclidean'
#                 else:
#                     params['metric'] = trial.suggest_categorical('metric', ['euclidean', 'cosine'])
#                 clusterer = AgglomerativeClustering(n_clusters=n,memory = None, **params)
#                 labels = clusterer.fit_predict(self.X)

#                 if len(cp.unique(labels)) < 2:
#                     return -1  

#                 score = float(silhouette_score(self.X, labels, metric=params['metric']))
#                 return score

#             # Optimize using Optuna
#             study = optuna.create_study(direction='maximize')
#             study.optimize(objective, n_trials=self.n_trials)

#             arr[n] = (study.best_value, study.best_params)
#             silhouette_scores.append(study.best_value)

#         n_clusters_ = range(self.n_clusters[0], self.n_clusters[1] + 1)
#         optimal_n = self.plot_wcss_elbow(n_clusters_, silhouette_scores)  # Replace WCSS with silhouette_scores
#         self.optimal_k['agglomerative'] = optimal_n

#         self.best_scores['agglomerative'] = arr[optimal_n][0]
#         self.best_params['agglomerative'] = arr[optimal_n][1]
#         self.best_params['agglomerative']['n_clusters'] = optimal_n

#         best_agg = AgglomerativeClustering(**self.best_params['agglomerative'])
#         self.best_labels['agglomerative'] = best_agg.fit_predict(self.X)

#         # Calculate and store the silhouette score for the optimal clustering
#         self.best_scores['agglomerative'] = float(silhouette_score(self.X, self.best_labels['agglomerative']))


#         label = cp.asnumpy(self.best_labels['agglomerative'])
#         label[label == -1] = label.max() + 1
#         cluster_counts = np.bincount(label)
        
#         # Print the results
#         print(f"Optimal number of clusters: {self.optimal_k['agglomerative']}")
#         print(f"Best parameters: {self.best_params['agglomerative']}")
#         print(f"Best Silhouette Score: {self.best_scores['agglomerative']}")
#         print(f"Cluster counts: {cluster_counts}")


#         gc.collect()
#         cp.get_default_memory_pool().free_all_blocks()

    
#     def optimize_all(self):
#         # try:
#         #     print("---------------------------------------------")
#         #     print("Optimizing K-means...")
#         #     self.optimize_kmeans()
#         # except:
#         #     pass

#         # try:
#         #     print("---------------------------------------------")
#         #     print("Optimizing DBSCAN...")
#         #     self.optimize_dbscan()
#         # except:
#         #     pass

#         # try:
#         #     print("---------------------------------------------")
#         #     print("Optimizing HDBSCAN...")
#         #     self.optimize_hdbscan()
#         # except:
#         #     pass
            
#         try:
#             print("---------------------------------------------")
#             print("Optimizing Agglomerative Clustering...")
#             self.optimize_agglomerative()
#         except:
#             pass

        
#         best_algorithm = max(self.best_scores.items(), key=lambda x: x[1])[0]
        
#         self.save_model_parametrs()
        
#         return (
#             best_algorithm,
#             self.best_params[best_algorithm],
#             self.best_scores[best_algorithm]
#         )
    
#     def get_best_labels(self, algorithm: str) -> cp.ndarray:
#         return self.best_labels.get(algorithm, None)

import torch
import optuna
import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
from kneed import KneeLocator
import logging
import json
import gc

class KMeans:
    def __init__(self, n_clusters, max_iter=300, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        
    def fit_predict(self, X):
        n_samples = X.shape[0]
        idx = torch.randperm(n_samples)[:self.n_clusters]
        self.centroids = X[idx].clone()
        
        prev_centroids = torch.zeros_like(self.centroids)
        for _ in range(self.max_iter):
            distances = torch.cdist(X, self.centroids)
            self.labels = torch.argmin(distances, dim=1)
            
            for k in range(self.n_clusters):
                mask = self.labels == k
                if mask.any():
                    self.centroids[k] = X[mask].mean(0)
                    
            if torch.abs(prev_centroids - self.centroids).max() < self.tol:
                break
                
            prev_centroids = self.centroids.clone()
            
        self.inertia_ = torch.sum(torch.min(torch.cdist(X, self.centroids), dim=1)[0])
        return self.labels.cpu().numpy()

class ClusteringOptimizer:
    def __init__(self, X, category='', n_clusters=(7,13), n_trials=100):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        self.X = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.X_scaled = (self.X - self.X.mean(0)) / self.X.std(0)
        self.n_clusters = n_clusters
        self.n_trials = n_trials
        self.best_scores = {}
        self.best_params = {}
        self.best_labels = {}
        self.best_wcss_scores = {}
        self.optimal_k = {}
        self.category = category

    def optimize_kmeans(self):
        arr = {}
        wcss = []

        for n in tqdm(range(self.n_clusters[0], self.n_clusters[1]+1)):
            def objective(trial):
                params = {
                    'max_iter': trial.suggest_int('max_iter', 100, 500),
                    'tol': trial.suggest_float('tol', 1e-5, 1e-3, log=True)
                }
                
                clusterer = KMeans(n_clusters=n, **params)
                labels = clusterer.fit_predict(self.X)
                
                if len(np.unique(labels)) < 2:
                    return -1
                    
                return -float(clusterer.inertia_.cpu())

            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=self.n_trials)
            
            arr[n] = (-study.best_value, study.best_params)
            wcss.append(-study.best_value)

        n = self.plot_wcss_elbow(range(self.n_clusters[0], self.n_clusters[1]+1), wcss)
        self.best_wcss_scores['kmeans'] = arr[n][0]
        self.best_params['kmeans'] = arr[n][1]
        self.best_params['kmeans']['n_clusters'] = n

        best_kmeans = KMeans(**self.best_params['kmeans'])
        self.best_labels['kmeans'] = best_kmeans.fit_predict(self.X)
        self.optimal_k['kmeans'] = n

        torch.cuda.empty_cache()
        gc.collect()

    def optimize_dbscan(self):
        X_np = self.X_scaled.cpu().numpy()
        
        def objective(trial):
            params = {
                'eps': trial.suggest_float('eps', 0.1, 5.0),
                'min_samples': trial.suggest_int('min_samples', 2, 20),
                'metric': trial.suggest_categorical('metric', ['euclidean', 'manhattan'])
            }

            clusterer = DBSCAN(**params)
            labels = clusterer.fit_predict(X_np)

            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters_ < 2 or np.all(labels == -1):
                return -1
            
            if n_clusters_ < self.n_clusters[0] or n_clusters_ > self.n_clusters[1]:
                return -1

            valid_mask = labels != -1
            return float(silhouette_score_torch(self.X_scaled[valid_mask], 
                                              torch.tensor(labels[valid_mask], device=self.device)))

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)

        self.best_params['dbscan'] = study.best_params
        best_dbscan = DBSCAN(**study.best_params)
        self.best_labels['dbscan'] = best_dbscan.fit_predict(X_np)
        self.optimal_k['dbscan'] = len(set(self.best_labels['dbscan']))

        torch.cuda.empty_cache()
        gc.collect()

    def optimize_agglomerative(self):
        X_np = self.X.cpu().numpy()
        
        arr = {}
        silhouette_scores = []

        for n in tqdm(range(self.n_clusters[0], self.n_clusters[1] + 1)):
            def objective(trial):
                params = {
                    'linkage': trial.suggest_categorical('linkage', ['ward', 'complete', 'average']),
                }
                params['metric'] = 'euclidean' if params['linkage'] == 'ward' else \
                                  trial.suggest_categorical('metric', ['euclidean', 'cosine'])

                clusterer = AgglomerativeClustering(n_clusters=n, **params)
                labels = clusterer.fit_predict(X_np)

                if len(np.unique(labels)) < 2:
                    return -1

                return float(silhouette_score_torch(self.X, torch.tensor(labels, device=self.device)))

            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=self.n_trials)
            
            arr[n] = (study.best_value, study.best_params)
            silhouette_scores.append(study.best_value)

        optimal_n = self.plot_wcss_elbow(range(self.n_clusters[0], self.n_clusters[1] + 1), silhouette_scores)
        self.optimal_k['agglomerative'] = optimal_n
        self.best_scores['agglomerative'] = arr[optimal_n][0]
        self.best_params['agglomerative'] = arr[optimal_n][1]
        self.best_params['agglomerative']['n_clusters'] = optimal_n

        best_agg = AgglomerativeClustering(**self.best_params['agglomerative'])
        self.best_labels['agglomerative'] = best_agg.fit_predict(X_np)

        torch.cuda.empty_cache()
        gc.collect()

    def plot_wcss_elbow(self, n_clusters_, wcss):
        kl = KneeLocator(n_clusters_, wcss, curve='convex', direction='decreasing')
        optimal_k = kl.elbow
        return optimal_k

    def optimize_all(self):
        for method in ['kmeans', 'dbscan', 'agglomerative']:
            try:
                print(f"\nOptimizing {method.upper()}...")
                getattr(self, f'optimize_{method}')()
            except Exception as e:
                print(f"{method.upper()} optimization failed: {e}")

        if self.best_scores:
            best_algorithm = max(self.best_scores.items(), key=lambda x: x[1])[0]
            return best_algorithm, self.best_params[best_algorithm], self.best_scores[best_algorithm]
        raise Exception("No clustering algorithms completed successfully")

def silhouette_score_torch(X, labels):
    unique_labels = torch.unique(labels)
    n_clusters = len(unique_labels)
    n_samples = len(X)
    
    if n_clusters <= 1:
        return -1
    
    a = torch.zeros(n_samples, device=X.device)
    b = torch.full((n_samples,), float('inf'), device=X.device)
    
    for label in unique_labels:
        mask = labels == label
        if not mask.any():
            continue
            
        cluster_points = X[mask]
        
        # Calculate mean intra-cluster distance (a)
        if mask.sum() > 1:
            distances = torch.cdist(X[mask].unsqueeze(0), cluster_points).squeeze(0)
            a[mask] = distances.sum(1) / (mask.sum() - 1)
        
        # Calculate mean nearest-cluster distance (b)
        for other_label in unique_labels:
            if label == other_label:
                continue
                
            other_mask = labels == other_label
            if not other_mask.any():
                continue
                
            mean_dist = torch.cdist(X[mask], X[other_mask]).mean(1)
            b[mask] = torch.minimum(b[mask], mean_dist)
    
    s = (b - a) / torch.maximum(a, b)
    return s[~torch.isnan(s)].mean()