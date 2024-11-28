import optuna
import numpy as np
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
import hdbscan
from sklearn.metrics import silhouette_score
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.preprocessing import StandardScaler
import logging
import json
import gc

warnings.filterwarnings("ignore")

class ClusteringOptimizer:
    def __init__(self, X, category = '', n_clusters = (7,13), n_trials=100, verbose=False):
        self.X = X
        self.X_scaled = StandardScaler().fit_transform(self.X)
        self.n_clusters = n_clusters
        self.n_trials = n_trials
        self.best_scores = {}
        self.best_params = {}
        self.best_labels = {}
        self.cluster_distribusion = {}
        self.best_wcss_scores = {}
        self.optimal_k = {}
        self.verbose = verbose
        self.category = category
        
        if not self.verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

    def save_model_parametrs(self):
        
        data = {
            'parametrs' : self.best_params,
            'scores' : self.best_scores,
        }
        cluster_counts = {}
        for x in self.best_labels:
            # Count the number of data points in each cluster
            label = self.best_labels[x]
            label[label == -1] = self.best_labels[x].max() + 1
            cluster_counts[x] = np.bincount(label)
        print(data)
        print(cluster_counts)

        # with open(f"data/data_{self.category}.json", "w") as json_file:
        #     json.dump(data, json_file, indent = 4)
            

    def plot_wcss_elbow(self, n_clusters_, wcss):
        import matplotlib.pyplot as plt
        import numpy as np

        kl = KneeLocator(n_clusters_, wcss, curve='convex', direction='decreasing')
        optimal_k = kl.elbow

        plt.figure(figsize=(8, 5))
        plt.plot(n_clusters_, wcss, marker='o', linestyle='-', color='blue', label='WCSS')
        plt.axvline(x=optimal_k, color='red', linestyle='--', label=f'Optimal K: {optimal_k}')
        plt.title('Elbow Method for Optimal Number of Clusters')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Score')
        plt.xticks(n_clusters_)
        plt.legend()
        plt.grid(True)
        plt.show()

        return optimal_k

        
    def optimize_kmeans(self):
        arr = dict()
        wcss =[]

        for n in tqdm(range(self.n_clusters[0], self.n_clusters[1]+1)):
            def objective(trial):
                params = {
                    # 'n_clusters': trial.suggest_int('n_clusters', 5, 20),
                    'init': trial.suggest_categorical('init', ['k-means++', 'random']),
                    'n_init': trial.suggest_int('n_init', 5, 15),
                    'max_iter': trial.suggest_int('max_iter', 100, 500),
                    'tol': trial.suggest_float('tol', 1e-5, 1e-3, log=True)
                }
                
                clusterer = KMeans(n_clusters = n, **params, random_state=42)
                clusterer.fit(self.X)
                
                if len(np.unique(clusterer.labels_)) < 2:
                    return -1
                    
                score =  clusterer.inertia_
                return -score


            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=self.n_trials)
            
            arr[n] = (-study.best_value, study.best_params)
            wcss.append(-study.best_value)


        n_clusters_ = range(self.n_clusters[0], self.n_clusters[1]+1)

        n = self.plot_wcss_elbow(n_clusters_, wcss)
        self.best_wcss_scores['kmeans'] = arr[n][0]
        self.best_params['kmeans'] = arr[n][1]
        self.best_params['kmeans']['n_clusters'] = n

        best_kmeans = KMeans(**self.best_params['kmeans'], random_state=42)
        self.best_labels['kmeans'] = best_kmeans.fit_predict(self.X)

        self.best_scores['kmeans'] =  silhouette_score(self.X, self.best_labels['kmeans'])

        self.optimal_k['kmeans'] = n

        label = self.best_labels['kmeans']
        label[label == -1] = self.best_labels['kmeans'].max() + 1
        cluster_counts = np.bincount(label)

        print(f"Optimal number of clusters: {self.optimal_k['kmeans']}")
        print(f"Best parameters: {self.best_params['kmeans']}")
        print(f"Best WCSS Inertia: {self.best_wcss_scores['kmeans']}")
        print(f"Silhouette Score: {self.best_scores['kmeans']}")
        print(f"Cluster counts: {cluster_counts}")

        gc.collect()

    
    def optimize_dbscan(self):
        def objective(trial):
            params = {
                'eps': trial.suggest_float('eps', 0.1, 5.0),
                'min_samples': trial.suggest_int('min_samples', 2, 20),
                'metric': trial.suggest_categorical('metric', ['euclidean', 'manhattan'])
            }

            clusterer = DBSCAN(**params)
            labels = clusterer.fit_predict(self.X_scaled)

            n_clusters_ = len(set(labels) - {-1})
            if n_clusters_ < 2 or np.all(labels == -1):
                return -1  
            
            if n_clusters_ < self.n_clusters[0] or n_clusters_ > self.n_clusters[1]:
                return -1  

            valid_labels = labels[labels != -1]
            valid_points = self.X[labels != -1]
            score = silhouette_score(valid_points, valid_labels, metric=params['metric'])
            return score

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)

        # self.best_scores['dbscan'] = study.best_value
        self.best_params['dbscan'] = study.best_params

        best_dbscan = DBSCAN(**study.best_params)
        labels = best_dbscan.fit_predict(self.X_scaled)
        self.best_labels['dbscan'] = labels

        # n_clusters_ = len(set(labels) - {-1})
        n_clusters_ = len(set(labels))

        label = self.best_labels['dbscan']
        label[label == -1] = self.best_labels['dbscan'].max() + 1
        cluster_counts = np.bincount(label)

        self.best_scores['dbscan'] = silhouette_score(self.X, labels)
        self.optimal_k['dbscan'] = n_clusters_
        # Print results
        print(f"Best Silhouette Score: {self.best_scores['dbscan']}")
        print(f"Best Parameters: {self.best_params['dbscan']}")
        print(f"Number of Clusters (excluding noise): {self.optimal_k['dbscan']}")
        print(f"Cluster counts: {cluster_counts}")

        gc.collect()
        
    def optimize_hdbscan(self) -> None:
        def objective(trial):
            params = {
                'min_cluster_size': trial.suggest_int('min_cluster_size', 5, 15),
                'min_samples': trial.suggest_int('min_samples', 1, 10),
                'cluster_selection_epsilon': trial.suggest_float('cluster_selection_epsilon', 0.0, 1.0),
                'metric': trial.suggest_categorical('metric', ['euclidean', 'manhattan'])
            }
            
            clusterer = hdbscan.HDBSCAN(**params)
            labels = clusterer.fit_predict(self.X_scaled)
            
            if len(np.unique(labels)) < 2 or np.all(labels == -1):
                return -1
            if len(np.unique(labels)) < self.n_clusters[0] or len(np.unique(labels)) > self.n_clusters[1]:
                return -1

                
            score = silhouette_score(self.X, labels)
            return score
            
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        
        self.best_scores['hdbscan'] = study.best_value
        self.best_params['hdbscan'] = study.best_params
        
        best_hdbscan = hdbscan.HDBSCAN(**study.best_params)
        self.best_labels['hdbscan'] = best_hdbscan.fit_predict(self.X_scaled)

        # n_clusters_ = len(set(self.best_labels['hdbscan']) - {-1})
        n_clusters_ = len(set(self.best_labels['hdbscan']))
        self.optimal_k['hdbscan'] = n_clusters_

        label = self.best_labels['hdbscan']
        label[label == -1] = self.best_labels['hdbscan'].max() + 1
        cluster_counts = np.bincount(label)
        
        print(f"Best Silhouette Score: {self.best_scores['hdbscan']}")
        print(f"Best Parameters: {self.best_params['hdbscan']}")
        print(f"Number of Clusters (excluding noise): {self.optimal_k['hdbscan']}")
        print(f"Cluster counts: {cluster_counts}")

        gc.collect()

        
    def optimize_agglomerative(self):
        arr = dict()
        silhouette_scores = []

        for n in tqdm(range(self.n_clusters[0], self.n_clusters[1] + 1)):

            def objective(trial):
                params = {
                    'linkage': trial.suggest_categorical('linkage', ['ward', 'complete', 'average']),
                }

                if params['linkage'] == 'ward':
                    params['metric'] = 'euclidean'
                else:
                    params['metric'] = trial.suggest_categorical('metric', ['euclidean', 'cosine'])
                clusterer = AgglomerativeClustering(n_clusters=n,memory = None, **params)
                labels = clusterer.fit_predict(self.X)

                if len(np.unique(labels)) < 2:
                    return -1  

                score = silhouette_score(self.X, labels, metric=params['metric'])
                return score

            # Optimize using Optuna
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=self.n_trials)

            arr[n] = (study.best_value, study.best_params)
            silhouette_scores.append(study.best_value)

        n_clusters_ = range(self.n_clusters[0], self.n_clusters[1] + 1)
        optimal_n = self.plot_wcss_elbow(n_clusters_, silhouette_scores)  # Replace WCSS with silhouette_scores
        self.optimal_k['agglomerative'] = optimal_n

        self.best_scores['agglomerative'] = arr[optimal_n][0]
        self.best_params['agglomerative'] = arr[optimal_n][1]
        self.best_params['agglomerative']['n_clusters'] = optimal_n

        best_agg = AgglomerativeClustering(**self.best_params['agglomerative'])
        self.best_labels['agglomerative'] = best_agg.fit_predict(self.X)

        # Calculate and store the silhouette score for the optimal clustering
        self.best_scores['agglomerative'] = silhouette_score(
            self.X, self.best_labels['agglomerative']
        )

        label = self.best_labels['agglomerative']
        label[label == -1] = self.best_labels['agglomerative'].max() + 1
        cluster_counts = np.bincount(label)

        
        # Print the results
        print(f"Optimal number of clusters: {self.optimal_k['agglomerative']}")
        print(f"Best parameters: {self.best_params['agglomerative']}")
        print(f"Best Silhouette Score: {self.best_scores['agglomerative']}")
        print(f"Cluster counts: {cluster_counts}")


        gc.collect()

    
    def optimize_all(self):
        # try:
        #     print("---------------------------------------------")
        #     print("Optimizing K-means...")
        #     self.optimize_kmeans()
        # except:
        #     pass

        try:
            print("---------------------------------------------")
            print("Optimizing DBSCAN...")
            self.optimize_dbscan()
        except:
            pass

        # try:
        #     print("---------------------------------------------")
        #     print("Optimizing HDBSCAN...")
        #     self.optimize_hdbscan()
        # except:
        #     pass
            
        try:
            print("---------------------------------------------")
            print("Optimizing Agglomerative Clustering...")
            self.optimize_agglomerative()
        except:
            pass

        
        best_algorithm = max(self.best_scores.items(), key=lambda x: x[1])[0]
        
        self.save_model_parametrs()
        
        return (
            best_algorithm,
            self.best_params[best_algorithm],
            self.best_scores[best_algorithm]
        )
    
    def get_best_labels(self, algorithm: str) -> np.ndarray:
        return self.best_labels.get(algorithm, None)

