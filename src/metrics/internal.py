import numpy as np
def calinski_harabasz_score_scratch(X, labels):
    n_samples = X.shape[0]
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    if n_clusters < 2:
        return 0.0
    
    global_center = np.mean(X, axis=0)
    
    # SS_B: Between-cluster Sum of Squares
    # SS_W: Within-cluster Sum of Squares
    ss_b = 0
    ss_w = 0
    
    for k in unique_labels:
        cluster_points = X[labels == k]
        n_k = cluster_points.shape[0]
        centroid = np.mean(cluster_points, axis=0)
        
        # SS_B : n_k * distance( cluster centroid, global centroid of dataset)^2
        #measure of how far cluster centroids are from global centroid, we want it to be high
        ss_b += n_k * np.sum((centroid - global_center)**2)
        
        # SS_W Compactness: sum of distance(points, centroid)^2
        #measures how scattered the data points are inside their assigned cluster, we want it to be low
        ss_w += np.sum((cluster_points - centroid)**2)
        
    if ss_w == 0:
        return np.inf
        
    # Formula: (SS_B / (k - 1)) / (SS_W / (N - k))
    #(N-k / k-1 ) penalize the score based on number of clusters
    #we want score to be high because it tells us clusters are well separated and compact
    score = (ss_b / (n_clusters - 1)) / (ss_w / (n_samples - n_clusters))
    
    return score



#Numerator is how scattered the points are within a cluster and called Intra-cluster distance w ana 3aizha tb2a low
#Denominator is how far the clusters are from each other Inter-cluster distance w ana 3aizha tb2a high
#3aizha el ratio yb2a low ==0
# لكل كلاستر هجيب المسافه مابينه و مابين 
# used to measure similarity between clusters
def davies_bouldin_score_scratch(X, labels):
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    if n_clusters < 2:
        return 0.0
    
    # 1. Calculate Centroids and Scatter (Average distance to centroid)
    centroids = []
    scatters = []
    
    for k in unique_labels:
        # 
        cluster_points = X[labels == k]
        centroid = np.mean(cluster_points, axis=0)
        centroids.append(centroid)
        
        # Scatter: Mean Euclidean distance of points to their centroid
        dist = np.sqrt(np.sum((cluster_points - centroid)**2, axis=1))
        scatters.append(np.mean(dist))
        
    centroids = np.array(centroids)
    scatters = np.array(scatters)
    
    # 2. Calculate Davies-Bouldin Score
    db_score = 0
    
    for i in range(n_clusters):
        max_ratio = -np.inf
        for j in range(n_clusters):
            if i == j:
                continue
            
            # Distance between centroids
            separation = np.sqrt(np.sum((centroids[i] - centroids[j])**2))
            
            if separation == 0:
                ratio = 0
            else:
                #rule= scatter_i(cluster 1) + scatter_j(cluster 2) / distance_between_centroids
                # scatters: The average distance of all points in cluster to the cluster centroid.

                ratio = (scatters[i] + scatters[j]) / separation
                
            max_ratio = max(max_ratio, ratio)
            
        db_score += max_ratio
        #if score is high that means clusters are close to each other and scattered within themselves
    return db_score / n_clusters

#هنا باخد نققطه و اقيس المسافه مابينها و بين كل النقط اللي معاها في نفس الكلاستر و اجيبلهم افريدج و هو ده a_i
#بقيس المسافه مابين النقطه دي و كل الكلاستر التانيه و باخد المينيمم و هو ده b_i
#بعدين باحسب s_i = (b_i - a_i) / max(a_i, b_i)
# between -1 and 1 , we want it to be high
def silhouette_score_scratch(X, labels):
    #Silhouette Score measures how well each data point fits within its cluster.
    n_samples = X.shape[0]
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    if n_clusters < 2 or n_clusters == n_samples:
        return 0.0
    
    # (x-y)^2 = x^2 + y^2 - 2xy
    #Eculidean distance matrix
    A = np.sum(X**2, axis=1).reshape(-1, 1)
    B = np.sum(X**2, axis=1).reshape(1, -1)
    dist_matrix = np.sqrt(np.maximum(A + B - 2 * np.dot(X, X.T), 0))
    
    s_values = np.zeros(n_samples)
    
    #find points in same cluster and calculate a_i 
    for i in range(n_samples):
        own_cluster = labels[i]
        
        mask_same = (labels == own_cluster)
        mask_same[i] = False # Exclude self
        
        if np.sum(mask_same) == 0:
            a_i = 0
        else:
            a_i = np.mean(dist_matrix[i][mask_same])
            
#we need a samll and b bigger
        # Calculate b_i: minimum mean distance to points in other clusters    
        b_i = np.inf
        
        for label in unique_labels:
            if label == own_cluster:
                continue
                
            mask_other = (labels == label)
            mean_dist_other = np.mean(dist_matrix[i][mask_other])
            b_i = min(b_i, mean_dist_other)
        
        max_ab = max(a_i, b_i)
        if max_ab == 0:
            s_values[i] = 0
        else:
            s_values[i] = (b_i - a_i) / max_ab
            
    return np.mean(s_values)



#calculates the sum of the squared distances between each data point and the centroid of its assigned cluster
#we want it to be low because it indicates that data points are close to their cluster centroids,which means compact clusters.
def calculate_wcss(X, labels, centroids):
    wcss = 0
    for i, point in enumerate(X):
        cluster_idx = labels[i]
        centroid = centroids[cluster_idx]
        wcss += np.sum((point - centroid)**2)
    return wcss




def calculate_gmm_metrics(X, log_likelihood, n_components, covariance_type):
#Log-likelihood alone overfits
#AIC (Akaike Information Criterion)->To fix the overfitting problem of log-likelihood.
    n_samples, n_features = X.shape
    
    # Weights parameters = k - 1
    # Means parameters = k * d
    n_params = (n_components - 1) + (n_components * n_features)
    
    if covariance_type == 'full':
        # k * d * (d+1) / 2
        n_cov_params = n_components * n_features * (n_features + 1) / 2
    elif covariance_type == 'diagonal':
        # k * d
        n_cov_params = n_components * n_features
    elif covariance_type == 'spherical':
        # k
        n_cov_params = n_components
    elif covariance_type == 'tied':
        # d * (d+1) / 2 (Shared across all clusters)
        n_cov_params = n_features * (n_features + 1) / 2
    else:
        raise ValueError(f"Unknown covariance type: {covariance_type}")
        
    n_params += n_cov_params
    
    # BIC = k * ln(n) - 2 * ln(L) ->BIC (Bayesian Information Criterion)
    #it penalizes model complexity more heavily than AIC, 
    # especially as the number of data points increases.
    #N: number of samples,k: number of parameters
    #Lower BIC = better model
    #Penalty increases with dataset size
    bic = n_params * np.log(n_samples) - 2 * log_likelihood
    

    # AIC = 2k - 2 * ln(L) ->we want it to be low
    #k->number of model parameters
    #Penalizes model complexity linearly
    aic = 2 * n_params - 2 * log_likelihood
    
    return {
        "Log-Likelihood": log_likelihood,
        "BIC": bic,
        "AIC": aic,
        "n_params": n_params
    }