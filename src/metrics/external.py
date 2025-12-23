import numpy as np

def confusion_matrix_scratch(y_true, y_pred):

    n_classes = len(np.unique(y_true))
    n_clusters = len(np.unique(y_pred))
    
    # This handles cases where labels might not be contiguous integers
    classes = np.unique(y_true)
    clusters = np.unique(y_pred)
    
    cm = np.zeros((n_classes, n_clusters), dtype=int)
    
    for i, true_label in enumerate(classes):
        for j, pred_label in enumerate(clusters):
            mask = (y_true == true_label) & (y_pred == pred_label)
            cm[i, j] = np.sum(mask)
            
    return cm



def purity_score_scratch(y_true, y_pred):
    #For each cluster, what is the dominant class
    cm = confusion_matrix_scratch(y_true, y_pred)
    #For each cluster (column), find the most frequent class (max row value)
    return np.sum(np.amax(cm, axis=0)) / np.sum(cm)




def entropy_scratch(labels):
  
    n_samples = len(labels)
    if n_samples == 0:
        return 0.0
    # Calculate class probabilities    
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / n_samples
    #entropy=-sum p(x)log(p(x))
    #we need it to be low
    return -np.sum(probs * np.log(probs + 1e-10)) 



def normalized_mutual_information_scratch(y_true, y_pred):

    n_samples = len(y_true)
    
    # 1. Entropy of Class (H(Y)) and Cluster (H(C))
    h_y = entropy_scratch(y_true)
    h_c = entropy_scratch(y_pred)
    
    if h_y == 0 or h_c == 0:
        return 0.0
    
    # 2. Mutual Information I(Y; C)
    # I(Y; C) = sum_y sum_c p(y,c) * log( p(y,c) / (p(y)*p(c)) )
    
    cm = confusion_matrix_scratch(y_true, y_pred)
    
    # Probabilities
    p_yc = cm / n_samples  # Joint probability
    p_y = np.sum(p_yc, axis=1) # Marginal prob of true classes
    p_c = np.sum(p_yc, axis=0) # Marginal prob of clusters
    
    mi = 0.0
    for i in range(cm.shape[0]): # Loop over classes
        for j in range(cm.shape[1]): # Loop over clusters
            if p_yc[i, j] > 0:
                mi += p_yc[i, j] * np.log(p_yc[i, j] / (p_y[i] * p_c[j] + 1e-10))
                
    # 3. Normalized MI
    return 2 * mi / (h_y + h_c)





def adjusted_rand_index_scratch(y_true, y_pred):

    n_samples = len(y_true)
    
    # Contingency matrix
    cm = confusion_matrix_scratch(y_true, y_pred)
    
    # Helper to calculate 'n choose 2' = n*(n-1)/2
    def n_choose_2(n):
        return n * (n - 1) / 2
    
    # Sum of choose_2 for the contingency matrix (Index term)
    sum_cm_choose2 = np.sum([n_choose_2(n) for n in cm.flatten()])
    
    # Row sums (a_i) and Column sums (b_j)
    a_sums = np.sum(cm, axis=1)
    b_sums = np.sum(cm, axis=0)
    
    sum_a_choose2 = np.sum([n_choose_2(n) for n in a_sums])
    sum_b_choose2 = np.sum([n_choose_2(n) for n in b_sums])
    
    total_choose2 = n_choose_2(n_samples)
    
    # Expected Index
    expected_index = (sum_a_choose2 * sum_b_choose2) / total_choose2
    
    # Max Index
    max_index = (sum_a_choose2 + sum_b_choose2) / 2
    
    # ARI Formula
    if max_index == expected_index:
        return 0.0
        
    ari = (sum_cm_choose2 - expected_index) / (max_index - expected_index)
    return ari