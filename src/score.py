import numpy as np

def compute_pbf_score(true_bias, predicted_bias):
    """
    Compute the political bias fidelity (PBF) score of a model 
    on the test set and the summaries. Bias should be labeled as
    -1 (liberal), 0 (center), 1 (conservative).
    :param true_bias: (N, ) the true bias of the source articles
    :param predicted_bias: (N, k) the predicted bias of k summaries of articles
    :return: the PBI score
    """
    if true_bias.shape[0] != predicted_bias.shape[0]:
        raise ValueError('true_bias and predicted_bias must have the same number of articles')
    
    k = predicted_bias.shape[1]
    squared_diff = (predicted_bias - true_bias[:, np.newaxis]) ** 2
    sum_all = np.sum(np.sum(squared_diff, axis=1) / k) / true_bias.shape[0]
    raw_score = np.sqrt(sum_all)
    
    upper = 2
    lower = 0
    scaled = (raw_score - lower) / (upper - lower)

    # reverse the score so that higher is better
    scaled = 1 - scaled
    return scaled

def compute_pbi_score(true_bias, predicted_bias):
    """
    Compute the political bias infidelity (PBI) score of a model 
    on the test set and the summaries. Bias should be labeled as
    -1 (liberal), 0 (center), 1 (conservative).
    :param true_bias: (N, ) the true bias of the source articles
    :param predicted_bias: (N, k) the predicted bias of k summaries of articles
    :return: the PBI score
    """
    if true_bias.shape[0] != predicted_bias.shape[0]:
        raise ValueError('true_bias and predicted_bias must have the same number of articles')
    
    k = predicted_bias.shape[1]
    all_bias = np.full((true_bias.shape[0], 3), [-1, 0, 1])

    other_bias = all_bias[np.nonzero(all_bias != true_bias[:, np.newaxis])].reshape(-1, 2)

    diff = predicted_bias[:, np.newaxis, :] - other_bias[:, :, np.newaxis] # (N, 2, k)
    diff = np.where(diff != 0, 1, 0) 
    print(diff)
    sum_all = np.sum(np.sum(diff, axis=(1, 2)) / (2 * k)) / true_bias.shape[0]

    upper = 1
    lower = 0.5
    scaled = (sum_all - lower) / (upper - lower) \
    
    # reverse the score so that higher is better
    scaled = 1 - scaled
    return scaled