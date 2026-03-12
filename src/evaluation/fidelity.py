import numpy as np
import pandas as pd
from collections import Counter
from scipy.stats import ks_2samp, wasserstein_distance
import math
from sklearn import linear_model
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from synthgauge.metrics.propensity import pmse_ratio
import matplotlib.pyplot as plt

"""
Compare to intial generation knobs
"""


"""
Marginal distribution
"""
#One issue here is that KS significance maybe affected by dataset size
def evaluate_numerical_features(samples: list, data: list):

    samples = np.array(samples, dtype=float)
    data = np.array(data, dtype=float)
    samples = samples[~np.isnan(samples)]
    data = data[~np.isnan(data)]
    
    #KS distance
    ks_statistic, ks_p_value = ks_2samp(samples, data)

    #Wasserstein distance
    w_distance = wasserstein_distance(samples, data)

    return ks_statistic, w_distance

def evaluate_categorical_features(categories, samples, data):
    samples = np.array(samples)
    data = np.array(data)

    synthetic = []
    real = []
    count_obs = Counter(samples)
    count_exp = Counter(data)
    for cat in categories:
        synthetic.append(count_obs[cat])
        real.append(count_exp[cat])
    
    synthetic_sum = sum(synthetic)
    real_sum = sum(real)
    synthetic = [(s / synthetic_sum) for s in synthetic]
    synthetic = np.asarray(synthetic)
    real = [(r / real_sum) for r in real]
    real = np.asarray(real)

    tv = 0.5 * np.abs(synthetic - real).sum() #Calculate the total variaiton distance
    return tv

"""
Joint Distribution
"""

def evaluate_pMSE(samples: pd.DataFrame, data: pd.DataFrame):
    combined = pd.concat([samples, data])
    labels = np.concatenate([np.zeros(len(samples)), np.ones(len(data))])
    pmse_ratio(combined, indicator=labels, method="logr")

    # input_samples = np.concatenate([samples, data])
    # labels = np.concatenate([np.zeros(len(samples)), np.ones(len(data))])
    # indices = np.random.permutation(len(labels))
    # input_samples = input_samples[indices]
    # labels = labels[indices]

    # logistic_regressor = linear_model.LogisticRegression()
    # logistic_regressor.fit(samples, labels)

"""
Support coverage and diversity
"""
#figure out k values and n_components
def embed_data(synthetic, real, n_components=10):
    scaler = StandardScaler()
    real_scaled = scaler.fit_transform(real)
    synthetic_scaled = scaler.transform(synthetic)
    pca = PCA(n_components=n_components, random_state=0)
    real_embed = pca.fit_transform(real_scaled)
    synth_embed = pca.transform(synthetic_scaled)

    return real_embed, synth_embed

def get_radii(real_embed, k):
    nn_real = NearestNeighbors(n_neighbors=k+1).fit(real_embed)
    distances, _ = nn_real.kneighbors(real_embed)
    radii = distances[:, k]
    return radii

def alpha_precision(real_embed, synth_embed, radii):
    """
    Fraction of synthetic samples within the radii of real-data samples
    """
    nn_real = NearestNeighbors(n_neighbors=1).fit(real_embed)
    distances, indices = nn_real.kneighbors(synth_embed)
    neighbours_radii = radii[indices[:, 0]]

    inside = distances[:, 0] <= neighbours_radii
    return inside.mean()

def beta_recall(real_embed, synth_embed, radii):
    """
    Fraction of real data samples covered by synthetic samples
    """

    nn_synth = NearestNeighbors(n_neighbors=1).fit(synth_embed)
    distances, indices = nn_synth.kneighbors(real_embed)

    inside = distances[:, 0] <= radii #What
    return inside.mean()

def get_alpha_beta_metrics(real, synthetic, k_fraction=0.05, n_components=10):
    real_embed, synthetic_embed = embed_data(synthetic, real, n_components=n_components)
    real_radii = get_radii(real_embed, k=4)
    alpha = alpha_precision(real_embed, synthetic_embed, real_radii)
    beta = beta_recall(real_embed, synthetic_embed, real_radii)

    return alpha, beta










