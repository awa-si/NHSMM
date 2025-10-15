# tests/tune.py
import numpy as np
import torch
import optuna
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

from nhsmm.models.gaussian import GaussianHSMM


# -------------------------
# Best permutation accuracy
# -------------------------
def best_permutation_accuracy(true, pred, n_classes):
    C = confusion_matrix(true, pred, labels=list(range(n_classes)))
    row_ind, col_ind = linear_sum_assignment(-C)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    mapped_pred = np.array([mapping.get(p, p) for p in pred])
    acc = (mapped_pred == true).mean()
    return acc, mapped_pred


# -------------------------
# Synthetic Gaussian sequence
# -------------------------
def generate_gaussian_sequence(
    n_states=3,
    n_features=1,
    seg_len_range=(5, 20),
    n_segments_per_state=3,
    seed=0,
    noise_scale=0.05
):
    rng = np.random.default_rng(seed)
    states_list, X_list = [], []
    base_means = rng.uniform(-1.0, 1.0, size=(n_states, n_features))

    for s in range(n_states):
        for _ in range(n_segments_per_state):
            L = rng.integers(*seg_len_range)
            noise = rng.normal(scale=noise_scale, size=(L, n_features))
            segment_obs = base_means[s] + noise

            states_list.extend([s] * L)
            X_list.append(segment_obs)

    X = np.vstack(X_list)
    states = np.array(states_list)
    return states, torch.tensor(X, dtype=torch.float32)


# -------------------------
# Optuna objective
# -------------------------
def objective(trial):
    # hyperparameters
    n_states = trial.suggest_int("n_states", 2, 6)
    n_features = trial.suggest_int("n_features", 1, 3)
    max_duration = trial.suggest_int("max_duration", 10, 60)
    alpha = trial.suggest_float("alpha", 0.1, 5.0, log=True)
    n_init = trial.suggest_int("n_init", 1, 3)
    k_means_init = trial.suggest_categorical("k_means_init", [True, False])

    # added hyperparameters
    noise_scale = trial.suggest_float("noise_scale", 0.01, 0.2)
    n_segments_per_state = trial.suggest_int("n_segments_per_state", 1, 5)

    # generate synthetic sequence
    true_states, X = generate_gaussian_sequence(
        n_states=n_states,
        n_features=n_features,
        seg_len_range=(5, 20),
        n_segments_per_state=n_segments_per_state,
        seed=0,
        noise_scale=noise_scale
    )

    # initialize model
    model = GaussianHSMM(
        n_states=n_states,
        n_features=n_features,
        max_duration=max_duration,
        k_means=k_means_init,
        alpha=alpha,
        seed=0
    )

    # initialize emissions
    model.initialize_emissions(X, method="kmeans" if k_means_init else "moment")

    # fit EM
    model.fit(
        X,
        max_iter=50,
        n_init=n_init,
        sample_B_from_X=True,
        verbose=False,
        tol=1e-4
    )

    # decode
    pred = model.decode(X, algorithm="viterbi")

    # compute best-permutation accuracy
    acc, _ = best_permutation_accuracy(true_states, pred, n_classes=n_states)
    return acc


# -------------------------
# Run Optuna study
# -------------------------
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=40)

    print("\nBest trial:")
    print(f"  Accuracy: {study.best_value:.4f}")
    print(f"  Params: {study.best_params}")
