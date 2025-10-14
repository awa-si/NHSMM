import torch
import torch.nn as nn
import optuna
from types import SimpleNamespace
from typing import List, Dict, Any

# ----------------------------
# Synthetic OHLCV data generator
# ----------------------------
def generate_synthetic_ohlcv(n_seqs=10, seq_len=50):
    """
    Generates synthetic OHLCV sequences for HSMM testing.
    Returns a list of SimpleNamespace objects mimicking utils.Observations.
    """
    X_train = []
    for _ in range(n_seqs):
        # Open ~ [0,100], High = Open + [0,10], Low = Open - [0,10], Close ~ uniform(L,H), Volume ~ [100,1000]
        O = torch.rand(seq_len) * 100
        H = O + torch.rand(seq_len) * 10
        L = O - torch.rand(seq_len) * 10
        C = L + (H - L) * torch.rand(seq_len)
        V = 100 + torch.rand(seq_len) * 900
        log_probs = torch.stack([O,H,L,C,V], dim=1)  # Shape (seq_len, 5)
        obs = SimpleNamespace(log_probs=log_probs, lengths=seq_len)
        X_train.append(obs)
    return X_train

# ----------------------------
# Optuna objective
# ----------------------------
def objective_em(trial, X_train, HSMMClass, em_steps=2):
    n_states = trial.suggest_int("n_states", 2, 6)
    max_duration = trial.suggest_int("max_duration", 5, 30)
    alpha_pi = trial.suggest_float("alpha_pi", 0.1, 2.0)

    # Initialize HSMM
    model = HSMMClass(
        n_states=n_states,
        max_duration=max_duration,
        alpha=alpha_pi
    )

    device = next(model.parameters(), torch.tensor(0.0)).device
    model.sample_model_params(inplace=True)

    # --- EM steps ---
    for _ in range(em_steps):
        total_gamma, total_xi, total_eta = [], [], []
        for X_seq in X_train:
            gamma_vec, xi_vec, eta_vec = model._compute_posteriors(X_seq)
            total_gamma.extend(gamma_vec)
            total_xi.extend(xi_vec)
            total_eta.extend(eta_vec)

        # M-step updates
        if total_gamma:
            pi_new = torch.stack([g[0] for g in total_gamma]).mean(dim=0)
            pi_new = pi_new / pi_new.sum()
            model.pi.data.copy_(torch.log(pi_new + 1e-12))

        if total_xi:
            A_new = torch.stack([xi.sum(dim=0) for xi in total_xi]).mean(dim=0)
            A_new = A_new / A_new.sum(dim=1, keepdim=True)
            model.A.data.copy_(torch.log(A_new + 1e-12))

        if total_eta:
            D_new = torch.stack([eta.sum(dim=0) for eta in total_eta]).mean(dim=0)
            D_new = D_new / D_new.sum(dim=1, keepdim=True)
            model.D.data.copy_(torch.log(D_new + 1e-12))

    # --- Approximate log-likelihood ---
    log_likelihood = 0.0
    for X_seq in X_train:
        gamma_vec, _, _ = model._compute_posteriors(X_seq)
        seq_ll = sum([torch.log(g.sum() + 1e-12) for g in gamma_vec])
        log_likelihood += seq_ll.item()

    return log_likelihood

# ----------------------------
# Optuna tuner
# ----------------------------
def tune_hsmm_em(X_train: List, HSMMClass, n_trials=10, em_steps=2) -> Dict[str, Any]:
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective_em(trial, X_train, HSMMClass, em_steps), n_trials=n_trials)
    
    print("Best hyperparameters:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")

    return study.best_trial.params

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    from nhsmm.models import HSMM

    # Generate synthetic OHLCV data
    X_train = generate_synthetic_ohlcv(n_seqs=15, seq_len=50)

    # Run Optuna tuner
    best_params = tune_hsmm_em(X_train, HSMMClass=HSMM, n_trials=10, em_steps=2)
