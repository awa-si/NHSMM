## 🧪 State Occupancy & Duration/Transition Diagnostics

This test demonstrates **how to inspect and validate the internal states of an NHSMM model** after training. It provides insights into the **initial distribution, transition probabilities, state durations, and inferred state occupancy**.

```python
    python3 NHSMM/scripts/run_test_ohlcv.py
```

### Key Components

1. **Initial State Distribution**  
   - Computes the mean initial probability across batches and time steps.
   - Applies softmax to obtain a normalized probability distribution.
   - Prints per-state probabilities with sum check for normalization.

2. **Transition Matrix**  
   - Computes the mean transition logits across batches and time steps.
   - Applies softmax row-wise to get conditional probabilities of moving from one state to another.
   - Displays the matrix in a readable format with a warning if rows are not normalized.

3. **Duration Distributions**  
   - Extracts per-state discrete duration probabilities from the duration module.
   - Computes **mode** and **mean duration** for each state.
   - Provides a check of total probability mass for each state.

4. **Inferred State Occupancy**  
   - Uses the **Viterbi-decoded sequence** to count the number of frames spent in each hidden state.
   - Outputs both absolute counts and percentages of total frames for intuitive interpretation.

### Example Output

```text
user@XXX:~# sudo -u 'user' bash -c 'cd /opt/; pipenv run python NHSMM/scripts/run_test_ohlcv.py'
[INFO] nhsmm - No data file found — using synthetic data with label map: {0: 'range', 1: 'bull', 2: 'bear'}
[INFO] nhsmm - [Config] n_states=3, n_features=5, max_duration=35

=== EM Training ===

=== Run 1/3 ===
[Iter 000] LL=-2509.411377 Δ=nan
[INFO] nhsmm - [Init 01] Iter 001 | Score -2393.112793 | Δ 1.163e+02 | Δ% 4.634e-02
[Iter 001] LL=-2393.112793 Δ=1.163e+02
[INFO] nhsmm - [Init 01] Iter 002 | Score -2340.115234 | Δ 5.300e+01 | Δ% 2.215e-02
[Iter 002] LL=-2340.115234 Δ=5.300e+01

=== Run 2/3 ===
[Iter 000] LL=-2305.989502 Δ=nan
[INFO] nhsmm - [Init 02] Iter 001 | Score -2254.996338 | Δ 5.099e+01 | Δ% 2.211e-02
[Iter 001] LL=-2254.996338 Δ=5.099e+01
[INFO] nhsmm - [Init 02] Iter 002 | Score -2207.835449 | Δ 4.716e+01 | Δ% 2.091e-02
[Iter 002] LL=-2207.835449 Δ=4.716e+01

=== Run 3/3 ===
[Iter 000] LL=-2167.392822 Δ=nan
[INFO] nhsmm - [Init 03] Iter 001 | Score -2129.604004 | Δ 3.779e+01 | Δ% 1.744e-02
[Iter 001] LL=-2129.604004 Δ=3.779e+01
[INFO] nhsmm - [Init 03] Iter 002 | Score -2095.892578 | Δ 3.371e+01 | Δ% 1.583e-02
[Iter 002] LL=-2095.892578 Δ=3.371e+01

=== Decoding ===
[decode] algorithm=viterbi, batch_size=1
[Predict] Sequences: 1, max_len: 310, device: cpu

Best-permutation accuracy: 0.9903
Confusion matrix (permuted):
[[163   0   0]
 [  0  58   0]
 [  3   0  86]]
Mapping (model→true):
  model_2 (bear) → true_0 (range)
  model_0 (range) → true_1 (bull)
  model_1 (bull) → true_2 (bear)

Metrics:
 F1: 0.9912 | Precision: 0.9940 | Recall: 0.9888
 Log-likelihood: -2063.62 | EM time: 16.00s

=== Initial Distribution per State ===
  00 (range): 0.3313
  01 (bull): 0.3325
  02 (bear): 0.3363
  All initial rows sum to 1 ✅

=== Duration Distributions per State ===
  range  | mode=2, mean=13.45, total_prob=1.0000
  bull   | mode=3, mean=13.20, total_prob=1.0000
  bear   | mode=2, mean=13.30, total_prob=1.0000
  All durations rows sum to 1 ✅

=== Transition Matrix (row = from, col = to) ===
  00 ( range)    0.3081   0.3566   0.3353
  01 (  bull)    0.2933   0.3526   0.3541
  02 (  bear)    0.2989   0.3327   0.3685
  All transition rows sum to 1 ✅

=== Inferred State Occupancies ===
  range : 58 frames (18.71%)
  bull  : 86 frames (27.74%)
  bear  : 166 frames (53.55%)
  Total frames: 310
```