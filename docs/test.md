## 🧪 State Occupancy & Duration/Transition Diagnostics

This test (scripts/run_test_ohlcv.py) demonstrates **how to inspect and validate the internal states of an NHSMM model** after training. It provides insights into the **initial distribution, transition probabilities, state durations, and inferred state occupancy**.

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

[INFO] NHSMM - No data file found — using synthetic data with label map: {0: 'range', 1: 'bull', 2: 'bear'}
[INFO] NHSMM - [Config] n_states=3, n_features=5, max_duration=35

=== EM Training ===

=== Run 1/3 ===
[Iter 000] LL=-2074.291504 Δ=nan
[INFO] NHSMM - [Init 01] Iter 001 | Score -1927.847656 | Δ 1.464e+02 | Δ% 7.060e-02
[Iter 001] LL=-1927.847656 Δ=1.464e+02
[INFO] NHSMM - [Init 01] Iter 002 | Score -1886.070679 | Δ 4.178e+01 | Δ% 2.167e-02
[Iter 002] LL=-1886.070679 Δ=4.178e+01
[INFO] NHSMM - [Init 01] Iter 003 | Score -1849.430664 | Δ 3.664e+01 | Δ% 1.943e-02
[Iter 003] LL=-1849.430664 Δ=3.664e+01
[INFO] NHSMM - [Init 01] Iter 004 | Score -1817.873169 | Δ 3.156e+01 | Δ% 1.706e-02
[Iter 004] LL=-1817.873169 Δ=3.156e+01

=== Run 2/3 ===
[Iter 000] LL=-1787.672729 Δ=nan
[INFO] NHSMM - [Init 02] Iter 001 | Score -1787.573486 | Δ 9.924e-02 | Δ% 5.552e-05
[Iter 001] LL=-1787.573486 Δ=9.924e-02
[INFO] NHSMM - [Init 02] Iter 002 | Score -1787.666504 | Δ -9.302e-02 | Δ% -5.204e-05
[Iter 002] LL=-1787.666504 Δ=-9.302e-02
[INFO] NHSMM - [Init 02] Iter 003 | Score -1787.620728 | Δ 4.578e-02 | Δ% 2.561e-05
[Iter 003] LL=-1787.620728 Δ=4.578e-02
[INFO] NHSMM - [Init 02] Iter 004 | Score -1787.655029 | Δ -3.430e-02 | Δ% -1.919e-05
[Iter 004] LL=-1787.655029 Δ=-3.430e-02

=== Run 3/3 ===
[Iter 000] LL=-1787.645264 Δ=nan
[INFO] NHSMM - [Init 03] Iter 001 | Score -1787.784546 | Δ -1.393e-01 | Δ% -7.791e-05
[Iter 001] LL=-1787.784546 Δ=-1.393e-01
[INFO] NHSMM - [Init 03] Iter 002 | Score -1787.718384 | Δ 6.616e-02 | Δ% 3.701e-05
[Iter 002] LL=-1787.718384 Δ=6.616e-02
[INFO] NHSMM - [Init 03] Iter 003 | Score -1787.722778 | Δ -4.395e-03 | Δ% -2.458e-06
[Iter 003] LL=-1787.722778 Δ=-4.395e-03
[INFO] NHSMM - [Init 03] Iter 004 | Score -1787.697021 | Δ 2.576e-02 | Δ% 1.441e-05
[Iter 004] LL=-1787.697021 Δ=2.576e-02

=== Decoding ===
[decode] algorithm=viterbi, batch_size=1
[Predict] Sequences: 1, max_len: 310, device: cpu

Best-permutation accuracy: 1.0000
Confusion matrix (permuted):
[[163   0   0]
 [  0  58   0]
 [  0   0  89]]
Mapping (model→true):
  model_2 (bear) → true_0 (range)
  model_1 (bull) → true_1 (bull)
  model_0 (range) → true_2 (bear)

Metrics:
 F1: 1.0000 | Precision: 1.0000 | Recall: 1.0000
 Log-likelihood: -1787.96 | EM time: 4.03s

=== Initial Distribution per State ===
  00 (range): 0.3308
  01 (bull): 0.3115
  02 (bear): 0.3577
  All initial rows sum to 1.00 ✅

=== Duration Distributions per State ===
  range  | mode=1, mean=13.20, total_prob=1.0000
  bull   | mode=6, mean=13.09, total_prob=1.0000
  bear   | mode=1, mean=13.67, total_prob=1.0000
  All durations rows sum to 1 ✅

=== Transition Matrix (row = from, col = to) ===
  00 ( range)    0.3793   0.3363   0.2844
  01 (  bull)    0.3287   0.3597   0.3116
  02 (  bear)    0.2879   0.3195   0.3927
  All transition rows sum to 1 ✅

=== Inferred State Occupancies (310 frames) ===
  range : 89 frames (28.71%)
  bull  : 58 frames (18.71%)
  bear  : 163 frames (52.58%)
```