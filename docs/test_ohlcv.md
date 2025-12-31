## 🧪 State Occupancy & Duration/Transition Diagnostics

This test (scripts/run_test_ohlcv.py) demonstrates **how to inspect and validate the internal states of an NHSMM model** after training. It provides insights into the **initial distribution, transition probabilities, state durations, and inferred state occupancy**.

```bash
    git clone https://github.com/awa-si/NHSMM.git;
    cd NHSMM;
    pip install -e ".[dev]";

    python3 scripts/run_test_ohlcv.py;
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
[INFO] NHSMM - [Config] n_states=3, n_features=5, max_duration=30

=== Model Training ===

=== Run 1/3 ===
[Iter 000] LL=-2022.915283 Δ=nan
[INFO] NHSMM - [Init 01] Iter 001 | Score -1866.342529 | Δ 1.566e+02 | Δ% 7.740e-02
[Iter 001] LL=-1866.342529 Δ=1.566e+02
[INFO] NHSMM - [Init 01] Iter 002 | Score -1812.605225 | Δ 5.374e+01 | Δ% 2.879e-02
[Iter 002] LL=-1812.605225 Δ=5.374e+01
[INFO] NHSMM - [Init 01] Iter 003 | Score -1771.639160 | Δ 4.097e+01 | Δ% 2.260e-02
[Iter 003] LL=-1771.639160 Δ=4.097e+01
[INFO] NHSMM - [Init 01] Iter 004 | Score -1735.427490 | Δ 3.621e+01 | Δ% 2.044e-02
[Iter 004] LL=-1735.427490 Δ=3.621e+01

=== Run 2/3 ===
[Iter 000] LL=-1702.538208 Δ=nan
[INFO] NHSMM - [Init 02] Iter 001 | Score -1670.011597 | Δ 3.253e+01 | Δ% 1.910e-02
[Iter 001] LL=-1670.011597 Δ=3.253e+01
[INFO] NHSMM - [Init 02] Iter 002 | Score -1634.531372 | Δ 3.548e+01 | Δ% 2.125e-02
[Iter 002] LL=-1634.531372 Δ=3.548e+01
[INFO] NHSMM - [Init 02] Iter 003 | Score -1602.994629 | Δ 3.154e+01 | Δ% 1.929e-02
[Iter 003] LL=-1602.994629 Δ=3.154e+01
[INFO] NHSMM - [Init 02] Iter 004 | Score -1573.131104 | Δ 2.986e+01 | Δ% 1.863e-02
[Iter 004] LL=-1573.131104 Δ=2.986e+01

=== Run 3/3 ===
[Iter 000] LL=-1546.682129 Δ=nan
[INFO] NHSMM - [Init 03] Iter 001 | Score -1520.065674 | Δ 2.662e+01 | Δ% 1.721e-02
[Iter 001] LL=-1520.065674 Δ=2.662e+01
[INFO] NHSMM - [Init 03] Iter 002 | Score -1491.858887 | Δ 2.821e+01 | Δ% 1.856e-02
[Iter 002] LL=-1491.858887 Δ=2.821e+01
[INFO] NHSMM - [Init 03] Iter 003 | Score -1463.880981 | Δ 2.798e+01 | Δ% 1.875e-02
[Iter 003] LL=-1463.880981 Δ=2.798e+01
[INFO] NHSMM - [Init 03] Iter 004 | Score -1438.347900 | Δ 2.553e+01 | Δ% 1.744e-02
[Iter 004] LL=-1438.347900 Δ=2.553e+01

=== Decoding ===
[decode] algorithm=viterbi, batch_size=1
[Predict] Sequences: 1, max_len: 251

Best-permutation accuracy: 1.0000
Confusion matrix (permuted):
[[142   0   0]
 [  0  58   0]
 [  0   0  51]]
Mapping (model→true):
  model_1 (bull) → true_0 (range)
  model_2 (bear) → true_1 (bull)
  model_0 (range) → true_2 (bear)

Metrics:
 F1: 1.0000 | Precision: 1.0000 | Recall: 1.0000
 Log-likelihood: -1420.21 | EM time: 3.09s

=== Initial Distribution per State ===
  00 (range): 0.3092
  01 (bull): 0.3823
  02 (bear): 0.3085
  All initial rows sum to 1.00 ✅

=== Duration Distributions per State ===
  range  | mode=1, mean=12.28, total_prob=1.00
  bull   | mode=3, mean=12.76, total_prob=1.00
  bear   | mode=2, mean=11.89, total_prob=1.00

=== Transition Matrix (row = from, col = to) ===
  00 ( range)    0.3922   0.3102   0.2976
  01 (  bull)    0.3006   0.3928   0.3067
  02 (  bear)    0.2834   0.3564   0.3602
  All transition rows sum to 1.00 ✅

=== Inferred State Occupancies (251 frames) ===
  range : 51 frames (20.32%)
  bull  : 142 frames (56.57%)
  bear  : 58 frames (23.11%)

```
