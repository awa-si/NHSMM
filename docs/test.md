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
=== Initial State Distribution ===
  00 (range): 0.3421
  01 (bull) : 0.4013
  02 (bear) : 0.2566
  Sum: 1.0000

=== Transition Matrix (row = from, col = to) ===
  00 (range)  0.8000 0.1500 0.0500
  01 (bull)   0.1000 0.8500 0.0500
  02 (bear)   0.0500 0.1000 0.8500

=== Duration Distributions per State ===
  range  | mode=2, mean=2.34, total_prob=1.0000
  bull   | mode=3, mean=3.12, total_prob=1.0000
  bear   | mode=1, mean=1.87, total_prob=1.0000

=== Inferred State Occupancies ===
  range : 142 frames (28.4%)
  bull  : 220 frames (44.0%)
  bear  : 138 frames (27.6%)
  Total frames: 500
