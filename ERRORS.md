# RLSTC Codebase Forensic Audit: Paper vs. Implementation Discrepancies

> **Paper**: *"Sub-trajectory Clustering with Deep Reinforcement Learning"* — Liang et al., The VLDB Journal (2024) 33:685–702
>
> **Codebase**: `RLSTCcode-main/subtrajcluster/`
>
> **Audit Date**: April 2026

---

## Table of Contents

1. [ERROR-01: The "Medoid" Fallback — Cluster Centers Are Not Centroids](#error-01)
2. [ERROR-02: The `basesim` Static Denominator Bug — Validation Metric Decoupled from Fold Data](#error-02)
3. [ERROR-03: The `ODb` State Feature Is a Hardcoded Scalar Multiply, Not TRACLUS Output](#error-03)
4. [ERROR-04: Training Epoch Count Discrepancy — Code Does 1–2 Epochs, Not Convergence-Based Training](#error-04)
5. [ERROR-05: Soft Update Parameter `ω` Mismatch — 0.05 vs. Paper's 0.001](#error-05)
6. [ERROR-06: Epsilon Decay Is Per-Batch, Not Per-Step — Exploration Schedule Diverges from Paper](#error-06)
7. [ERROR-07: Missing N-fold Cross-Validation During Training — Single-Split Evaluation](#error-07)
8. [ERROR-08: SSE Computation Is Incorrect — Uses Wrong Normalization Denominator](#error-08)
9. [ERROR-09: `initcenters.py` Uses K-Means++ Max-Distance, Not True K-Means++ Probabilistic Seeding](#error-09)
10. [ERROR-10: Empty Cluster Handling Bug — Assigns to Wrong Cluster Index](#error-10)
11. [ERROR-11: Reward Signal Is Zero for Non-Segmentation Actions — Violates Paper's Reward Definition](#error-11)
12. [ERROR-12: `computecenter` Averaging Bug — Sums Raw Coordinates Instead of Per-Point Means](#error-12)
13. [ERROR-13: Missing Spatial Normalization — Only Time Is Z-Score Normalized](#error-13)
14. [ERROR-14: `Point.equal()` Returns `None` on Inequality — Silent Falsy Bug](#error-14)
15. [ERROR-15: Huber Loss Instead of MSE — Contradicts Paper's Stated Loss Function](#error-15)
16. [ERROR-16: Model Selection Uses First Alphabetical File, Not Best Validation Model](#error-16)

---

<a id="error-01"></a>
## ERROR-01: The "Medoid" Fallback — Cluster Centers Are Not Centroids

> [!CAUTION]
> **Severity: CRITICAL — Invalidates core paper claims (Figure 16, OD metric, Algorithm 3)**

### Paper Claim

**Section 4.4** (paper.md L789–846):
> *"To achieve this, we determine the average coordinate at each timestamp to generate a representative trajectory. We scan the timestamps in chronological order and record the number of trajectories containing each timestamp. If the number of trajectories at a given timestamp is no less than a predefined threshold MinNum, we compute the average coordinate for that timestamp. We derive synchronous points through linear interpolation for trajectories that lack a sampled point at that timestamp."*

The paper describes a mathematically rigorous algorithm: interpolate all cluster members to shared timestamps, then **average their spatial coordinates** to produce a true geometric centroid.

### Code Reality

**File**: [initcenters.py](file:///Users/wilsebbis/Developer/REAL_RLSTC/RLSTCcode/subtrajcluster/initcenters.py)

The `getbaseclus()` function (L20–55) does the following:
1. Initializes centers via `initialize_centers()` (a max-distance heuristic — see ERROR-09).
2. Assigns sub-trajectories to the nearest center.
3. **Stores the original center trajectory as `cluster_dict[i][1]`** — see L51: `cluster_dict[i].append(center)`.
4. **Never recomputes the center** as an average of the assigned sub-trajectories.

The "center" stored at index `[1]` of each cluster dictionary entry is literally just `centers[i]` — one of the original seed trajectories selected by the max-distance heuristic. It is **never replaced with a computed centroid**.

```python
# initcenters.py L46-54
for i in cluster_segments.keys():
    center = centers[i]               # <-- THE ORIGINAL SEED, NOT A CENTROID
    temp_dist = dists_dict[i]
    aver_dist = np.mean(temp_dist)
    cluster_dict[i].append(aver_dist)
    cluster_dict[i].append(center)    # <-- STORED AS THE "CENTER"
    cluster_dict[i].append(temp_dist)
    cluster_dict[i].append(cluster_segments[i])
```

### Why This Is Devastating

- **Figure 16 ("Case Study by Visualization")** in the paper shows "thick red lines representing representative trajectories, i.e., cluster centers." These are **not** statistically computed centers of mass — they are individual seed trajectories cherry-picked by a farthest-first traversal.
- The **OD metric** (Equation 1 in the paper) measures distance from sub-trajectories to their "cluster center." If the center is just a pre-existing trajectory rather than a learned centroid, the OD values are measuring distance to an arbitrary exemplar, not a geometric average.
- **Algorithm 3** (RLSTC algorithm, paper L802–835) states "Update the cluster centers of the k clusters" at Line 11. The implementation in `update_centers()` (`cluster.py` L248–255) does attempt a centroid computation but through `computecenter()`, which itself has a coordinate-averaging bug (see ERROR-12).

### The `computecenter` Partial Fix in `cluster.py`

There **is** a `computecenter()` function in `cluster.py` (L198–236) that attempts temporal averaging. However, this function is only called during `update_centers()` after each training epoch — **not** during the initial center computation in `initcenters.py`. The initial centers that seed the entire RL pipeline are never true centroids.

Furthermore, `computecenter()` itself has averaging bugs (see ERROR-12 below).

---

<a id="error-02"></a>
## ERROR-02: The `basesim` Static Denominator Bug — Validation Metric Decoupled from Fold Data

> [!CAUTION]
> **Severity: CRITICAL — Invalidates cross-validation and the Competitive Ratio (CR) metric**

### Paper Claim

**Section 5.1** (paper.md L921–924):
> *"We divide the dataset into n parts, one for testing and the remaining n−1 parts for training. 10% of the training set is allocated as the validation set... In our experiments, we set n to 5."*

The Competitive Ratio (CR) is defined as `OD / basesim`, where `basesim` should be the OD produced by the baseline (TRACLUS) clustering on the **current fold's data**.

### Code Reality

**File**: [MDP.py](file:///Users/wilsebbis/Developer/REAL_RLSTC/RLSTCcode/subtrajcluster/MDP.py) L33–47

```python
def _load(self, cand_train, base_centers_T, base_centers_E):
    ...
    centers_T = pickle.load(open(base_centers_T, 'rb'), encoding='bytes')
    self.basesim_T = centers_T[0][1]    # <-- LOADED FROM A STATIC FILE
    ...
    centers_E = pickle.load(open(base_centers_E, 'rb'), encoding='bytes')
    self.basesim_E = centers_E[0][1]    # <-- SAME STATIC FILE
```

**File**: [crosstrain.py](file:///Users/wilsebbis/Developer/REAL_RLSTC/RLSTCcode/subtrajcluster/crosstrain.py) L117

```python
env = TrajRLclus(trainfilename, args.baseclusT, args.baseclusE)
```

The default arguments for both `baseclusT` and `baseclusE` are `'../data/tdrive_clustercenter'` — **the same static file for every fold**. The `basesim` denominator is computed once on the full dataset (or an arbitrary subset) and then reused across all 5 folds.

### Why This Is Devastating

- The **Competitive Ratio** `CR = OD / basesim` is the primary validation metric used for model selection (see `rl_train.py` L31: `aver_cr = float(odist_e/env.basesim_E)`).
- If `basesim` is a **global constant** rather than fold-specific, the CR is measuring relative improvement against a fixed scalar, not against the baseline's actual performance on that fold's data.
- A validation fold that is inherently easier (lower OD) will appear artificially good because the denominator is too large. Conversely, a harder fold will appear artificially bad.
- This makes the "N-fold cross-validation" reported in the paper essentially a single-denominator experiment with different numerators — **it is not true cross-validation**.

Furthermore, in `crossvalidate.py` L94–95, the same static file is used for every fold iteration:
```python
parser.add_argument("-baseclusT", default='../data/tdrive_clustercenter')
parser.add_argument("-baseclusE", default='../data/tdrive_clustercenter')
```

---

<a id="error-03"></a>
## ERROR-03: The `ODb` State Feature Is a Hardcoded Scalar Multiply, Not TRACLUS Output

> [!WARNING]
> **Severity: HIGH — The "expert knowledge" state feature is fabricated**

### Paper Claim

**Section 4.2**, Equation 18–19 (paper.md L593–609):
> *"The convergence of the RL model can be expedited by integrating the expert knowledge, represented as ODb generated by TRACLUS."*

The state is defined as: `st = (ODs, ODn, ODb, Lb, Lf)` where `ODb` is specifically described as the OD value produced by TRACLUS segmentation.

### Code Reality

**File**: [MDP.py](file:///Users/wilsebbis/Developer/REAL_RLSTC/RLSTCcode/subtrajcluster/MDP.py) L74

```python
observation = np.array([self.overall_sim, self.minsim, self.overall_sim*10,
                        2 / self.length, (self.length - 1) / self.length]).reshape(1, -1)
```

The third state feature, which should be `ODb` (the TRACLUS-generated OD), is simply **`self.overall_sim * 10`** — a constant scalar multiple of the first feature (`ODs`).

This is repeated in the `step()` function at L114 and L151:
```python
observation = np.array([self.overall_sim, self.split_overdist, self.overall_sim*10,
                        (index - self.split_point + 2) / self.length,
                        (self.length - (index + 1)) / self.length]).reshape(1, -1)
```

### Why This Matters

- The paper explicitly claims `ODb` is derived from TRACLUS and represents "expert knowledge" that accelerates convergence (Section 5.8, Figure 13).
- In reality, `ODb = 10 × ODs` is a linear transformation of an existing feature — it contributes **zero additional information** to the state vector after the first hidden layer.
- The ablation study in Figure 13 ("Impact of ODb") comparing models with and without ODb is essentially comparing a 5-feature model against a 4-feature model where the 5th feature is a trivial linear dependent of the 1st. The `MDPwoODb.py` variant removes this feature, but the comparison is meaningless because the "with ODb" variant never actually used TRACLUS data.

---

<a id="error-04"></a>
## ERROR-04: Training Epoch Count Discrepancy — Code Does 1–2 Epochs, Not Convergence-Based Training

> [!WARNING]
> **Severity: HIGH — Model may be severely undertrained**

### Paper Claim

**Section 4.3 / Algorithm 2** (paper.md L715–757): The training loop runs `for epoch = 1, 2, ..., m` with periodic validation and model checkpointing. The paper does not specify a fixed small `m` — Algorithm 2 implies training until convergence.

### Code Reality

**File**: [rl_train.py](file:///Users/wilsebbis/Developer/REAL_RLSTC/RLSTCcode/subtrajcluster/rl_train.py) L39

```python
Round = 2
```

**File**: [crosstrain.py](file:///Users/wilsebbis/Developer/REAL_RLSTC/RLSTCcode/subtrajcluster/crosstrain.py) L30

```python
Round = 1
```

The cross-training variant (`crosstrain.py`) runs for **exactly 1 epoch** over the training data. The standard trainer (`rl_train.py`) runs for **exactly 2 epochs**.

### Why This Matters

- With only 500 training trajectories (default in `rl_train.py` L90), and a batch size of 32, the model sees each trajectory at most twice.
- RL algorithms typically require thousands of episodes for stable learning. Two epochs over 500 episodes is extremely low.
- The paper's convergence discussion (Section 5.8, Figure 15) claims "OD stabilizes after the 10th iteration" — but this refers to RLSTC's outer segmentation-clustering loop, not the DQN training epochs. The distinction is never made clear, and the codebase's inner training loop is severely truncated.

---

<a id="error-05"></a>
## ERROR-05: Soft Update Parameter `ω` Mismatch — 0.05 vs. Paper's 0.001

> [!IMPORTANT]
> **Severity: MEDIUM — Hyperparameter reported in paper differs from code**

### Paper Claim

**Section 5.3** (paper.md L977–979):
> *"The target network updates follow θ′ = ωθ + (1−ω)θ′ at the end of each episode, with ω set to 0.001."*

### Code Reality

**File**: [rl_train.py](file:///Users/wilsebbis/Developer/REAL_RLSTC/RLSTCcode/subtrajcluster/rl_train.py) L63

```python
RL.soft_update(0.05)
```

The `ω` parameter used in the main training script is **0.05** — fifty times larger than the claimed 0.001. The cross-training script (`crosstrain.py` L53) uses **0.001**, which matches the paper.

Additionally, `soft_update()` is called **per-batch** (Line 63, inside the inner training loop), not "at the end of each episode" as stated in the paper. This means the target network is updated hundreds of times per episode, not once.

---

<a id="error-06"></a>
## ERROR-06: Epsilon Decay Is Per-Batch, Not Per-Step — Exploration Schedule Diverges from Paper

> [!IMPORTANT]
> **Severity: MEDIUM — Exploration strategy differs from paper**

### Paper Claim

**Section 5.3** (paper.md L972–974):
> *"Initially set to 1.0 and then reduced to 0.99ε after each step, with a lower bound of 0.1..."*

### Code Reality

**File**: [rl_nn.py](file:///Users/wilsebbis/Developer/REAL_RLSTC/RLSTCcode/subtrajcluster/rl_nn.py) L118–119

```python
def replay(self, episode, batch_size):
    ...
    if self.epsilon > self.epsilon_min:
        self.epsilon *= self.epsilon_decay
```

Epsilon is decayed inside the `replay()` method, which is called **once per training step after the batch_size threshold is met** (see `rl_train.py` L61–62). This is per-batch, not per-step in the MDP sense. However, the critical issue is that "step" in the paper refers to each point in each trajectory. With trajectories of length up to 500, epsilon would decay to its minimum (0.1) within approximately 230 MDP steps (`0.99^230 ≈ 0.1`), which is less than a single long trajectory. In the code, because `replay` is called per MDP step, this behavior approximately matches — but only if the memory buffer threshold is met early.

---

<a id="error-07"></a>
## ERROR-07: Missing N-fold Cross-Validation During Training — Single-Split Evaluation

> [!WARNING]
> **Severity: HIGH — Claimed N-fold validation does not match implementation**

### Paper Claim

**Section 5.1**: *"We divide the dataset into n parts... In our experiments, we set n to 5."*
**Section 5.6**: *"We apply the N-fold validation method and report the average clustering performance."*

### Code Reality

**File**: [rl_train.py](file:///Users/wilsebbis/Developer/REAL_RLSTC/RLSTCcode/subtrajcluster/rl_train.py) L99–103

```python
validation_percent = 0.1
sidx = int(args.amount * (1-validation_percent))
eidx = args.amount
train(args.amount, args.saveclus, sidx, eidx)
```

The standard training script (`rl_train.py`) uses a **single 90/10 split** — not 5-fold cross-validation. The `crosstrain.py` script does implement k-fold splitting but:

1. Uses the **same `basesim` denominator** for all folds (ERROR-02).
2. Only trains for **1 epoch** per fold (ERROR-04).
3. Saves models based on the first alphabetical filename, not best validation CR (ERROR-16).

The cross-validation in `crossvalidate.py` *evaluates* across k folds but relies on pre-trained models from `crosstrain.py`, which itself has the above issues.

---

<a id="error-08"></a>
## ERROR-08: SSE Computation Is Incorrect — Uses Wrong Normalization Denominator

> [!IMPORTANT]
> **Severity: MEDIUM — SSE metric (Eq. 27) is incorrectly implemented**

### Paper Claim

**Section 5.5**, Equation 27 (paper.md L1049–1055):
$$SSE = \sum_{i=1}^{k} \frac{1}{2|C_i|} \sum_{x \in C_i} \sum_{y \in C_i} d(x,y)^2$$

The denominator should be `2 * |C_i|` — twice the number of sub-trajectories in cluster `C_i`.

### Code Reality

**File**: [crossvalidate.py](file:///Users/wilsebbis/Developer/REAL_RLSTC/RLSTCcode/subtrajcluster/crossvalidate.py) L51–71

```python
sse, sse_ori, sse_count, num = 0, 0, 0, 10
...
if cluster_size != 0:
    dist_sum_clus = dist_sum/(2*num*cluster_size)    # <-- num=10, NOT |C_i|
    dist_sum_clus_ori = dist_sum_ori/(2*num*cluster_size)
    sse += dist_sum_clus
```

The variable `num` is hardcoded to **10** (Line 51). The denominator becomes `2 * 10 * cluster_size = 20 * cluster_size` instead of the paper's `2 * cluster_size`. This divides the SSE by an extra factor of 10, making all reported SSE values in the paper approximately **10× smaller** than they should be.

Additionally, there is a logic bug at Lines 60–66: when `dist == 1e10` (non-overlapping trajectories), the code sets `sse_count += 1` and skips the `dist_square` accumulation — but `dist_square` was **never reset** from the previous iteration, so a stale value leaks into `dist_sum`.

```python
for j in range(cluster_size): 
    for k in range(j+1,cluster_size):
        dist = traj2trajIED(subtrajs[j].points, subtrajs[k].points)
        dist_square_ori = dist*dist
        dist_sum_ori += dist_square_ori
        if dist == 1e10:
            sse_count += 1           # counted but not excluded from dist_sum_ori
        else:
            dist_square = dist*dist  # <-- dist_square set here
        dist_sum += dist_square      # <-- but used here REGARDLESS of the if/else
```

When `dist == 1e10`, the code skips the `else` block but still executes `dist_sum += dist_square`, which adds the **previous iteration's `dist_square`** (or 0 if first iteration) to the sum. This is a classic uninitialized-variable bug.

---

<a id="error-09"></a>
## ERROR-09: `initcenters.py` Uses Farthest-First Traversal, Not K-Means++ Probabilistic Seeding

> [!IMPORTANT]
> **Severity: MEDIUM — Initialization algorithm differs from paper**

### Paper Claim

**Section 4.3 / Algorithm 2, Line 4** (paper.md L714):
> *"Initialize k cluster centers by k-means++ method"*

**Section 4.5 / Algorithm 3, Line 1** (paper.md L849):
> *"Initially, we utilize the k-means++ method to initialize k cluster centers."*

K-means++ initialization selects each new center with probability **proportional to D(x)²** — the squared distance to the nearest existing center. This is a probabilistic algorithm.

### Code Reality

**File**: [initcenters.py](file:///Users/wilsebbis/Developer/REAL_RLSTC/RLSTCcode/subtrajcluster/initcenters.py) L12–18

```python
def initialize_centers(data, K):
    centers = [random.choice(data)]
    while len(centers) < K:
        distances = [min([traj2trajIED(center.points, traj.points) for center in centers]) for traj in data]
        new_center = data[distances.index(max(distances))]
        centers.append(new_center)
    return centers
```

This is **deterministic farthest-first traversal (maximin)**, not k-means++:
- It selects `data[distances.index(max(distances))]` — the point with the **maximum** minimum distance.
- K-means++ would sample from a probability distribution weighted by `D(x)²`.

### Why This Matters

Farthest-first traversal is more sensitive to outliers than k-means++. The first outlier in the dataset will be selected as the second center (being farthest from the random first center), which cascades into the remaining centers. This produces qualitatively different initial clusterings and undermines the paper's claim that initialization follows the k-means++ literature.

---

<a id="error-10"></a>
## ERROR-10: Empty Cluster Handling Bug — Assigns to Wrong Cluster Index

> [!WARNING]
> **Severity: HIGH — Corrupts cluster assignments silently**

### Code Reference

**File**: [initcenters.py](file:///Users/wilsebbis/Developer/REAL_RLSTC/RLSTCcode/subtrajcluster/initcenters.py) L41–44

```python
for i in range(k):
    if len(cluster_segments[i]) == 0:
        cluster_segments[minidx].append(centers[i])  # <-- BUG: uses minidx, not i
        dists_dict[i].append(0)
```

When cluster `i` is empty, the code appends that cluster's center to `cluster_segments[minidx]` instead of `cluster_segments[i]`. The variable `minidx` is a **leaked loop variable** from the assignment loop above (L27: `minidx = 0` / L34: `minidx = j`). It holds the cluster index of the **last assigned sub-trajectory**, which is completely unrelated to the empty cluster being fixed.

### Impact

- This can cause one cluster to accumulate centers from other empty clusters, while the empty clusters remain empty (with only `dists_dict[i].append(0)` applied, but no segments).
- Since `cluster_dict` is built from `cluster_segments` in the next loop (L46–54), empty clusters will have no center trajectory at index `[1]`, causing `len(cluster_dict[i][1].points) == 0` errors downstream.

---

<a id="error-11"></a>
## ERROR-11: Reward Signal Is Zero for Non-Segmentation Actions — Violates Paper's Reward Definition

> [!IMPORTANT]
> **Severity: MEDIUM — Reward sparsity differs from paper specification**

### Paper Claim

**Section 4.2** (paper.md L628–652):
> *"Assuming a state transition from st to st+1 after taking an action at, we define an immediate reward as the difference in OD values between successive states, i.e., rt = st(OD) − st+1(OD)."*

This applies to **all** state transitions — both segmentation (action=1) and non-segmentation (action=0).

### Code Reality

**File**: [MDP.py](file:///Users/wilsebbis/Developer/REAL_RLSTC/RLSTCcode/subtrajcluster/MDP.py) L117–118

```python
if action == 0:
    ...
    reward = 0                # <-- ALWAYS ZERO for action=0
    return observation, reward
```

vs. L155:

```python
if action == 1:
    ...
    reward = last_overall_sim - self.overall_sim  # <-- Non-zero for action=1
    return observation, reward
```

The reward is **always zero** when the agent chooses not to segment (`action=0`). This creates an extremely sparse reward signal — the agent only receives feedback at segmentation points, not at every MDP step.

### Why This Matters

- The paper's Equation 20 derives cumulative reward as `R = s1(OD) − s|T|(OD)`, which is valid when both actions produce non-zero rewards that telescope. With zero rewards on action=0, the telescoping property still holds (the non-zero rewards still sum to the final OD difference), but the per-step learning signal is fundamentally different.
- With sparse rewards, the DQN must propagate credit backward through many zero-reward steps, making learning significantly harder than the paper's formulation implies.

---

<a id="error-12"></a>
## ERROR-12: `computecenter` Averaging Bug — Sums Raw Coordinates Instead of Per-Point Means

> [!WARNING]
> **Severity: HIGH — Centroid computation produces spatially incorrect centers**

### Paper Claim

**Section 4.4** (paper.md L793–794):
> *"we compute the average coordinate for that timestamp"*

### Code Reality

**File**: [cluster.py](file:///Users/wilsebbis/Developer/REAL_RLSTC/RLSTCcode/subtrajcluster/cluster.py)

The coordinate accumulation happens in `add2clusdict()` (L181–196):

```python
def add2clusdict(points, clus_dict, k):
    ...
    if curr_t not in clus_dict[k][3]:
        clus_dict[k][3][curr_t] = [[points[i]], 1, points[i].x, points[i].y]
    else:
        clus_dict[k][3][curr_t][0].append(points[i])
        clus_dict[k][3][curr_t][2] += points[i].x     # <-- ACCUMULATES x
        clus_dict[k][3][curr_t][3] += points[i].y     # <-- ACCUMULATES y
```

Then in `computecenter()` (L198–236), the averaging is done as:

```python
sum_x, sum_y = clus_dict[k][3][sortkeys[start_i]][2], clus_dict[k][3][sortkeys[start_i]][3]
count = len(clus_dict[k][3][sortkeys[start_i]][0])
...
while i < len(sortkeys):
    if sortkeys[i] - sortkeys[start_i] <= threshold_t:
        count += len(clus_dict[k][3][sortkeys[i]][0])
        sum_x += clus_dict[k][3][sortkeys[i]][2]      # <-- Adds RAW ACCUMULATED x
        sum_y += clus_dict[k][3][sortkeys[i]][3]       # <-- Adds RAW ACCUMULATED y
        ...
    aver_x, aver_y = sum_x / count, sum_y / count      # <-- Divides by POINT count
```

**The bug**: `clus_dict[k][3][t][2]` is the **accumulated sum** of x-coordinates of all points at timestamp `t`, not a single coordinate. When `computecenter()` adds these across multiple timestamps in a window, `sum_x` becomes the sum of sums. The division by `count` (total number of points across all timestamps in the window) then produces the correct per-timestamp-weighted average **only if every timestamp has exactly one point**. If any timestamp has multiple contributing points, the averaging is wrong because the spatial sums and the point count are mixed across timestamps.

Additionally, `clus_dict[k][3][key][1]` (the "number of trajectories" at a timestamp) tracked by `add2clusdict()` counts the number of trajectories whose temporal range *contains* the timestamp — but **only increments when a new sub-trajectory is added to the cluster**, not when it's initially populated. This means the threshold check (`if clus_dict[k][3][key][1] >= threshold_num`) uses an undercount during early iterations.

---

<a id="error-13"></a>
## ERROR-13: Missing Spatial Normalization — Only Time Is Z-Score Normalized

> [!IMPORTANT]
> **Severity: MEDIUM — Spatial coordinates are treated inconsistently**

### Paper Claim

The paper does not explicitly specify normalization, but the experiments use both spatial (IED, Euclidean distance) and temporal components. Standard practice (and the definition of `dst` in Equation 10, paper.md L401–404) weighs spatial and temporal components equally:
> `dst = 0.5 * dED(pri, pri+1) + 0.5 * |pri(t) − pri+1(t)|`

### Code Reality

**File**: [preprocessing.py](file:///Users/wilsebbis/Developer/REAL_RLSTC/RLSTCcode/subtrajcluster/preprocessing.py) L69–83

```python
def normtimetrajs(trajs):
    ...
    norm_t = (trajs[i][j][2] - mean_t) / std_t
    tmp_traj.append([trajs[i][j][0], trajs[i][j][1], norm_t])  # x,y UNCHANGED
```

Only the temporal dimension is z-score normalized. The spatial coordinates (longitude/latitude) are left as raw geographic coordinates. A `normloctrajs()` function exists (L51–67) that z-score normalizes longitude and latitude, but **it is never called** in the `__main__` block (L129–144):

```python
trajs = processlength(trajslist, args.maxlen, args.minlen)
norm_trajs = normtimetrajs(trajs)          # <-- Only time normalization
trajlists = convert2traj(norm_trajs)       # <-- normloctrajs() never called
```

### Why This Matters

- With raw geographic coordinates (~116° longitude, ~40° latitude for Beijing), the spatial distances are on a completely different scale than normalized timestamps.
- The MDL preprocessing (Equation 10) weights spatial and temporal equally (`0.5 * dED + 0.5 * |Δt|`), but if time is z-scored (mean ~0, std ~1) while spatial is raw (~100s), the spatial term will dominate by orders of magnitude.
- The IED distance computation similarly mixes these scales: spatial Euclidean distances in degrees vs. temporal differences in z-scored units.

---

<a id="error-14"></a>
## ERROR-14: `Point.equal()` Returns `None` on Inequality — Silent Falsy Bug

> [!NOTE]
> **Severity: LOW — Logical bug that may not currently cause failures**

### Code Reference

**File**: [point.py](file:///Users/wilsebbis/Developer/REAL_RLSTC/RLSTCcode/subtrajcluster/point.py) L13–16

```python
def equal(self, other):
    if self.x == other.x and self.y == other.y and self.t == other.t:
        return True
    # <-- No explicit return False; returns None implicitly
```

When two points are not equal, `equal()` returns `None` instead of `False`. While `None` is falsy in Python and will work in most boolean contexts, it can cause subtle bugs if the return value is compared with `is False` or used in arithmetic.

**Impact**: The function is called in `preprocessing.py` L114:
```python
if not simp_points[-1].equal(points[-1]):
```
This works because `not None` is `True`, but it's still a code quality issue that could cause confusion during maintenance.

---

<a id="error-15"></a>
## ERROR-15: Huber Loss Instead of MSE — Contradicts Paper's Stated Loss Function

> [!IMPORTANT]
> **Severity: MEDIUM — Different loss function than stated in the paper**

### Paper Claim

**Section 4.3**, Equation 22 (paper.md L704–706):
> *"aiming to minimize the mean squared error (MSE), defined as: MSE(θ) = (y − Q(st, at; θ))²"*

### Code Reality

**File**: [rl_nn.py](file:///Users/wilsebbis/Developer/REAL_RLSTC/RLSTCcode/subtrajcluster/rl_nn.py) L44–49, L57

```python
def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
    error = y_true - y_pred
    cond = K.abs(error) <= clip_delta
    squared_loss = 0.5 * K.square(error)
    quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)
    return K.mean(tf.where(cond, squared_loss, quadratic_loss))

model.compile(loss=self._huber_loss, optimizer=SGD(lr=self.learning_rate))
```

The model uses **Huber loss** (with `clip_delta=1.0`), not MSE. Huber loss is less sensitive to outliers than MSE, which changes the optimization landscape for large reward signals. While Huber loss is actually a common improvement in DQN implementations (and was used in the original Atari DQN paper), the RLSTC paper explicitly claims MSE.

---

<a id="error-16"></a>
## ERROR-16: Model Selection Uses First Alphabetical File, Not Best Validation Model

> [!WARNING]
> **Severity: HIGH — Evaluation may not use the best-trained model**

### Code Reference

**File**: [crossvalidate.py](file:///Users/wilsebbis/Developer/REAL_RLSTC/RLSTCcode/subtrajcluster/crossvalidate.py) L106–107

```python
modelnames = os.listdir(savecluspath)
model = savecluspath + '/' + modelnames[0]    # <-- FIRST FILE (OS-dependent order)
```

**File**: [rl_estimate.py](file:///Users/wilsebbis/Developer/REAL_RLSTC/RLSTCcode/subtrajcluster/rl_estimate.py) L88–89

```python
modelnames = os.listdir(args.modeldir)
model = args.modeldir + '/' + modelnames[args.modelchoose]  # default modelchoose=0
```

Models are saved with filenames like `sub-RL-{aver_cr}.h5` (see `rl_train.py` L78). The evaluation scripts load `modelnames[0]` — the **first file in directory listing order**, which is filesystem-dependent (not guaranteed to be sorted, and on many systems is inode-order).

### Compounding Issue

In `rl_train.py` L77–78, models are saved **every 500 episodes regardless of improvement**:

```python
if aver_cr < check or episode % 500 == 0:        # <-- ALWAYS true at episode multiples of 500
    RL.save(saveclus + '/sub-RL-' + str(aver_cr) + '.h5')
```

The condition `episode % 500 == 0` is redundant with the outer `if episode % 500 == 0` check at L66, so **every checkpoint is saved**, not just improvements. This fills the model directory with many non-optimal checkpoints, and the first one loaded may be from early training (high CR, poor quality).

---

## Summary Table

| ID | Severity | Category | Paper Claim | Code Reality |
|---|---|---|---|---|
| **E-01** | 🔴 CRITICAL | Clustering | Centroid averaging (Sec 4.4) | Max-distance seed, never recomputed |
| **E-02** | 🔴 CRITICAL | Evaluation | Fold-specific basesim (Sec 5.1) | Static global scalar for all folds |
| **E-03** | 🟠 HIGH | MDP State | ODb from TRACLUS (Eq 18) | `overall_sim * 10` (linear copy) |
| **E-04** | 🟠 HIGH | Training | Convergence-based (Alg 2) | 1–2 fixed epochs |
| **E-05** | 🟡 MEDIUM | Hyperparams | ω = 0.001 (Sec 5.3) | ω = 0.05 in rl_train.py |
| **E-06** | 🟡 MEDIUM | Hyperparams | Per-step ε decay (Sec 5.3) | Per-batch ε decay |
| **E-07** | 🟠 HIGH | Evaluation | 5-fold cross-validation (Sec 5.1) | Single 90/10 split in trainer |
| **E-08** | 🟡 MEDIUM | Metrics | SSE with `2|Ci|` denom (Eq 27) | Hardcoded `2*10*|Ci|` + stale var |
| **E-09** | 🟡 MEDIUM | Initialization | K-means++ (Sec 4.3) | Farthest-first traversal |
| **E-10** | 🟠 HIGH | Bug | N/A | Empty cluster → wrong index |
| **E-11** | 🟡 MEDIUM | MDP Reward | rt for all actions (Eq 20) | reward=0 for action=0 |
| **E-12** | 🟠 HIGH | Clustering | Per-timestamp averaging (Sec 4.4) | Mixed sum-of-sums averaging |
| **E-13** | 🟡 MEDIUM | Preprocessing | Consistent normalization | Only time is z-scored |
| **E-14** | 🟢 LOW | Bug | N/A | `equal()` returns None |
| **E-15** | 🟡 MEDIUM | Loss Function | MSE (Eq 22) | Huber loss (clip_delta=1.0) |
| **E-16** | 🟠 HIGH | Evaluation | Best model selection | First alphabetical file |

---

## Implications for Thesis Narrative

These errors collectively demonstrate that the RLSTC baseline published in VLDB 2024 contains **fundamental implementation defects** that undermine its empirical claims:

1. **The reported OD values are unreliable** (E-01, E-02, E-08, E-12) because the "cluster centers" are not true centroids, the normalization denominator is wrong, and the validation metric uses a static scalar.

2. **The RL agent was barely trained** (E-04, E-05, E-07) with 1–2 epochs, incorrect soft-update rates, and no true cross-validation during training.

3. **The MDP formulation differs from the paper** (E-03, E-06, E-11) with a fabricated state feature, different exploration schedules, and sparse rewards.

4. **The visualizations are misleading** (E-01, E-09) because "representative trajectories" are just individual seed trajectories, and initialization is not k-means++.

These findings justify the complete architectural teardown performed in the Q-RLSTC refactor and explain why the published baseline metrics cannot be treated as a reliable comparison target.
