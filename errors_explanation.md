# RLSTC Master Taxonomy
### Exhaustive Reference: Reinforcement Learning Sub-Trajectory Clustering & Recursive Least Squares Timing Correction

---

## 0. Acronym Disambiguation

| Domain | Expansion | Primary Problem | Core Mechanism |
|---|---|---|---|
| Spatial Data Mining | Reinforcement Learning-based Sub-Trajectory Clustering | GPS trajectory over-segmentation & rigid heuristics | MDP + DQN + K-means feedback loop |
| Aerospace / Embedded Systems | Recursive Least Squares Timing Correction | UAV DShot timing attack vulnerability | Recursive residual minimization on ESC interface |
| Telecom (Legacy) | `rlstc:cell=cellID` terminal command | GSM/LTE cell activation state management | Ericsson Abis interface command |
| Cheminformatics | `rlstca` — Realistic Splits | QSAR model data partitioning bias | Structural overlap-aware train/test splitting |

---

## 1. Classical RLSTC — Spatial Sub-Trajectory Clustering

### 1.1 Problem Genealogy

```
Full-Trajectory Clustering
  └── Flaw: Averages out localized movement patterns
      └── Sub-Trajectory Clustering (conceptual fix)
          └── Flaw: Two-phase pipeline with human-crafted segmentation heuristics
              ├── Brittle ruleset (velocity, turning angle, pause thresholds)
              ├── Hyper-sensitive to initial segmentation parameters
              └── Requires continuous domain-expert recalibration
                  └── RLSTC (unified RL solution)
```

**Coverage Maximization Framing:** Segmentation can be formalized as finding the minimum number of sub-trajectory clusters covering maximum dataset variance — solvable in cubic time via deterministic geometry, but impractical at scale → motivates approximate learning agents.

---

### 1.1.1 Formal Notation Table (Paper §2)

| Symbol | Description |
|---|---|
| $D$ | Dataset of $N$ trajectories: $D = \{T_1, T_2, \ldots, T_N\}$ |
| $N$ | Number of trajectories in dataset $D$ |
| $T$ | A trajectory |
| $p_i$ | The $i$-th point of $T$; triplet $(x_i, y_i, t_i)$ where $(x_i, y_i)$ = location, $t_i$ = timestamp |
| $p^{(x)}, p^{(y)}, p^{(t)}$ | Longitude, latitude, timestamp of point $p$ |
| $\|T\|$ | Length of $T$ (number of points) |
| $T(i,j)$ | Sub-trajectory of $T$ from point $i$ to point $j$: $p_i, \ldots, p_j$ |
| $T(t_s), T(t_e)$ | Starting and ending time of trajectory $T$ |
| $s_t$ | State at time step $t$ |
| $a_t$ | Action taken at time step $t$ |
| $r_t$ | Reward obtained at time step $t$ |
| $\theta$ | Parameters of the main neural network |
| $\theta'$ | Parameters of the target neural network |
| $\{C_j\}_{j=1}^k$ | Learned $k$ clusters |
| $c_i$ | Center of cluster $C_i$ (not necessarily a sub-trajectory in $C_i$) |
| $OD$ | Metric for evaluating clustering quality (Overall Distance / Agglomeration Degree) |

---

### 1.1.2 Formal Definitions (Paper §2)

**Definition 1 — Trajectory:** A sequence of time-ordered spatial points $T = p_1, p_2, \ldots, p_n$ where each $p_i = (x_i, y_i, t_i)$. Length $|T| = n$. Time interval $[t_s, t_e]$.

**Definition 2 — Sub-trajectory:** Given $T = p_1, \ldots, p_n$, a sub-trajectory $T(i,j)$ $(1 \leq i \leq j \leq n)$ is $p_i, \ldots, p_j$.

**Definition 3 — Sub-trajectory Cluster:** A cluster $C_i$ of sub-trajectories similar w.r.t. a distance metric. Center $c_i$ is not necessarily a sub-trajectory in $C_i$.

**Definition 4 — Clustering Quality / OD:**
$$OD = \sum_{i=1}^{k} \frac{m_i}{m} OD_i, \quad OD_i = \frac{\sum_{x \in C_i} d(x, c_i)}{m_i}$$
where $d(\cdot, \cdot)$ is any trajectory distance metric. Any distance metric can be substituted in $OD_i$.

**Definition 5 — Synchronous Point:** For two trajectories $T$ and $Q$, point $p_j \in Q$ is the synchronous point of $p_i \in T$ if they share the same timestamp. For points lacking a synchronous counterpart, linear interpolation produces one:
$$p'^{(x)} = p_s^{(x)} + \frac{p^{(t)} - p_s^{(t)}}{p_e^{(t)} - p_s^{(t)}} \cdot (p_e^{(x)} - p_s^{(x)})$$
$$p'^{(y)} = p_s^{(y)} + \frac{p^{(t)} - p_s^{(t)}}{p_e^{(t)} - p_s^{(t)}} \cdot (p_e^{(y)} - p_s^{(y)}), \quad p'^{(t)} = p^{(t)}$$

**Boundary condition:** Object corresponding to $T$ is assumed stationary before $T(t_s)$ and after $T(t_e)$. Points with timestamps outside $T$'s range inherit $T$'s first or last point location.

**Definition 6 — Trajectory Spatio-temporal Distance (IED):**
$$d_{\text{IED}}(Q, T) = \int_{t_{\min}}^{t_{\max}} d_{Q,T}(t)\, dt$$
where $t_{\min} = \min(Q(t_s), T(t_s))$, $t_{\max} = \max(Q(t_e), T(t_e))$, and $d_{Q,T}(t)$ is the synchronous Euclidean distance (SED) at time $t$.

Trapezoid approximation (reduces to $O(n+m)$):
$$d_{\text{IED}}(Q, T) = \frac{1}{2}\sum_{k=1}^{n+m-1}(d_{Q,T}(t_k) + d_{Q,T}(t_{k+1})) \cdot (t_{k+1} - t_k)$$

**Incremental IED** (reduces to $O(1)$ per step for sub-trajectory extension):
$$d_{\text{IED}}(Q, T(i,j)) = \underbrace{d_{\text{IED}}(Q, T(i,j{-}1))}_{\text{already computed}} + \frac{1}{2}(d_{Q,T}(t_{j-1}) + d_{Q,T}(t_j)) \cdot (t_j - t_{j-1})$$

**Problem Statement:** Given dataset $D$, find segmentation producing optimal clustering $C_{\text{opt}} = \{C_j\}_{j=1}^k$ that minimizes $OD$ (Eq. 1).

---

### 1.2 Markov Decision Process (MDP) Formulation

| MDP Component | RLSTC Implementation | Detail |
|---|---|---|
| **State Space** | 5D continuous feature vector | Captures current geometric + temporal trajectory configuration |
| **Action Space** | Binary | `EXTEND` (0) — continue aggregating; `CUT` (1) — end segment, start new |
| **Transition** | State reset | Geometric parameters re-initialized upon CUT |
| **Reward Function** | Δ Overall Distance (OD) | Positive reward for improved cluster cohesion |
| **Environment Class** | `TrajRLclus` | Isolates physical env dynamics from neural learning signals |

#### 1.2.1 State Feature Vector (5D — Paper Eq. 15–19)

The state at the $t$-th point of trajectory $T$ given $k$ cluster centers is $s_t = (s_t(OD_s), s_t(OD_n), OD_b, s_t(L_b), s_t(L_f))$.

| Index | Paper Symbol | Identifier | Formula | Semantic Definition |
|---|---|---|---|---|
| 0 | $s_t(OD_s)$ | `ODs` | $s_{t-1}(OD) \cdot \frac{num_{s_{t-1}}}{num_{s_{t-1}}+1} + \frac{\min_j d(x, c_j)}{num_{s_{t-1}}+1}$ | Projected OD if CUT is executed at current point |
| 1 | $s_t(OD_n)$ | `ODn` | $s_{t-1}(OD)$ | OD if current point is not cut (EXTEND) |
| 2 | $OD_b$ | `ODb` | $\frac{\sum_{i=1}^{num} \min_j d(x'_i, c_j)}{num}$ | Expert baseline OD from TRACLUS segmentation |
| 3 | $s_t(L_b)$ | `Lb` | $m_1 / \|T\|$ | Normalized length of generated sub-trajectory so far |
| 4 | $s_t(L_f)$ | `Lf` | $m_2 / \|T\|$ | Normalized length of remaining trajectory |

where $x$ = sub-trajectory produced by cutting at $p_t$; $m_1$ = points in generated sub-trajectory; $m_2$ = points from $p_t$ to end of $T$.

> **Design rationale for $OD_s$ vs. $OD_n$:** Relying on $OD_s < OD_n$ alone makes the agent shortsighted — it may cut at $p_t$ when better cuts exist downstream. $L_b$ and $L_f$ provide horizon awareness.

> **$OD_b$ purpose:** Integrating TRACLUS expert knowledge accelerates convergence. Paper ablation (Fig. 13) confirms significantly better clustering quality with $OD_b$ included on both T-Drive and Geolife.

#### 1.2.2 Reward Formulation (Paper §4.2)

Immediate reward at transition $s_t \to s_{t+1}$:
$$r_t = s_t(OD) - s_{t+1}(OD)$$

Accumulative reward over trajectory of length $|T|$:
$$R = \sum_{t=1}^{|T|-1} r_t = s_1(OD) - s_{|T|}(OD)$$
where $s_1(OD) = 0$. Maximizing $R$ is equivalent to minimizing terminal OD.

#### 1.2.3 Anti-Gaming Constraints

- **Minimum segment length:** `min_seg_len`
- If CUT is output but active segment < `min_seg_len`, environment silently overrides → forces `EXTEND`
- Same override applies if remaining forward trajectory < `min_seg_len`
- Prevents degenerate micro-segmentation (cutting at every point)

---

### 1.3 Reward Engineering

**Agglomeration Degree (Overall Distance):**

$$OD = \sum_{i=1}^{k} \frac{m_i}{m} OD_i, \quad OD_i = \frac{\sum_{x \in C_i} d(x, c_i)}{m_i}$$

| Symbol | Definition |
|---|---|
| $k$ | Predefined total number of clusters |
| $OD_i$ | Average spatio-temporal distance: geometric center $c_i$ → all sub-trajectories in cluster $C_i$ |
| $m_i$ | Count of sub-trajectories assigned to $C_i$ |
| $m$ | Total aggregate sub-trajectories across entire dataset |

**Reward signal = $r_t = s_t(OD) - s_{t+1}(OD)$**
- Lower OD → tighter, more cohesive clusters → positive reward for each OD-reducing action
- Accumulative reward telescopes: $R = s_1(OD) - s_{|T|}(OD)$ — maximizing $R$ ↔ minimizing terminal OD

**Sum of Squared Errors (SSE) — Paper Eq. 27:**
$$SSE = \sum_{i=1}^{k} \left(\frac{1}{2|C_i|} \sum_{x \in C_i} \sum_{y \in C_i} d(x, y)^2 \right)$$

SSE measures intra-cluster proximity (particularly relevant for density-based methods like DBSCAN). Lower values = higher quality. OD and SSE together provide complementary cluster quality assessment covering both center-based and density-based perspectives.

---

### 1.4 Neural Architecture — Classical DQN (Paper §4.3, §5.3)

```
Input: 5D State Vector  (ODs, ODn, ODb, Lb, Lf)
  └── Hidden Layer 1: 64 neurons, ReLU activation
      └── Output Layer: 2 neurons  (EXTEND=0 | CUT=1)
          └── Optimizer: SGD, lr=0.001
```

#### 1.4.1 Hyperparameter Reference Table (Paper §5.3 — Authoritative)

| Parameter | Paper Value | Notes |
|---|---|---|
| Hidden layer neurons | 64 | Single hidden layer; ReLU activation |
| Output neurons | 2 | Binary action space |
| Optimizer | SGD | Learning rate = 0.001; empirically determined |
| Exploration parameter $\varepsilon$ | 1.0 → $0.99\varepsilon$ per step, floor 0.1 | ε-greedy strategy |
| Discount factor $\gamma$ | 0.99 | Balances immediate vs. future rewards |
| Replay memory size | **5,000** | Sample size per training iteration = 32 |
| Target network update | $\theta' = \omega\theta + (1-\omega)\theta'$, **ω = 0.001** | Soft update at **end of each episode** |
| Cluster number $k$ | 10 | Best performance across tested values (5, 8, 10, 12, 14) |
| Convergence threshold $\tau$ | 0.1 | Max distance between cluster centers in successive iterations |
| Training set size | 2,000 trajectories | Optimal TR trade-off point |
| Trajectory length bounds | 10–500 points | <10 discarded; >500 randomly reduced to 500 |
| Total parameters | **514** (5→64→2) | Base model footprint |

> **NOTE — Paper vs. Code Discrepancy (ERROR-05):** The paper specifies replay memory = 5,000 and ω = 0.001 with update at end of each episode. The published codebase used ω = 0.05 applied per-batch — a major training instability defect corrected in REAL_RLSTC_FIXED.

#### 1.4.2 Q-Value Learning (Paper §4.3)

Optimal Q-value via Bellman Expectation Equation:
$$Q^*(s, a) = \mathbb{E}\left[r_t + \gamma \max_{a'} Q^*(s_{t+1}, a') \,\middle|\, s_t, a_t\right]$$

MSE loss for DQN training:
$$\text{MSE}(\theta) = (y - Q(s_t, a_t; \theta))^2$$

Target Q-value:
$$y = \begin{cases} r_t & \text{if } s_{t+1} \text{ is terminal} \\ r_t + \gamma \max_{a'} \hat{Q}(s_{t+1}, a'; \theta') & \text{otherwise} \end{cases}$$

#### 1.4.3 Training Stability Strategies (Paper §4.3)

1. **Trajectory shuffling** before each epoch to mitigate data-access-order effects
2. **ε-greedy exploration** to avoid local optima during policy learning
3. **k-means++ initialization** to reflect comprehensive data distribution
4. **Cluster center update** after each training epoch to adapt to evolving sub-trajectory assignments

**Evaluation platform:** Dell server, NVIDIA P100 GPU, 48-core Intel Xeon E5-2678 v3 @ 2.50GHz, 128GB RAM; Python 3.6, TensorFlow 2.2.0

---

### 1.5 Iterative Convergence Loop (Algorithm 3)

```
1. Initialize k cluster centers via k-means++
2. DQN policy traverses + segments each trajectory
3. Assign sub-trajectories to nearest cluster center
4. Recalculate cluster centers
5. Compute max distance shift between old and new centers
6. If max_shift > convergence_threshold:
     → Forward updated centers + OD to segmentation agent
     → DQN adjusts weights
     → Repeat
7. If max_shift ≤ threshold → CONVERGED
```

> Convergence is guaranteed regardless of trajectory traversal order (cluster centers mathematically converge after sufficient iterations).

---

### 1.6 Preprocessing Pipeline (Paper §3)

#### 1.6.1 MDL-Based Trajectory Simplification (Algorithm 1)

**Objective:** Minimize trajectory points while minimizing distortion — two contradictory goals balanced via Minimum Description Length principle.

MDL cost of simplified trajectory $T_{\text{simp}} = \{p_{r_1}, p_{r_2}, \ldots, p_{r_m}\}$ vs. original $T = p_1, \ldots, p_n$:
$$\text{MDL}_{T_{\text{simp}}} = L(H) + L(D|H) = \log_2 \sum_{i=1}^{m-1} d_{\text{st}}(p_{r_i}, p_{r_{i+1}}) + \log_2 \sum_{i=1}^{m-1} d_{\text{IED}}(T_i, T_i^{\text{simp}})$$

$$\text{MDL}_T = L(H) + L(D|H) = \log_2 \sum_{i=1}^{n-1} d_{\text{st}}(p_i, p_{i+1}) + 0$$

where $d_{\text{st}} = \frac{1}{2}d_{\text{ED}}(p_a, p_b) + \frac{1}{2}|p_a^{(t)} - p_b^{(t)}|$ (spatio-temporal segment length).

**Algorithm 1 — Greedy MDL Simplification:**
```
Input:  T = p1, p2, ..., pn
Output: Simplified Tsimp

RP ← [p1];  i ← 0;  step ← 1
while i + step < n:
    j ← i + step
    if MDL_Tsimp(pi, pj) > MDL_T(pi, pj):
        insert pj into RP
        i ← j;  step ← 1
    else:
        step ← step + 1
insert pn into RP
return Tsimp from RP
```

**Time complexity:** $O(n)$ — MDL computation per point is $O(1)$ via incremental IED.

#### 1.6.2 Full Preprocessing Steps

| Stage | Mechanism | Purpose |
|---|---|---|
| **MDL simplification** | Algorithm 1 greedy; retains only "reserved points" | Strips redundant points preserving geometric shape; $O(n)$ |
| Temporal alignment | Linear interpolation → synthetic "synchronous points" | Rectifies inconsistent GPS sampling frequencies |
| MinNum thresholding | Average coordinates only where trajectory count ≥ `MinNum` | Generates lightweight representative proxy trajectories for cluster center computation |
| Bounding box filter | Geographic extent clipping | Removes spatial outliers |
| Z-score normalization | Per-feature standardization | Normalizes coordinate distributions for IED calculation |
| Trajectory bounds | 10-point minimum; 500-point maximum (random reduction) | Discards trivially short trajectories; controls compute cost |

#### 1.6.3 Cluster Center Computation (Paper §4.4)

For each cluster $C_i$, scan timestamps chronologically. At each timestamp $t$:
- Record number of trajectories containing timestamp $t$
- If count ≥ `MinNum`: compute average $(x, y)$ coordinates across all trajectories at $t$
- For trajectories missing timestamp $t$: derive synchronous point via linear interpolation
- Resulting averaged coordinate sequence = representative trajectory (cluster center)

> **Paper claim vs. code reality (ERROR-01):** The paper describes this genuine geometric mean computation. The published code implemented a `max_span` medoid fallback — selecting the longest existing trajectory as center — because true averaging caused stationary-point degeneration. See Section 12 for full forensic details.

---

### 1.7 Distance Metrics and Indexing (Paper §5.4–5.5)

#### 1.7.1 Primary Metric: Integrated Euclidean Distance (IED)

$$d_{\text{IED}}(Q, T) = \frac{1}{2}\sum_{k=1}^{n+m-1}(d_{Q,T}(t_k) + d_{Q,T}(t_{k+1})) \cdot (t_{k+1} - t_k)$$

- Computes total geometric area between two trajectories over shared temporal overlap
- If time intervals of $T$ and $Q$ do not intersect: $d_{\text{IED}} = \infty$
- Standard complexity: $O(n+m)$ → optimized via incremental update to $O(1)$ per step
- Incremental update: add new trapezoidal area only when segment extended by one point

#### 1.7.2 Discrete Fréchet Distance ($d_{\text{DF}}$)

$$d_{\text{DF}}(Q, T) = \begin{cases} d_{\text{ED}}(p_1, q_1) & \text{if } |Q| = |T| = 1 \\ \infty & \text{if } Q = \emptyset \text{ or } T = \emptyset \\ \max(d_{\text{ED}}(p_1, q_1),\ \min(d_{\text{DF}}(Q_h, T_h),\ d_{\text{DF}}(Q, T_h),\ d_{\text{DF}}(Q_h, T))) & \text{otherwise} \end{cases}$$

where $Q_h$ ($T_h$) = sub-trajectory of $Q$ ($T$) starting from the second point.

#### 1.7.3 Dynamic Time Warping Distance ($d_{\text{DTW}}$)

$$d_{\text{DTW}}(Q, T) = \begin{cases} d_{\text{ED}}(p_1, q_1) & \text{if } |Q| = |T| = 1 \\ \infty & \text{if } Q = \emptyset \text{ or } T = \emptyset \\ d_{\text{ED}}(p_1, q_1) + \min(d_{\text{DTW}}(Q_h, T_h),\ d_{\text{DTW}}(Q, T_h),\ d_{\text{DTW}}(Q_h, T)) & \text{otherwise} \end{cases}$$

#### 1.7.4 Weighted Distance ($d_{\text{WD}}$ — TRACLUS-style)

$$d_{\text{WD}}(Q, T) = \omega_\perp d_\perp(Q, T) + \omega_{||} d_{||}(Q, T) + \omega_\theta d_\theta(Q, T)$$

where $d_\perp$, $d_{||}$, $d_\theta$ = perpendicular, parallel, and angular distances; weights $\omega_\perp = \omega_{||} = \omega_\theta = 1$.

#### 1.7.5 Indexing

- **R-tree geometric indexing:** Extracts neighboring segments by bounding rectangle; bypasses $O(n^2)$ pairwise computation
- R-tree structures can themselves be optimized via RL agents (node splitting, subtree insertion); demonstrated on datasets >100M objects

**Overall time complexity:** $O(nN)$ where $n$ = avg trajectory length, $N$ = total trajectory count

Full time complexity: $O(c(nN + mM + m'))$ where $c$ = iterations, $m$ = avg sub-trajectory length, $M$ = total sub-trajectories, $m'$ = avg cluster center length. Simplifies to $O(nN)$ since $O(nN) \approx O(mM)$ and $O(m') < O(nN)$, and $c$ is small.

---

### 1.8 Empirical Performance — Classical (Paper §5)

#### 1.8.1 Dataset Specifications

| Dataset | Source | Scale | Sampling Rate | After Filtering |
|---|---|---|---|---|
| **Geolife** | 182 users, 3 years | 17,364 trajectories, ~1.2M km, >48,000 hrs | 1–5 seconds | **17,070 trajectories** (10–500 pts) |
| **T-Drive** | 10,289 Beijing taxis, 1 week | ~17M points, >9M km total | Every 3–5 min | **9,937 trajectories** (10–500 pts) |

Dataset split: N-fold (N=5); 10% of training set held for validation.

#### 1.8.2 Baseline Methods

| Method | Category | Key Mechanism |
|---|---|---|
| **TRACLUS** | Phase-separated | MDL-based partition + DBSCAN clustering |
| **S2T-Clustering** | Quality-aware | Global voting algorithm based on local density |
| **SubCLUS** | Quality-aware | Set-cover formulation; approximation algorithms; NP-hard exact solution |
| **Greedy Method** | Degraded RLSTC | Segments at $p_t$ only if $OD_s < OD_n$ (local optimum only; no RL) |
| **CLUSTMOSA** | Multi-objective | Three clustering objectives + archived multi-objective simulated annealing |
| **HyperCLUS** | Hypercube-based | Segments into hypercubes; identifies clusters by hypercube intersection |
| **RTCLUS** | Distance-enhanced | Novel distance metric + R-tree index for efficiency |
| **RLSTC w/o simplification** | Ablation | Same method; MDL preprocessing omitted |

> Note: SubCLUS evaluated exclusively with $d_{\text{DF}}$ (leverages Fréchet-specific properties). TRACLUS evaluated on 1,000 trajectories only (generates ~33,168 sub-trajectories for 1,000 inputs — impractical at scale).

#### 1.8.3 Effectiveness Results (OD and SSE vs. Baselines)

| Dataset | Metric | RLSTC Avg. Improvement |
|---|---|---|
| T-Drive | OD | **36%** over all baselines |
| Geolife | OD | **39%** over all baselines |
| T-Drive | SSE | **57%** over all baselines |
| Geolife | SSE | **44%** over all baselines |

RLSTC outperforms across all four distance measurements ($d_{\text{DF}}$, $d_{\text{DTW}}$, $d_{\text{WD}}$, $d_{\text{IED}}$) and all tested datasets.

#### 1.8.4 Segmentation Method Cross-Evaluation (Paper §5.7)

RLSTC-S segmentation applied to AHC, DBSCAN, and k-means clustering algorithms:

| Method | OD Improvement vs. RLSTC-S |
|---|---|
| HyperCLUS-S | −46% (RLSTC-S better) |
| RTCLUS-S | −47% |
| CLUSTMOSA-S | −45% |
| TRACLUS-S | −44% |
| S2T-Clustering-S | −33% |
| SubCLUS-S | −11% |

RLSTC-S is adaptable to any clustering algorithm — density-based or center-based.

#### 1.8.5 Efficiency Results (Paper §5.6)

- RLSTC runs faster than TRACLUS, S2T-Clustering, SubCLUS, Greedy, and CLUSTMOSA by ≥20%
- On T-Drive: RLSTC outperforms TRACLUS by 37%, S2T-Clustering by 24%
- On Geolife: RLSTC outperforms SubCLUS by 32%, Greedy Method by 30%
- HyperCLUS and RTCLUS are slightly faster (use approximate representations — lower quality trade-off)
- RLSTC w/o simplification is much slower but still faster than TRACLUS on T-Drive and several methods on Geolife

---

### 1.9 Parameter Studies (Paper §5.8)

#### 1.9.1 Impact of $k$ (Cluster Count)

| $k$ | OD T-Drive | OD Geolife |
|---|---|---|
| 5 | 0.49 | 0.052 |
| 8 | 0.46 | 0.051 |
| **10** | **0.21** | **0.023** |
| 12 | 0.44 | 0.051 |
| 14 | 0.42 | 0.04 |

**Optimal $k = 10$ for both datasets.** Model is relatively robust to $k$ within ±50% of training $k$; degrades significantly beyond that.

#### 1.9.2 Pretrained Model Robustness to Different $k$ (trained at $k=10$)

| Test $k$ | 5 | 8 | 10 | 12 | 15 | 20 | 40 | 60 | 80 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|
| T-Drive OD | 0.25 | 0.22 | 0.21 | 0.26 | 0.26 | 0.51 | 0.59 | 0.61 | 0.60 | 0.62 |
| Geolife OD | 0.04 | 0.04 | 0.02 | 0.03 | 0.04 | 0.08 | 0.09 | 0.10 | 0.09 | 0.12 |

**Rule:** Retrain when actual $k$ differs from training $k$ by >50%.

#### 1.9.3 Small Sample Proxy Test (200 trajectories)

| Test $k$ | 5 | 8 | 10 | 12 | 15 | 20 | 40 | 60 | 80 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|
| T-Drive OD (small) | 0.14 | 0.19 | 0.18 | 0.18 | 0.19 | 0.32 | 0.34 | 0.34 | 0.35 | 0.37 |
| Geolife OD (small) | 0.03 | 0.03 | 0.02 | 0.03 | 0.04 | 0.06 | 0.07 | 0.08 | 0.08 | 0.11 |

Small samples produce similar trends → use small dataset to check model viability before full retraining.

#### 1.9.4 Training Data Size vs. Performance Trade-off

| Training Size | OD | Training Time | TR (trade-off metric) |
|---|---|---|---|
| 1,000 | Higher | Shorter | Higher |
| **2,000** | Lower | Moderate | **Minimum (optimal)** |
| 3,000 | Slightly lower | Longer | Higher |
| 4,000–5,000 | Stable | Much longer | Higher |

**Trade-off metric:**
$$TR = t_{\text{norm}} + OD, \quad t_{\text{norm}} = t \cdot \frac{OD}{\bar{t}}$$
where $t$ = training time, $\bar{t}$ = mean training time. TR minimized at 2,000 trajectories → default training set size.

#### 1.9.5 Number of Iterations

- OD fluctuates for first ~9 iterations, stabilizes after iteration 10
- Cluster center `maxdist` ($\tau$) drops sharply in early iterations, stabilizes after iteration 10
- OD stable when $\tau = 0.1$ → confirms default convergence threshold

#### 1.9.6 Impact of $OD_b$ (Expert Knowledge)

Including TRACLUS-derived $OD_b$ in state significantly improves clustering quality on both T-Drive and Geolife (Paper Fig. 13). Attributed to faster convergence from expert-guided state representation.

---

### 1.10 Related Work Taxonomy (Paper §6)

#### 1.10.1 Sub-trajectory Clustering — Two Research Categories

**Category 1: Phase-Separated (Segmentation independent of clustering)**

| Method | Segmentation Criterion | Clustering |
|---|---|---|
| TRACLUS | MDL principle | DBSCAN |
| S2T-Clustering | Local density voting | Density-based |
| SubCLUS | Pathlet set-cover (NP-hard; approximated) | — |
| CLUSTMOSA | Bearing measurements; turning-point detection | Multi-objective simulated annealing |
| HyperCLUS | Location, time, motion direction → hypercubes | Hypercube intersection |
| RTCLUS | Motion direction; novel distance metric | R-tree accelerated |

**Category 2: Quality-Aware (Clustering informs segmentation)**

| Method | Mechanism | Limitation |
|---|---|---|
| S2T-Clustering | Global voting → local density segmentation | Complex preprocessing; many predefined parameters; high cost |
| SubCLUS | Pathlet optimization | Relies on complex artificial rules; specific to Fréchet distance |
| **RLSTC** | RL agent guided by OD reward | **First RL-based; general across distance metrics** |

#### 1.10.2 Relevant RL Applications

| System | Problem | MDP State Design |
|---|---|---|
| RLS (Wang et al. 2020) | Sub-trajectory similarity search | Query-to-sub-trajectory similarity |
| RLTS (Wang et al. 2021) | Trajectory simplification | Point-drop error |
| RL R-tree (Gu et al. 2021) | Spatial index optimization | Node-split and subtree-insertion decisions |
| **RLSTC** | Sub-trajectory clustering | Cluster center distance + segment length features |

Key differentiator: RLSTC focuses on distance between cluster centers and trajectories — fundamentally different MDP from similarity search or simplification problems.

#### 1.10.3 Proposed Future Extensions (Paper §5.10 + §7)

1. **Grid index acceleration:** Map trajectory points to grid cells; approximate trajectory distance as distance between grid cell sequences; adjust cell size for efficiency/accuracy trade-off
2. **SKIP action:** Augment MDP with action $S$ that skips $p_{i+1}, \ldots, p_{i+S}$ when scanning $p_i$ — leverages fact that successive points often share similar motion trends; directly implemented in Q-RLSTC Version D
3. **Multi-modal extension:** Integrate spatial, temporal, and semantic trajectory features (NLP/semantic feature layers)
4. **Training acceleration:** Reduce wall-clock convergence time while preserving Agglomeration Degree integrity

---

## 2. Q-RLSTC — Quantum-Enhanced Extension

### 2.1 Motivation: Classical Bottlenecks

| Limitation | Classical RLSTC | Consequence |
|---|---|---|
| Parameter count | 514–1,314 trainable weights | Memory bloat; slow convergence |
| Optimization | Requires gradient access (SGD/Adam) | Infeasible without large differentiable dataset |
| Data requirement | 3,000+ trajectories for stable convergence | Impractical for edge/federated deployment |
| Model size | >100 KB serialized | Cannot run on low-power drone/mobile hardware |
| Privacy | Requires sending raw GPS to centralized server | Violates data privacy constraints |

---

### 2.2 Core Design Principle: NISQ Awareness

Current quantum hardware (NISQ — Noisy Intermediate-Scale Quantum) constraints:
- High systemic error rates
- Rapid environmental decoherence
- Coherence windows: ~100–200 μs

**Architectural response:** Extreme parameter efficiency + shallow circuit depths

---

### 2.3 VQ-DQN Architecture

Classical MLP replaced by **Variational Quantum Deep Q-Network (VQ-DQN)**:

```
Classical 5→64→2 MLP
  └── REPLACED BY:
      Hardware-Efficient Ansatz (HEA) Circuit
        ├── Qubits: 5–8 (version-dependent)
        ├── Variational layers: 2–3 (strictly limited)
        ├── Per-layer: RY + RZ single-qubit rotations
        └── Entanglement: Linear CNOT chain (not fully-connected or ring)
```

**Why linear CNOT over full/ring entanglement:**
- Minimizes two-qubit gate operations
- Reduces hardware noise and cross-talk errors
- Simplifies transpilation to physical quantum topologies
- Ensures execution within ~100–200 μs coherence windows

**Parameter compression:**

| Architecture | Parameters | Model Size |
|---|---|---|
| Classical RLSTC (base) | 514 | >100 KB |
| Classical RLSTC (deep) | 1,314 | ~200+ KB |
| VQ-DQN (5q × 3L) | **20–34** | **80–224 bytes** |

5-qubit circuit operates within 32-dimensional Hilbert space ($2^5$).

---

### 2.4 Data Encoding: Arctan Angle Encoding

Each feature $x_i$ encoded as:

$$RY(2 \cdot \arctan(x_i))$$

**Why $\arctan$ specifically:**
- GPS features (segment lengths, spatial variances, curvature gradients) have unbounded natural ranges
- $\arctan$ continuously maps $(-\infty, +\infty) \to [-\pi, \pi]$ without overflow
- Sign-preserving (positive distances → positive phase angles)
- Graceful saturation: extreme spatial outliers do not cause numerical instability
- Features encoded independently → no global normalization preprocessing required
- Bounded quantum measurement range + trigonometric activation = **massive implicit regularization**
  - Eliminates need for dropout layers or normalization matrices

---

### 2.5 Data Re-Uploading

Standard VQCs suffer from limited expressivity. Solution without deepening circuit (which triggers decoherence):

**Mechanism:** Between internal HEA variational layers, classical input state is continuously re-encoded into the system. Subsequent rotational layers process identical input from differing entangled perspectives.

**Effect:** Drastically enhances trainability and function approximation of universal quantum classifiers without increasing qubit count.

---

### 2.6 Readout Schemas

| Version | Schema | Mechanism | Use Case |
|---|---|---|---|
| **A** | Standard Z-expectation | $\langle Z_i \rangle$ from measurement counts: $P(0) - P(1)$ | Classical parity / baseline |
| **B** | Multi-Observable | Parity observables $\langle Z_a Z_b \rangle$; leverages CNOT-generated entanglement correlations | Quantum-enhanced spatial model |
| **C** | Softmax Distribution | Raw quantum expectations → smooth probability distributions | Actor-Critic / stochastic selection |

---

### 2.7 Optimization: SGD vs. SPSA

| Property | Classical SGD / Adam | Parameter Shift Rule | SPSA |
|---|---|---|---|
| Requires differentiable graph | Yes | No | No |
| Circuit evaluations per step | N/A | $2 \times N_{params}$ | **2 (constant)** |
| Viable on NISQ hardware | No | Theoretically yes, practically no | **Yes** |
| 34-param model evaluations/step | N/A | 68 | 2 |
| Complexity scaling | — | Linear in params | **O(1)** |

**SPSA:** Simultaneously perturbs all parameters in random directions to estimate full gradient vector.

**m-SPSA (Momentum SPSA):** Integrates Exponential Moving Average (EMA) momentum term. Smooths volatile gradients from quantum measurement shot noise. Trades slightly delayed convergence for greatly improved training stability.

---

### 2.8 Replay Buffer and Target Network: Classical vs. Quantum

| Property | Classical RLSTC | Q-RLSTC |
|---|---|---|
| DQN type | Single DQN | **Double DQN** |
| Target network update | Soft-update (Polyak, τ=0.05 per batch) | **Periodic hard copies** (complete param override) |
| Rationale | Gradual blending sufficient for classical weights | Soft-updating rotational angles destabilizes quantum state |
| Replay buffer size | 2,000 (Python deque) | **5,000** (Python `@dataclasses`) |
| Buffer implementation | Python dictionaries | `@dataclasses` with `to_array()` + type-checked distance verification |

---

### 2.9 Distance Computation: Classical vs. Quantum Boundary

| Stage | Classical RLSTC | Q-RLSTC |
|---|---|---|
| During MDP exploration | Full IED computation at every CUT action | **OD Proxy** (`od_segment` / `od_continue`) — lightweight directional heuristic |
| End of training episode | — | Full K-means + IED verification |
| Decision-making (policy) | Classical MLP | **Fully quantum VQC** |
| Geometric feature extraction | Classical | **Classical only** (prohibited from quantum offloading) |
| IED distance queries | Classical | **Classical only** |

**Why prohibit geometric computation on quantum processor:**
- Simple continuous arithmetic on quantum circuit = zero algorithmic advantage
- Re-encoding cost: estimated ~100,000× latency slowdown vs. classical CPU

**Quantum Swap Test (optional research probe):**
- Uses controlled-SWAP gate between data registers bounded by Hadamard gates
- Estimates inner products between amplitude-encoded trajectory states
- Outputs: $P(0) = 0.5 \times (1 + |\langle\psi|\phi\rangle|^2)$
- Adds ~10 sec simulation latency per episode; requires thousands of shots for ±0.01 precision
- **Not used in training loop** — classical distance remains primary

---

## 3. Q-RLSTC Architectural Taxonomy (Versions A–D)

### 3.1 Version A — Classical Parity / Scientific Control

| Property | Specification |
|---|---|
| Purpose | 1:1 isolation of quantum function approximator vs. classical MLP |
| Qubits | 5 (mirrors classical 5D feature space) |
| Variational layers | 2 |
| Parameters | 20 |
| Readout | Standard Z-expectation |
| Similarity metric | MDL baseline compression score (`baseline_cost`) — replaces volatile "minimum similarity" metric |
| Distance | Proxy-based projected OD (no full incremental IED recalculation) |

---

### 3.2 Version B — Quantum-Enhanced Spatial Model

| Property | Specification |
|---|---|
| Purpose | Probe true quantum utility and high-dimensional expressiveness |
| Qubits | 8 |
| Parameters | 32 |
| Input dimensions | 8 (expanded beyond classical 5D) |
| Readout | Multi-Observable ($\langle Z_a Z_b \rangle$) |
| Entanglement | 7-CNOT hardware-efficient layout |

**Three quantum-native spatial signals added:**

| Signal | Definition |
|---|---|
| Angle Spread | Spatial variance of arctan-encoded features across Bloch sphere |
| Curvature Gradient | Second-order derivative tracking acute trajectory turning changes |
| Segment Density | Geographic congestion points per unit distance (urban grid analytics) |

---

### 3.3 Version C — Q-RNN and Shadow Memory

| Property | Specification |
|---|---|
| Purpose | Quantum recurrent temporal memory without expanding classical feature overhead |
| Qubits | 6 (5 data + 1 shadow qubit) |
| Shadow qubit | Persists entangled quantum state across sequential time steps; carries temporal amplitude forward |
| Circuit type | Equivariant Quantum Circuit (EQC) with SO(2)-equivariant gates |
| Topology | Circular CNOT (respects rotational symmetry of GPS coordinate data) |
| Exploration | Soft Actor-Critic (SAC) with entropy regularization (replaces ε-greedy) |
| Action space | **Ternary:** EXTEND / CUT / **DROP** |
| DROP action | Filters and permanently discards GPS noise anomalies in real-time; guided by DROP anomaly reward bonus |
| Shot management | Adaptive: 32-shot burst (high-confidence decisions) → 512 shots (high-uncertainty decisions) |

---

### 3.4 Version D — Strict VLDB Alignment + SKIP Optimization

| Property | Specification |
|---|---|
| Purpose | Direct 1:1 mathematical reproduction of 2024 VLDB classical RLSTC paper |
| Input state | Exact paper equations: $(OD_s, OD_n, OD_b, L_b, L_f)$ — no proxies |
| Reward function | Exact classical: $OD(s_t) - OD(s_{t+1})$ |
| Parameters | 30 |
| Layers | 3 |
| Novel addition | **SKIP(S) action** |

**SKIP(S) action:**
- Fast-forwards through low-variance, straight-path data points (e.g., highway segments)
- Reward: $+0.05 \times S$ (linear SKIP reward)
- Eliminates quantum circuit evaluations for geometrically uninformative coordinates
- Saves significant quantum compute budget on long-haul trajectories

---

## 4. Advanced Reward Engineering (Q-RLSTC)

| Signal | Formula / Value | Purpose |
|---|---|---|
| Boundary Sharpness | $+0.5 \times (\text{angle}/\pi)$ | Incentivizes CUT at acute geographic turns |
| Over-segmentation Penalty | $-0.12$ per segment created (`CUT_PENALTY`) | Forces cost-benefit analysis on every cut |
| Single-cluster collapse | $-2.0$ | Prevents degenerate all-in-one clustering |
| Empty cluster penalty | $-1.0$ | Prevents K-means empty-cluster failure |
| SKIP reward | $+0.05 \times S$ | Incentivizes bypassing low-variance linear segments |
| DROP reward | Anomaly bonus | Incentivizes real-time GPS noise filtering (Version C) |

---

## 5. Metric Pathology: The Fragmentation Attractor

**The Bug:** Raw Validation Competitive Ratio (Val CR) has a denominator vulnerability.

**Mechanism:**
- IED geometrically scales with segment length
- More cuts → shorter segments → lower IED → better Val CR **regardless of cluster quality**
- Agent can game the metric by hyper-fragmenting without producing meaningful clusters

**Empirical verification (Geolife, `master_results.csv`):**

| Cut Budget | Realized Cut | N_Segments | Avg_Seg_Length | SSE |
|---|---|---|---|---|
| 0.0 | 0.0 | 50 | 398.3 | 0.1367 |
| 0.02 | 0.0199 | 444 | 45.74 | 1.2061 |
| 0.1 | 0.0855 | 1,736 | 12.44 | 4.7208 |
| 0.4 | 0.2194 | 4,385 | 5.53 | 12.0870 |
| 0.5 | 0.2480 | 4,948 | 5.01 | 13.6238 |

**Fix:** Budget-constrained diagnostic model
- Plots Pareto frontier of CR metrics against forced CUT thresholds (5%, 10%, 20%, 50%)
- Evaluates length-weighted CR variant (`wValCR`) to prevent gaming

---

## 6. Empirical Benchmarking: Classical vs. Quantum

### 6.1 Evaluation Protocol

- 90/10 deterministic data split via custom `TrajectoryScheduler`
- Identical trajectory order and exploration paths for classical controls and quantum experiments
- Statistical validation: Mann-Whitney U tests, Cohen's d effect sizes, bootstrap 95% CI (10,000 resamples)

### 6.2 Full Results Table

| Rank | Model | Architecture | Params | Val CR | Optimizer | Training Data |
|---|---|---|---|---|---|---|
| 1 | RLSTCcode (best, 3k) | Classical RLSTC | 514 | **0.5892** | SGD (Backprop) | 3,000 trajectories |
| 2 | RLSTCcode (5-fold CV) | Classical RLSTC | 514 | 0.7543 ± 0.0369 | SGD (Backprop) | ~4,000 trajectories |
| 3 | **VQ-DQN (5q × 3L)** | **Quantum Q-RLSTC** | **34** | **1.4811** | **SPSA** | **30 trajectories** |
| 4 | RLSTCcode (modelstate4) | Classical RLSTC | 514 | 1.5453 | SGD (Backprop) | Unknown |
| 5 | Control C (h=32×32) | Classical Control | 1,314 | 1.6390 | SPSA | 30 trajectories |
| 6 | Control B (h=64) | Classical Control | 514 | 5.1133 | SPSA | 30 trajectories |
| 7 | Control A (linear) | Classical Control | 12 | 9.2051 | SPSA | 30 trajectories |

> **Lower CR = better clustering.** Note: Ranks 1–2 vs. 3+ are not directly comparable — see iso-condition experiment below.

### 6.3 Iso-Condition Experiment (Fair Comparison)

**Controlled variables:** SPSA optimizer, 30 training trajectories, 2 epochs

| Model | Params | Val CR | Cut Rate | Outcome |
|---|---|---|---|---|
| **VQ-DQN (5q)** | **34** | **1.4811** | 33.2% | ✅ Learned valid policy |
| Control C (Deep MLP) | 1,314 | 1.6390 | 3.9% | ❌ Outmatched by 39× smaller quantum model |
| Control B (Med MLP) | 514 | 5.1133 | 0.2% | ❌ Catastrophic 8.7× degradation |
| Control A (Linear) | 12 | 9.2051 | 0.0% | ❌ Complete failure |

**Key finding:** Classical MLPs require gradient access (SGD) + massive datasets to function. Under gradient-free constraints, quantum circuit's inductive bias for spatial geometry is architecturally superior.

### 6.4 Hardware Simulation Viability

**Simulation:** Qiskit Aer statevector simulation

**Noise testing against real IBM hardware profiles:**

| Processor | Coherence Time ($T_2$) | Projected Fidelity (Versions A, B, D) |
|---|---|---|
| IBM Eagle | ~100 μs | 80%–85% |
| IBM Heron | ~200 μs | **90%–95%** |

Circuit depth-11 shallow HEA proved robust against IBM Eagle noise floor. Custom readout calibration matrices + momentum SPSA shot-noise resilience = practically viable on near-term NISQ hardware.

---

## 7. SMART DShot — RLSTC in UAV Cyber-Physical Security

### 7.1 Attack Surface

| Component | Vulnerability | Attack Vector |
|---|---|---|
| Electronic Speed Controller (ESC) | Cannot run heavy cryptographic algorithms without crashing flight loop | — |
| DShot protocol | Communication timings between ESC and rotors | Microsecond timing anomaly injection |
| Standard hash-based protection | Validates data payload only | **Zero-day timing attacks bypass payload integrity** |

**Attack outcome:** Minute, undetectable timing anomalies → destabilized motor operations → bypass data-integrity alarms → catastrophic hardware failure

---

### 7.2 SMART DShot Framework — Four-Algorithm Synergy

| Algorithm | Acronym | Optimized For |
|---|---|---|
| Kalman Filter Timing Correction | KFTC | Linear, Gaussian-noise-dominated timing drift |
| **Recursive Least Squares Timing Correction** | **RLSTC** | Sudden, non-linear timing anomalies; recursive minimization of squared timing errors |
| Fuzzy Logic Timing Correction | FLTC | Highly ambiguous, erratic fluctuations defying rigid statistical modeling |
| Hybrid Adaptive Timing Correction | HATC | Meta-learning apex; evaluates incoming noise profile → dynamically routes to optimal sub-algorithm in real-time |

**RLSTC (UAV):** Recursively minimizes sum of squared microsecond timing errors → highly aggressive, instantaneous flight corrections to stabilize drone

---

### 7.3 Hardware Implementation

**Platform:** PolarFire SoC FPGA
- True parallel hardware processing
- Bypasses software OS latency overhead

**Validation Protocol:** 32,000 Monte Carlo test iterations × 16 distinct operational threat matrices (500 tests/scenario × 4 algorithms)

---

### 7.4 Performance Metrics

| Metric | Result | Operational Implication |
|---|---|---|
| Attack Detection Rate | **88.3%** | High resilience against DShot timing manipulation |
| False Positive Incidence | **2.3%** | Minimal disruption to standard flight operations |
| Execution Latency | **< 10 μs** (deterministic) | Ensures absolute motor control stability |
| FPGA LUT Utilization | **2.16%** | Negligible logic gate overhead on PolarFire SoC |
| Power Consumption | **16.57 mW/channel** | Viable for battery-constrained autonomous edge devices |
| CVSS Severity Reduction | **7.3 → 3.1** ("High" → "Low") | Successful vulnerability class downgrade |
| HATC Correction Success Rate | **31.01%** (95% CI: [30.2%, 31.8%]) | 26.5% improvement over baseline linear approaches |

**Statistical validation:** ANOVA — $F(3, 1996) = 30.30$, $p < 0.001$, Cohen's $d = 4.57$

---

## 8. Alternative RLSTC Contexts

### 8.1 Telecommunications — Ericsson GSM/LTE

| Property | Detail |
|---|---|
| Context | Legacy Ericsson GSM and LTE network administration |
| Command | `rlstc:cell=cellID,chgr=0,state=active;` |
| Function | Physically activates designated cellular broadcasting cells across the Abis interface |
| Related commands | `rldec` (define cell properties), `rxble` (block TRUs for maintenance), `rlsli` (enable alarm dictionaries — e.g., "ABIS PATH UNAVAIL") |

---

### 8.2 Cheminformatics — QSAR Realistic Splits

| Property | Detail |
|---|---|
| Context | Quantitative Structure-Activity Relationship (QSAR) model evaluation |
| Identifier | `rlstca` — "Realistic Splits" |
| Databases | PubChem, ChEMBL, NVS |
| Application | Partitions non-random training/test data to respect structural distribution |

**Key finding:** PubChem kinase pQSAR models returned median $R^2 = 0.16$ vs. ChEMBL ($R^2 = 0.40$) and NVS ($R^2 = 0.59$). Failure cause: not data sparsity (both datasets share 99% missing value rate) — but **fundamental inter-assay structural overlap deficit**. PubChem compounds tested across only 1.9 assays on average vs. ChEMBL (8.7) and NVS (8.4). Disrupts the similarity principle QSAR depends on.

---

## 9. Trajectory Representation Extensions

### 9.1 CLMTR — Contrastive Multi-Modal Trajectory Representation

*(Published: GeoInformatica, 2024 — same research group as RLSTC spatial)*

| Component | Function |
|---|---|
| **Problem** | Standard trajectory mining uses spatio-temporal data only; ignores textual check-ins, imagery, POI data |
| **Intra-trajectory mechanism** | Captures correlations between diverse data types within a single path |
| **Inter-trajectory mechanism** | Maps macroscopic relationships among differing trajectories |
| **Fusion** | Attention-based; dynamically weights modality importance by spatial context |
| **Output** | Low-dimensional, multi-modal embeddings that outperform unimodal baselines on downstream tasks |

---

### 9.2 Supporting Ecosystem

| Framework | Function | Repository |
|---|---|---|
| PGTuner | Automatic, transferable configuration tuning for Proximity Graphs | `github.com/hao-duan/PGTuner` |
| UNIFY Index (SIG) | Range-filtered approximate nearest neighbor search; partitions datasets by attribute value | — |
| Ganos Aero | Cloud-native big raster dataset management | — |
| STAR (Stream Warehouse) | Cache-based handling of mobile location-based data streams | — |
| Functional Trajectories (Rcpp) | Bayesian nonparametric clustering of functional trajectories over time | `github.com/mliang4/ClusteringFunctionalTrajectoriesOverTime` |

---

## 10. Limitations and Future Research Vectors

### 10.1 Classical / Q-RLSTC Spatial

| Limitation | Detail |
|---|---|
| Computational scaling | $O(nN)$ complexity remains restrictive at planetary GPS data scales |
| Static convergence thresholds | Fixed K-means deviation threshold → premature termination or excessive computation depending on dataset volatility |
| Validation metric fragility | Raw Val CR susceptible to fragmentation attractor; requires budget-constrained evaluation |
| Data requirements (classical) | Requires thousands of trajectories for stable SGD convergence |
| NISQ fidelity ceiling | 80–85% fidelity on IBM Eagle; 90–95% on IBM Heron — not yet production-grade |

**Future directions:**
- Cloud-native validation at billion-scale GPS datasets
- Dynamic convergence thresholds adaptive to dataset density
- NLP module integration for semantic feature extraction (eliminates manual tagging)
- Hypergraph-Driven High-Order Knowledge Tracing (HGKT) architectures for high-order trajectory correlation
- Federated deployment of Q-RLSTC on mobile edge devices

### 10.2 SMART DShot / UAV Security

| Limitation | Detail |
|---|---|
| Threshold-based detection | Slow-drift and sophisticated zero-day anomalies over long time horizons can evade fixed thresholds |
| Attack signature dependency | RLSTC optimized for known non-linear patterns; novel attack vectors may require retraining |

**Future directions:**
- Transformer-based architectures for long-term dependency modeling in frequency drift patterns
- SD-TCXO (Software-Defined Temperature-Compensated Crystal Oscillators) disciplined by Transformer models for predictive timing stabilization
- Container-based virtualization for industrial control system isolation against DoS attacks
- True predictive early threat detection in volatile FANET (Flying Ad-hoc Network) environments

---

## 11. Cross-Domain Convergence Summary

| Dimension | Classical RLSTC | Q-RLSTC | SMART DShot RLSTC |
|---|---|---|---|
| Domain | Spatial GPS analytics | Quantum spatial analytics | UAV ESC security |
| Core mechanism | DQN + K-means feedback | VQ-DQN (HEA) + K-means feedback | Recursive LS timing error minimization |
| Optimizer | SGD / Adam (backprop) | SPSA / m-SPSA (gradient-free) | Adaptive meta-routing (HATC) |
| Parameter count | 514–1,314 | **20–34** | N/A (algorithm-level) |
| Data requirement | 3,000+ trajectories | **30 trajectories** | Real-time stream |
| Hardware target | GPU / CPU server | NISQ quantum + CPU classical hybrid | FPGA (PolarFire SoC) |
| Action space | Binary (EXTEND / CUT) | Binary + optional SKIP / DROP | Continuous correction magnitude |
| Convergence guarantee | Yes (cluster center mathematics) | Empirically demonstrated | Deterministic sub-10μs latency |
| Key advantage | Eliminates heuristic segmentation brittleness | 90%+ parameter compression vs. classical | CVSS 7.3 → 3.1 vulnerability reduction |

---

## 12. Forensic Codebase Audit — RLSTCcode-main

### 12.1 Audit Overview

Independent forensic review of the published `RLSTCcode-main` repository (VLDB 2024) revealed **16 distinct implementation defects** — ranging from silent metric fabrication to structurally broken RL mechanics. The published performance claims (36–39% OD improvement, 44–57% SSE improvement over baselines) cannot be reproduced or trusted from the submitted codebase. Defects are categorized below by severity and component.

> **Thesis framing:** The Q-RLSTC project is not merely a quantum-inspired enhancement over a functioning classical baseline. It is the first mathematically sound implementation of the RLSTC theoretical framework — the original codebase was a defective proof-of-concept incapable of genuine policy convergence.

---

### 12.2 Critical / High-Severity Defects

#### ERROR-01 — Medoid Fallback ("Fake Centroid")
**File:** `initcenters.py`
**Paper claim (Section 4.4):** "We determine the average coordinate at each timestamp to generate a representative trajectory... We derive synchronous points through linear interpolation."
**Code reality:** Averaging varying-length temporal sequences caused geometric degeneration (clusters collapsing to a stationary point). Authors silently replaced the centroid computation with a `max_span` heuristic — selecting the single longest existing sub-trajectory in the bucket as the "center."

**Consequences:**
- **Figure 16 is visually fabricated.** The "representative trajectories" plotted over Beijing / Geolife maps are the longest raw GPS tracks that already existed in the dataset — not computed centers of mass. Clusters containing hundreds of short, disjointed segments look smooth and map-spanning because one long outlier trajectory was used as the representative.
- **OD metric is invalidated.** Overall Distance is supposed to measure compactness (distance from each segment to the true center of mass). With a boundary-outlier as the "center," OD no longer measures variance — it measures distance to an extreme edge case. The RL agent's reward signal (Δ OD) was therefore optimizing for alignment with an outlier.
- **The published ablation studies are meaningless** insofar as they test the impact of state features on a reward signal that was already mathematically broken.

```
TRUE centroid: geometric mean of all sub-trajectory coordinates at each timestamp
PUBLISHED centroid: argmax(segment_length) within each cluster bucket
```

---

#### ERROR-02 — Static Global `basesim` Denominator
**File:** `crossvalidate.py`
**Defect:** The Competitive Ratio (CR) metric is computed as `model_OD / basesim`. Rather than calculating `basesim` per fold from the current validation split, the code loads a single static scalar file globally. This scalar is reused unchanged across all cross-validation folds.
**Impact:** The CR denominator is completely decoupled from the actual data being validated. The metric is measuring against a phantom baseline — all reported cross-validation scores are synthetic.

---

#### ERROR-03 — Fabricated TRACLUS Expert State (ODb)
**File:** `MDP.py`
**Paper claim:** Injects TRACLUS segmentation OD into the state vector as "expert knowledge" to expedite convergence. An ablation study (Figure 13) is published demonstrating its impact.
**Code reality:** `ODb = overall_sim * 10`. The code multiplies the agent's own current OD by 10 and passes it off as an independent TRACLUS feature.
**Impact:** The state receives a linearly dependent scalar — zero new information is injected. The Figure 13 ablation study is measuring the impact of a constant multiplier on the state, not genuine expert knowledge. The published ablation is fabricated.

---

#### ERROR-04 — Severely Undertrained Agent
**File:** `rl_train.py`
**Defect:** Training loop is hardcoded to `Round = 1` or `Round = 2` epochs over 500 trajectories.
**Impact:** An RL network trained for 1–2 epochs has barely initialized its weights, let alone converged to an optimal policy. The paper describes convergence behavior and stable learned policies that are mathematically impossible under these training conditions.

---

#### ERROR-08 — Hardcoded SSE Denominator Shrinkage
**File:** `crossvalidate.py`
**Defect 1:** SSE denominator hardcoded as `2 * num * cluster_size` instead of `2 * cluster_size`, where `num` is an arbitrary multiplier (≈10). This shrinks all reported SSE values by ~1 order of magnitude.
**Defect 2:** `dist_square` not re-initialized inside the inner loop. When two trajectories have no temporal overlap (`dist == 1e10`), the condition skips re-assignment but does not reset `dist_square` — the previous iteration's distance leaks into the accumulator.
**Impact:** All published SSE values are artificially deflated. The 44–57% SSE improvement claims cannot be reproduced.

---

#### ERROR-09 — Deterministic Farthest-First K-Means++ (Not Probabilistic)
**File:** `initcenters.py`
**Paper claim:** Uses K-Means++ initialization.
**Code reality:** Uses `max(distances)` farthest-first traversal — a deterministic greedy approach, not probabilistic $D(x)^2$ weighted sampling. Susceptible to outlier sensitivity and reproducibility failure.

---

#### ERROR-11 — Sparse Reward Signal (Zero for EXTEND)
**File:** `MDP.py`
**Defect:** Reward for `action == 0` (EXTEND) is hardcoded to `0`. Only CUT actions receive a non-zero reward.
**Impact:** The agent receives no learning signal for the majority of its decisions. With a dense trajectory of hundreds of points and sparse CUT actions, the agent navigates an almost entirely silent reward landscape. Policy convergence under these conditions is severely impaired.

---

### 12.3 Medium-Severity Defects

| Error ID | File | Defect | Impact |
|---|---|---|---|
| **ERROR-05** | `rl_train.py` | Soft update called with τ=0.05 **inside the batch loop** (should be τ=0.001 at episode end) | Target network updates far too aggressively and too frequently; destroys training stability |
| **ERROR-06** | `rl_train.py` | Epsilon decay executes inside `replay()` function (per memory batch), not per trajectory step | Exploration collapses much faster than intended; agent stops exploring before learning good cuts |
| **ERROR-07** | `rl_train.py` | Single 90/10 static data split used for all "cross-validation" | No genuine K-fold validation; single split results reported as if cross-validated |
| **ERROR-10** | `initcenters.py` | Empty cluster re-seeding uses `cluster_segments[minidx]` instead of `cluster_segments[i]` | Empty clusters dump their seeds into the last-accessed cluster index — "Bucket Zero Leak" |
| **ERROR-12** | `cluster.py` | `add2clusdict` / `computecenter` maintain running sum-of-sums rather than per-timestamp coordinate lists | Coordinate averaging mathematically corrupts centroid values across iterations |
| **ERROR-13** | `preprocessing.py` | `normloctrajs()` spatial normalization never called on the dataset before training | IED calculations massively over-weight geographic (lat/lon) axes vs. temporal; distance metric is geometrically distorted |
| **ERROR-15** | `rl_nn.py` | Custom Huber loss implementation used; should be standard MSE per DQN convention | Non-standard loss creates asymmetric gradient behavior inconsistent with DQN theory |
| **ERROR-16** | `rl_train.py` | Final model loaded via `modelnames[0]` (alphabetically first) rather than tracking best validation CR | Production model is not necessarily the best-performing checkpoint |

---

### 12.4 Low-Severity Defects

| Error ID | File | Defect | Impact |
|---|---|---|---|
| **ERROR-14** | `point.py` | `equal(self, other)` method has no `return False` at bottom | Implicitly returns `None` on inequality; downstream boolean checks silently fail |

---

### 12.5 Compound Pathology Map

```
Published VLDB 2024 Claims
  │
  ├── "36–39% OD improvement"
  │     ├── Invalidated by ERROR-01 (OD measured against outlier medoid, not centroid)
  │     ├── Invalidated by ERROR-11 (agent never learned a real policy)
  │     └── Invalidated by ERROR-04 (1–2 epoch training incapable of convergence)
  │
  ├── "44–57% SSE improvement"
  │     ├── Invalidated by ERROR-08 (SSE denominator artificially shrinks values ~10×)
  │     └── Compounded by ERROR-02 (CR baseline is a static phantom scalar)
  │
  ├── "TRACLUS expert knowledge accelerates convergence" (Figure 13 ablation)
  │     └── Fabricated by ERROR-03 (ODb = overall_sim × 10, zero new information)
  │
  └── "Figure 16 — Representative cluster visualizations"
        └── Fabricated by ERROR-01 (max_span trajectory cherry-pick, not computed center)
```

---

### 12.6 Training Performance Bottleneck (Practical)

Separate from logical errors, the original training loop runs at approximately **17 seconds per trajectory** (~14+ hours for a full dataset run) because the optimizer is called after every individual MDP step.

**Fix:** Increase batch size; move optimizer update to execute after every episode (or every 3–4 steps) rather than every step. This also resolves the ERROR-05 soft-update frequency issue as a side effect.

**Environment compatibility:** Original codebase requires Linux + Python 3.6 + TensorFlow 1.x. Will not run on macOS or Python 3.7+. Attempting to recreate Figure 16 on a modern stack will fail silently due to dependency incompatibilities before any of the above errors are reached.

---

## 13. REAL_RLSTC_FIXED — Remediation Implementation

### 13.1 Remediation Philosophy

The remediation applies fixes strictly to **implementation violations** of the paper's theoretical claims — not to the underlying theoretical framework itself. Two deliberate preservation decisions:
1. The original authors' random trajectory truncation (cap at 500 points, discard < 10 points) is retained
2. The unnormalized IED equation is preserved

This maintains a strict 1:1 empirical match with the publication's theoretical scope. Fixing foundational theoretical choices would constitute proposing a new algorithm rather than validating the existing one.

---

### 13.2 Algorithmic Fixes — Code

#### True K-Means++ + Geometric Centroid (`initcenters.py`)

```python
# initcenters.py (Remediated)
import numpy as np
import random
from collections import defaultdict
from trajdistance import traj2trajIED

def initialize_centers(data, K):
    """FIX ERROR-09: True K-Means++ probabilistic seeding."""
    centers = [random.choice(data)]
    while len(centers) < K:
        distances = []
        for traj in data:
            min_dist = min([traj2trajIED(center.points, traj.points) for center in centers])
            distances.append(min_dist)

        squared_distances = np.array(distances) ** 2
        total_dist = np.sum(squared_distances)

        # Probability distribution weighted by D(x)^2
        probs = [1.0 / len(data)] * len(data) if total_dist == 0 else squared_distances / total_dist
        new_center_idx = np.random.choice(len(data), p=probs)
        centers.append(data[new_center_idx])
    return centers

def getbaseclus(trajs, k, subtrajs):
    centers = initialize_centers(trajs, k)
    cluster_segments = defaultdict(list)

    for i in range(len(subtrajs)):
        mindist = float("inf")
        minidx = 0
        for j in range(k):
            dist = traj2trajIED(centers[j].points, subtrajs[i].points)
            if dist < mindist:
                mindist = dist
                minidx = j
        if mindist != float("inf") and dist != 1e10:
            cluster_segments[minidx].append(subtrajs[i])

    # FIX ERROR-10: Assign empty cluster seeds to own index i (not leaked minidx)
    for i in range(k):
        if len(cluster_segments[i]) == 0:
            cluster_segments[i].append(centers[i])

    cluster_dict = defaultdict(lambda: [[], None, [], {}])
    for i in range(k):
        cluster_dict[i][0] = []
        cluster_dict[i][2] = centers[i]

    return cluster_dict, cluster_segments
```

#### True Coordinate Averaging (`cluster.py`)

```python
# cluster.py (Remediated)
import numpy as np
from point import Point

def add2clusdict(points, clus_dict, k):
    """FIX ERROR-12: Store pure coordinates per timestamp, not sum-of-sums."""
    for i in range(len(points)):
        curr_t = points[i].t
        if curr_t not in clus_dict[k][3]:
            clus_dict[k][3][curr_t] = {'x': [points[i].x], 'y': [points[i].y]}
        else:
            clus_dict[k][3][curr_t]['x'].append(points[i].x)
            clus_dict[k][3][curr_t]['y'].append(points[i].y)

def computecenter(clus_dict, k, threshold_num, threshold_t):
    """FIX ERROR-12: Independent mean alignment per timestamp."""
    keys = sorted(clus_dict[k][3].keys())
    center = []
    for key in keys:
        if len(clus_dict[k][3][key]['x']) >= threshold_num:
            aver_x = np.mean(clus_dict[k][3][key]['x'])
            aver_y = np.mean(clus_dict[k][3][key]['y'])
            center.append(Point(aver_x, aver_y, key))
    return center
```

---

### 13.3 MDP & Reward Fixes — Code

#### Dense Reward + Authentic ODb (`MDP.py`)

```python
# MDP.py (Remediated step function)
def step(self, episode, action, index, mode):
    # FIX ERROR-03: Inject authentic TRACLUS baseline (self.basesim_T),
    #               not overall_sim * 10
    st_ODb = self.basesim_T if mode == 'T' else self.basesim_E

    last_overall_sim = self.overall_sim
    new_overall_sim = self.compute_current_od()

    # FIX ERROR-11 (1e10 numeric leak): Cap OD to prevent float explosion in DQN
    self.overall_sim = min(new_overall_sim, 100.0)

    # FIX ERROR-11 (sparse reward): Dense telescoping signal for ALL actions
    reward = last_overall_sim - new_overall_sim

    next_state = (self.ODs, self.ODn, st_ODb, self.Lb, self.Lf)
    return next_state, reward
```

#### Standard MSE Loss (`rl_nn.py`)

```python
# rl_nn.py (Remediated)
from tensorflow.keras.losses import MeanSquaredError

def _build_model(self):
    # ... layer definitions ...
    # FIX ERROR-15: Native Keras MSE replaces custom Huber implementation
    model.compile(loss=MeanSquaredError(), optimizer=Adam(lr=self.learning_rate))
    return model
```

---

### 13.4 Training Loop & Validation Fixes — Code

#### Convergence-Gated K-Fold with Best-Model Checkpointing (`rl_train.py`)

```python
# rl_train.py (Remediated)
import numpy as np
from sklearn.model_selection import KFold

def train(amount):
    dataset = list(range(amount))

    # FIX ERROR-07: True N-Fold validation matrix
    kf = KFold(n_splits=5, shuffle=True, random_state=1)

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        best_val_cr = float('inf')
        patience = 20
        patience_counter = 0

        # FIX ERROR-04: Convergence-gated loop (replaces hardcoded Round = 1/2)
        while patience_counter < patience:
            np.random.shuffle(train_idx)

            for idx in train_idx:
                # ... episode evaluation ...

                # FIX ERROR-06: Epsilon decays per trajectory step, not per batch
                RL.epsilon = max(RL.epsilon_min, RL.epsilon * RL.epsilon_decay)

            # FIX ERROR-05: Soft update at episode tier, τ = 0.001 (not 0.05 per batch)
            RL.update_target_model()

            val_cr = evaluate(val_idx)
            if val_cr < best_val_cr:
                best_val_cr = val_cr
                # FIX ERROR-16: Save best checkpoint, not alphabetically first model
                RL.model.save_weights(f'best_model_fold_{fold}.h5')
                patience_counter = 0
            else:
                patience_counter += 1
```

#### Corrected SSE Computation (`crossvalidate.py`)

```python
# crossvalidate.py (Remediated)
def compute_sse(cluster_segments):
    sse_sum = 0
    for k in cluster_segments.keys():
        cluster_size = len(cluster_segments[k])
        if cluster_size == 0:
            continue

        cluster_dist_sum = 0
        for i in range(cluster_size):
            for j in range(cluster_size):
                dist = traj2trajIED(cluster_segments[k][i].points,
                                    cluster_segments[k][j].points)
                # FIX ERROR-08: Initialize dist_square inside condition; no residue leak
                if dist != 1e10:
                    dist_square = dist ** 2
                    cluster_dist_sum += dist_square

        # FIX ERROR-08: Correct denominator — 2 * |Ci|, not 2 * 10 * |Ci|
        denominator = 2 * cluster_size
        sse_sum += cluster_dist_sum / denominator

    return sse_sum
```

---

### 13.5 Remediation Completion State

| Component | Original State | Fixed State |
|---|---|---|
| Cluster centroid | `max_span` medoid outlier | True geometric mean (linear-interpolated, timestamp-aligned) |
| K-Means++ seeding | Deterministic farthest-first | Probabilistic $D(x)^2$ weighted sampling |
| Empty cluster assignment | Leaks to `minidx` ("Bucket Zero") | Correctly assigns to own index `i` |
| Coordinate averaging | Running sum-of-sums (corrupted) | Per-timestamp independent mean |
| TRACLUS state feature | `overall_sim × 10` | Authentic pre-computed TRACLUS OD |
| Reward signal (EXTEND) | Hardcoded `0` | Dense `last_OD - new_OD` telescoping signal |
| 1e10 numeric leak | Raw infinity enters DQN state | Ceiling-bracketed at 100.0 |
| Loss function | Custom Huber (non-standard) | Keras native MSE |
| Soft update | τ = 0.05, per-batch | τ = 0.001, per-episode |
| Epsilon decay | Per memory batch (in `replay()`) | Per trajectory step (in MDP loop) |
| Training epochs | Hardcoded 1–2 rounds | Convergence-gated with patience=20 |
| Cross-validation | Single 90/10 static split | True 5-fold KFold |
| CR denominator (`basesim`) | Static global scalar file | Dynamically computed per fold |
| SSE denominator | `2 * 10 * cluster_size` | `2 * cluster_size` |
| Model selection | `modelnames[0]` (alphabetical) | Best validation CR checkpoint |
| `point.py` equality | Returns `None` on inequality | Returns `False` explicitly |
| Spatial normalization | `normloctrajs()` never called | Called during preprocessing |

---

## 14. Research Dynamics — Q-RLSTC Thesis Context

### 14.1 Training Dynamics Observed

| Observation | Classical RLSTC (original reward) | Q-RLSTC (modified reward) | Classical under Q-RLSTC reward |
|---|---|---|---|
| Policy collapse (never cut) | Yes (under original broken reward) | Solved via reward tuning | **Yes — classical collapses** |
| Policy collapse (always cut) | Possible | Mitigated by over-segmentation penalty | Possible |
| Effective learning | Yes (with SGD + large dataset) | Yes (with SPSA + 30 trajectories) | No |

**Root cause of asymmetry:** The reward function was modified extensively to make the quantum model learn. When that same modified reward is applied to the classical MLP (which wasn't designed for it), the classical model fails. This means:
- Direct reward-function comparisons between architectures require a shared, unmodified reward (original OD, Discrete Fréchet, or DTW)
- Any "quantum beats classical" claim must use the same reward function evaluated on both architectures

### 14.2 The "Competitive Ratio" Metric Risk

**Construction:** `CR = model_OD / baseline_OD` where baseline = "never cut" (entire trajectory as one segment)

**Fragmentation attractor (advisor critique):** If the denominator is the "never cut" scenario, the agent can drive CR below 1.0 by cutting every single point (reducing trajectories to tiny scattered dots), because each micro-segment has minimal internal IED. The metric improves regardless of whether cuts are semantically meaningful.

**Fix applied:** Budget-constrained CR evaluation (see Section 5 — Fragmentation Attractor) with `wValCR` length-weighting.

### 14.3 Figure 16 Reconstruction Failure — Root Cause Chain

```
Attempt to recreate Geolife cluster visualization (Figure 16)
  ├── Problem 1: Plotting initial k-means++ centroids instead of post-segmentation clusters
  │     └── Fix: Load from pickled outputs (geo-subtrajs, geolife_rlstc_clusters)
  ├── Problem 2: Raw unfiltered data (no 500-point cap, no 10-point floor)
  │     └── Fix: Apply preprocessing filters matching original paper
  ├── Problem 3: No street map underlay
  │     └── Fix: Plot over semi-transparent tile map to verify street alignment
  └── Problem 4 (root cause): Original "representative trajectories" were never computed centers
        └── Reality: max_span medoid cherry-pick (ERROR-01) — impossible to reproduce
            mathematically because the "center" was always an existing raw GPS track
```

### 14.4 Thesis Architecture Implications

**Proposed chapter framing options for forensic audit:**

| Option | Structure | Trade-off |
|---|---|---|
| **Option A** | Dedicated early "Baseline Pathology" chapter | Establishes broken state-of-the-art before quantum contribution; stronger narrative setup |
| **Option B** | Integrate as "Chapter 4 — Pathology #3" alongside fragmentation attractor and basesim bug | Keeps audit embedded in technical analysis section |
| **Option C** | Full 16-point audit in Appendix; Chapter 4 highlights Critical/High errors only | Maintains thesis focus on Q-RLSTC; audit available for scrutiny without dominating narrative |

**Background section requirements** (quantum-specialized committee): Quantum gates, variational circuits, VQC expressibility, QML basics, and how Hilbert space representation connects to the trajectory decision problem — necessary for Dr. Peterson/Sutton and external member (Imran Hayee) to evaluate quantum contribution claims.

### 14.5 Current State Summary

```
Original RLSTCcode-main
  └── Status: Empirically defective — 16 identified pathologies
      └── REAL_RLSTC_FIXED (Phase 1 + Phase 2 Complete)
            └── Status: Mathematically sound classical baseline
                ├── True geometric centroids
                ├── Authentic TRACLUS state features
                ├── Dense reward signal
                ├── Convergence-gated 5-fold CV
                ├── Correct SSE and CR metrics
                └── Ready for head-to-head Q-RLSTC benchmarks
```

---

*Sources:*

*— Liang, A., Yao, B., Wang, B., Liu, Y., Chen, Z., Xie, J., Li, F. "Sub-trajectory clustering with deep reinforcement learning." The VLDB Journal 33, 685–702 (2024). https://doi.org/10.1007/s00778-023-00833-w. Affiliations: Shanghai Jiao Tong University; Alibaba Group; Hangzhou Institute of Advanced Technology. Supported by NSFC (61832017), Alibaba AIR Program, National Key R&D Program of China (2020YFB1710200), Hangzhou Qianjiang Distinguished Expert Program.*

*— SMART DShot: Secure Machine-Learning-Based Adaptive Real-Time Timing Correction. Applied Sciences, MDPI 2025. https://www.mdpi.com/2076-3417/15/15/8619*

*— Comparison Report (internal Q-RLSTC research); master_results.csv (Geolife benchmark evaluation data)*

*— RLSTCcode-main forensic audit; REAL_RLSTC_FIXED remediation; Q-RLSTC research meeting transcripts and advisor correspondence (thesis research, 2025–2026)*