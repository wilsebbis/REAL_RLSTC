The VLDB Journal (2024) 33:685–702
https://doi.org/10.1007/s00778-023-00833-w
R E G U L A R P A P E R
Sub-trajectory clustering with deep reinforcement learning
Anqi Liang1
· Bin Yao1,3 · Bo Wang1
· Yinpei Liu2
· Zhida Chen2
· Jiong Xie2
· Feifei Li2
Received: 8 May 2023 / Revised: 4 December 2023 / Accepted: 10 December 2023 / Published online: 25 January 2024
© The Author(s), under exclusive licence to Springer-Verlag GmbH Germany, part of Springer Nature 2024
Abstract
Sub-trajectory clustering is a fundamental problem in many trajectory applications. Existing approaches usually divide the
clustering procedure into two phases: segmenting trajectories into sub-trajectories and then clustering these sub-trajectories.
However, researchers need to develop complex human-crafted segmentation rules for specific applications, making the clus-
tering results sensitive to the segmentation rules and lacking in generality. To solve this problem, we propose a novel algorithm
using the clustering results to guide the segmentation, which is based on reinforcement learning (RL). The novelty is that the
segmentation and clustering components cooperate closely and improve each other continuously to yield better clustering
results. To devise our RL-based algorithm, we model the procedure of trajectory segmentation as a Markov decision process
(MDP). We apply Deep-Q-Network (DQN) learning to train an RL model for the segmentation and achieve excellent cluster-
ing results. Experimental results on real datasets demonstrate the superior performance of the proposed RL-based approach
over state-of-the-art methods.
Keywords Sub-trajectory clustering· Spatio-temporal similarity· Reinforcement learning· Deep learning
1 Introduction
With the proliferation of GPS-equipped devices and location-
based services, a vast amount of trajectory data is generated
at an unprecedented rate. Trajectory data capture the move-
ments of moving objects and are highly valuable for various
B Bin Yao
yaobin@cs.sjtu.edu.cn
Anqi Liang
lianganqi@sjtu.edu.cn
Bo Wang
Wangbo324@sjtu.edu.cn
Yinpei Liu
yinpei.lyp@alibaba-inc.com
Zhida Chen
zhida.chen@alibaba-inc.com
Jiong Xie
xiejiong.xj@alibaba-inc.com
Feifei Li
lifeifei@alibaba-inc.com
1 Shanghai Jiao Tong University, Shanghai, China
2 Alibaba Group, Hangzhou, China
3 Hangzhou Institute of Advanced Technology, Hangzhou,
China
applications such as site selection, green transport, tourism
planning, and traffic monitoring [19, 29, 35, 36]. In recent
years, sub-trajectory clustering has attracted much atten-
tion. The sub-trajectory clustering task is to segment the
trajectories into sub-trajectories based on certain principles
and then cluster them w.r.t a given distance metric. Cluster-
ing the whole trajectories poses a challenge in discovering
local similarities among trajectories since they vary in length
and time range, leading to the loss of valuable information.
As illustrated in Fig. 1, the four trajectories share similar
sub-trajectories in the box, indicating the existence of a
common pattern among them. However, clustering these
trajectories as a whole will make the pattern unobservable
because they are dissimilar and will be put into different clus-
ters. Sub-trajectory clustering is helpful in many real-world
applications, particularly for analyzing the correlations or
common patterns among portions of different trajectories in
regions of interest. For example, in hurricane forecasting,
given the historical hurricane trajectories, researchers focus
on the common patterns of sub-trajectories near the coastline
because they can predict the locations of hurricane landfall
by analyzing the patterns. Therefore, discovering the com-
mon patterns of sub-trajectories can improve the accuracy
of hurricane forecasting and reduce damages caused by the
hurricane.
123
686 A. Liang et al.
Fig. 1 black box)
Four dissimilar trajectories with a similar portion (the part in the
Existing solutions to the sub-trajectory clustering problem
follow the first-segment-then-cluster paradigm. The effec-
tiveness of the clustering result heavily depends on the
segmentation step, which is a crucial difference from the
trajectory clustering problem. Existing solutions can be clas-
sified into two categories. The first one regards segmentation
and clustering as two independent procedures. It segments
the trajectories into sub-trajectories according to spatial fea-
tures and then clusters these sub-trajectories. For instance,
Lee et al. [18] proposed TRACLUS, which segments trajec-
tories into line segments based on the minimum description
length principle and then clusters line segments by applying
DBSCAN. The second one segments trajectories by taking
the clustering quality into account. For example, Pelekis et
al. [25] segment trajectories based on the local density cri-
terion and then conduct clustering according to the density.
These methods usually rely on complicated hand-crafted seg-
mentation rules tailored to specific applications, making the
clustering result sensitive to these rules and limiting their
generality.
To address these issues, we consider utilizing the clus-
tering quality to guide the segmentation in a data-driven
manner instead of using hand-crafted segmentation rules.
A straightforward approach is enumerating all segmentation
schemes and choosing the one with the best clustering qual-
ity. However, this method is computationally expensive and
is thus impractical for real-world applications. Therefore,
we propose an efficient and effective reinforcement learn-
ing (RL)-based framework that leverages clustering quality
to guide segmentation. It allows segmentation and clustering
components to cooperate closely, mutually enhancing each
other and achieving superior clustering results.
Specifically, we treat the segmentation procedure as a
sequential decision process, i.e., it sequentially scans the
trajectory points and decides whether to segment at that
point, which can be modeled as a Markov decision process
(MDP) [26]. The crucial challenge of the MDP formulation
is to define the state and reward, which significantly affect
the performance of the learned model, i.e., the RL agent.
Regarding the design of the state, the intuitive idea is to
include more features to reflect the environment. However,
experimental results indicate that such an approach may lead
to much longer training time without noticeable improve-
ments in the model’s performance. To solve this problem, we
propose a representative set of features that can effectively
summarize the sub-trajectory clustering characteristics while
maintaining acceptable training time. Designing an appropri-
ate reward is vital to the training process as it encourages the
RL agent to choose good actions to boost performance. We
define the reward based on the evaluation metrics of clus-
tering quality, which allows us to leverage the clustering
quality to guide the trajectory segmentation, thus achiev-
ing the data-driven trajectory segmentation. We employ the
Deep-Q-Network (DQN) method [24] to learn the policy for
the MDP in our trajectory segmentation problem. Based on
the learned policy, we propose a RL-based Sub-Trajectory
Clustering algorithm called RLSTC. Our algorithm is data-
driven and can adapt to different dynamics of the underlying
trajectories. Moreover, it is a general solution that can be
applied to various trajectory similarity measurements and
application scenarios.
To further improve efficiency, we propose several opti-
mizations to decrease the cost of trajectory distance computa-
tion. First, we implement a preprocessing step that simplifies
trajectories by reserving only the critical points. Second, we
apply an approximate method to compute trajectory similar-
ity in linear time. Finally, we utilize an incremental similarity
computation strategy to avoid recomputing the distance from
scratch.
In summary, our main contributions are as follows:
• We propose a novel RL-based framework for the sub-
trajectory clustering problem, which is the first RL-based
solution to the best of our knowledge.
• We creatively model the procedure of trajectory seg-
mentation as an MDP, especially defining the state and
reward of MDP by considering the distinct features
of the sub-trajectory clustering. This method achieves
data-driven trajectory segmentation without relying on
complex hand-crafted rules.
• We have developed several optimizations to enhance the
efficiency of our solution.
• We conduct extensive experiments on various real-world
trajectory datasets. The experimental results demonstrate
that our RL-based algorithm outperforms state-of-the-art
methods in terms of both effectiveness and efficiency.
The remainder of the paper is as follows: In Sect. 2, we
present preliminaries and our problem statement. Section 3
discusses the preprocessing algorithm. In Sect. 4, we detail
123
Sub-trajectory clustering with deep… 687
Table 1 Symbols and description
Notation Description
D A dataset of N trajectories, i.e.,
D = {T1, T2,
· · ·
, TN }
N The number of trajectories in dataset D
T A trajectory
pi p(x) p(y) p(t) |T | The i t h point of T
The longitude of point p
The latitude of point p
The timestamp of point p
The length of T , i.e., the number of points
in T
T (i , j ) The sub-trajectory of T that starts and
ends at the i t h and j t h point of T ,
respectively
T (ts ) Starting time of T
T (te ) Ending time of T
st State at time step t
at Action taken at time step t
rt Reward obtained at time step t
θ Parameters of the main neural network
θ′ Parameters of the target neural network
{C j }k
j=1 Learned k clusters
ci The center of cluster Ci
OD Metric for evaluating the clustering
quality
Fig. 2 An illustration of the synchronous point
our RL-based algorithm, including modeling trajectory seg-
mentation as an MDP and the algorithm itself. Experimental
results are provided in Sect. 5. We give a brief review of
related work in Sect. 6. Finally, Sect. 7 concludes the paper.
Definition 3 Sub-trajectory Clustering A sub-trajectory clus-
ter consists of sub-trajectories that are similar w.r.t a certain
distance metric. The clustering procedure is to partition
the sub-trajectories into k clusters, where k is a predefined
parameter. Each cluster Ci has a center ci . Note that ci is not
necessarily a sub-trajectory in Ci.
Definition 4 Clustering Quality Our objective is to cluster the
sub-trajectories such that sub-trajectories in the same cluster
are similar to each other. To measure the clustering quality,
we adopt the popular agglomeration degree [27] and compute
the overall distance of clusters as follows:
OD=
k
i=1
mi
m
ODi , (1)
where ODi is the average distance between the center and
the sub-trajectories in the cluster Ci , mi is the number of
sub-trajectories in Ci , and m is the total number of sub-
trajectories. The computation of ODi is given by Eq. (2):
ODi =
x∈Ci
d(x,ci )
. (2)
2 Preliminaries
In this section, we explain the preliminaries and present the
definition of the sub-trajectory clustering problem. Com-
monly used notations are described in Table 1.
Definition 1 Trajectory A trajectory T is a sequence of
time-ordered spatial points (points, for short), i.e., T=
⟨p1, p2,
· ··
, pn ⟩, where pi is a triplet (xi , yi ,ti ). Here,
(xi , yi ) represents a location and ti the timestamp. The vari-
able n denotes the number of points in T . The length of T
is defined by the number of points, i.e., |T | = n. The time
interval of T is denoted by [ts ,te], where ts and te are the
earliest and latest timestamp in T , respectively.
Definition 2 Sub-trajectory Given a trajectory T=
⟨p1, p2,
· ··
, pn ⟩, a sub-trajectory T (i , j )(1 ≤ i ≤ j ≤ n)
of T is defined as pi ,
·· ·
, p j , representing that T (i , j )
starts at the i th point and ends at the j th point of T .
mi
Here, x is a sub-trajectory in cluster Ci , and d(·
,
·) is a
distance function that measures the distance between (sub-
)trajectories.
By substituting ODi in Eq. (1) with Eq. (2), OD becomes
the average distance between each sub-trajectory and the
cluster to which it belongs, thus allowing us to measure the
effectiveness of the clustering. Our goal is to achieve clus-
tering with a minimal OD. Note that any trajectory distance
metric can be utilized in Eq. (2).
Definition 5 Synchronous point Consider any two trajec-
tories T and Q, with pi representing a point in T and p j
representing a point in Q. We define p j as the synchronous
point of pi , and vice versa, if they share the same timestamp.
As shown in Fig. 2, pi and p j form a pair of synchronous
points.
For points on trajectory Q that lack a corresponding syn-
chronous point on trajectory T , we establish synchronous
123
688 A. Liang et al.
points using linear interpolation. This process is detailed in
Eqs. (3), (4), and (5). The dotted-line points illustrated in
Fig. 2 represent the synchronous points obtained through this
interpolation.
p
′(x)= ps (x) + p(t)− ps (t)
· (pe(x)− ps (x)), (3)
pe(t)− ps (t)
p
′(y)= ps (y) + p(t)− ps (t)
· (pe(y)− ps (y)), (4)
pe(t)− ps (t)
p
′(t)= p(t). (5)
Fig. 3 Example of the L(H ) and L(D|H ) computation
We assume that any object corresponding to trajectory T
does not move before T (ts ) and after T (es ). For a point p in
trajectory Q with a timestamp before T (ts ), the location of
p’s corresponding synchronous point in trajectory T is set
to the same as T ’s first point, and the timestamp of the syn-
chronous point matches that of p. Similarly, if the timestamp
of p is after T (es ), then the location of the synchronous point
aligns with the last point in T , and its timestamp is set to be
the same as that of p.
Definition 6 Trajectory Spatio-temporal Distance Spatial
similarity is insufficient to evaluate the relationship between
trajectories in real-world applications. Thus, we also consider
temporal similarity. We define the spatio-temporal distance
between two trajectories T and Q over a specific time inter-
val [tmin,tmax] by integrating their Euclidean distance over
time:
dIED(Q, T )=
tmax
tmin
dQ,T (t)dt. (6)
Here, tmin = min(Q(ts ), T (ts )), tmax = max(Q(te), T (te)),
with dQ,T (t) representing the Euclidean distance between
synchronous points of the trajectories at time t, known as the
synchronous Euclidean distance (SED) [23]. We perform lin-
ear interpolation on each point in T to find its synchronous
point in Q and vice versa. Figure 2 illustrates this interpo-
lation process, where solid circles represent original points
and hollow ones are interpolated points. Therefore, Eq. (6)
can be transformed into:
cost to O(n + m). The equation is as follows:
1
dIED(Q, T )=
k=n+m−1
k=1
((dQ,T (tk ) + dQ,T (tk+1))
2
(8)
· (tk+1− tk )).
Furthermore, we employ an incremental calculation
approach to reduce computational costs further. The distance
between sub-trajectory T (i , j ) and trajectory Q can be com-
puted as follows:
dIED(Q, T (i , j ))
=
1
2
k=j−2
k=1
((dQ,T (tk ) + dQ,T (tk+1))· (tk+1− tk ))
(9)
1
+
2 (dQ,T (t j−1) + dQ,T (t j ))· (t j− t j−1).
Here, the first term corresponds to the distance between
T (i , j− 1) and Q, a value that has already been com-
puted. Thus, we only need to compute the second term, which
has a time complexity of O(1). Consequently, this approach
reduces the time complexity from O(m +n) to O(1) through
incremental calculation.
Problem 1 Given a trajectory dataset D, the task of sub-
trajectory clustering is to conduct segmentation on the
trajectories in D and to form an optimal clustering Copt =
{C j }k
j=1 such that Eq. (1) is minimized.
dIED(Q, T )=
k=n+m−1
k=1
tk+1
tk
dQ,T (t)dt, (7)
where n and m are the number of points in Q and T , respec-
tively, and tk is the timestamp of an (interpolated) point. If
the time intervals of T and Q do not intersect, dIED(Q, T ) is
considered infinite.
We adopt the Trapezoid approximation method [13] to cal-
culate the integral of dQ,T (t), which reduces the computation
3 Preprocessing
A preprocessing step is employed to simplify the trajecto-
ries, which is necessary for two main reasons. First, not
all points in a trajectory are equally important. For exam-
ple, when an object moves at a constant speed along a
straight line, the points between the first and last ones do not
carry significant information. Second, the cost of computing
123
Sub-trajectory clustering with deep… 689
spatio-temporal similarity between trajectories is propor-
tional to the number of points. Therefore, we retain only
the significant points, referred to as reserved points, to sim-
plify trajectories. The goal is to minimize the difference
between the simplified and original trajectories while min-
imizing the number of points in the simplified trajectory.
However, these two objectives are contradictory and require
a trade-off. Inspired by [15, 18], we employ the minimum
description length (MDL) principle to get an optimal sim-
plification scheme. MDL consists of two components: L(H )
and L(D|H ). Here, L(H ) is the sum of the spatio-temporal
lengths of all line segments in the simplified trajectory, and
L(D|H )represents the difference between the simplified and
original trajectory. L(H ) denotes the degree of conciseness,
and L(D|H ) the degree of preciseness. Suppose the original
trajectory is T= ⟨p1, p2,
·· ·
, pn ⟩ and its simplified version
is Tsimp= pr1 , pr2 ,
·· ·
, prm , where pri denotes a reserved
point in T . The MDL of Tsimp and T is given by:
MDLTsimp = L(H ) + L(D|H )
= log2
m−1
dst (pri , pri +1 )
i=1
+ log2
m−1
i=1
(10)
where dst =
1
2 dED(pri , pri +1 ) + 1
2 |pri (t)− pri +1 (t)|, and
dED(·
,
·) is the Euclidean distance between two points. Ti =
pi ,
· ··
, pri +1 , and Tisimp = pri , pri +1.
n−1
i=1
dst (pi , pi +1), (11)
Algorithm 1: Trajectory preprocessing
Input : A trajectory T= ⟨p1, p2,
· · ·
, pn ⟩
Output: A simplified trajectory Tsimp
1 Initialize the set of reserved points as a empty list R P ← [ ];
2 i ← 0, step ← 1;
3 Insert p1 into R P;
4 while i + step < n do
5 j ← i + step;
6 if MDLTsimp (pi ,p j ) > MDLT (pi ,p j ) then
7 Insert p j into R P;
8 i ← j ;
9 step ← 1;
10 else
11 step ← step + 1;
12 Insert pn into R P;
13 Convert R P into trajectory Tsimp;
14 return Tsimp;
dIED(Ti , Tisimp ),
MDLT = L(H ) + L(D|H )= log2
in a trajectory. To address this, we propose an approximate
greedy solution that regards the set of local optima as the
global optimum. Assume the original trajectory T is rep-
resented as pi ,
···
, p j . A local optimum is a simplified
trajectory containing the minimum points while satisfying
MDLTsimp(pi ,pk ) ≤ MDLT (pi ,pk ) for every k ( i < k ≤
j ). Here, T (pi , pk )= ⟨pi ,
···
, pk ⟩, and Tsimp(pi , pk )=
⟨pi , pk ⟩.
The preprocessing procedure is outlined in Algorithm 1.
If MDLTsimp(pi ,p j ) > MDLT(pi,p j ), then p j is chosen as a
reserved point (Lines 6–9). Otherwise, p j is omitted, and the
next point is checked (Line 11). The time complexity of this
algorithm is O(n), where n is the number of points in T .
This is because the number of MDL computations equals the
number of points in T , and the time complexity of each MDL
computation is O(1) based on the incremental computation
of dIED.
where the L(D|H ) of MDLT is 0.
Example 1 As shown in Fig. 3, the original trajectory T is
denoted by a solid line (T= ⟨p1, p2, p3, p4⟩), and the
simplified trajectory Tsimp is represented by a dashed line
(Tsimp= pr1 , pr2 ), where pr1 and pr2 correspond to p1 and
p4 in the original trajectory T , respectively. The MDL of
Tsimp and T is as follows:
MDLTsimp = log2 dst (pr1 , pr2 )
+ log2 dIED(T , Tsimp), (12)
MDLT = log2(dst (p1, p2) + dst (p2, p3)
+ dst (p3, p4)). (13)
Since each simplification scheme has its own MDL, we
aim to determine the optimal scheme that yields the minimum
MDL. However, obtaining an exact solution is impractical as
it requires enumerating all combinations of reserved points
4 Methodology
In this section, we first provide an overview of our RL-based
solution in Sect. 4.1. We then detail how to model trajectory
segmentation as an MDP in Sect. 4.2 and describe the learn-
ing procedure in Sect. 4.3. The computation of cluster centers
is presented in Sect. 4.4. Finally, in Sect. 4.5, we introduce
the RL-based Sub-Trajectory Clustering algorithm, termed
as RLSTC.
4.1 Framework overview
Figure 4 presents the overview of our RL-based sub-trajectory
clustering framework.
In our approach, we employ an offline training pro-
cess, modeling the segmentation procedure as an MDP and
123
690 A. Liang et al.
Fig. 5 Overview of the MDP formulation
Fig. 4 Overview of the RL-based Solution
applying the DQN method to learn an optimal segmenta-
tion policy. Subsequently, we develop the RLSTC algorithm,
which comprises segmentation and clustering components.
The segmentation component performs trajectory segmen-
tation based on the learned policy, and then, the clustering
component assigns each sub-trajectory to its nearest cluster
and updates the cluster centers. After that, the clustering com-
ponent forwards the updated cluster centers and clustering
quality to the segmentation component, which leverages this
information to generate improved segmentation. This itera-
tive process continues until a satisfactory clustering result is
achieved.
represent a state. This method involves calculating the OD
values by assigning a new sub-trajectory to each cluster cen-
ter and utilizing the resulting k OD values to represent the
state. However, this design is constrained by the fact that a
model trained for a specific k is not adaptable to different k
values, thereby limiting its overall flexibility. Furthermore,
our experiments reveal that this approach to defining the state
leads to poor performance.
Therefore, we establish a comprehensive representation
of the state by identifying a set of features that consider both
the trajectories and the quality of clustering. In most cluster-
ing situations, the proximity of objects to the center of their
respective clusters is a crucial indicator of effective cluster-
ing. Thus, newly generated sub-trajectories are assigned to
the nearest cluster to ensure optimal clustering results. In this
case, the OD at state st can be computed as:
4.2 Modeling trajectory segmentation as MDP
Trajectory segmentation is a sequential decision-making task
that involves scanning trajectory points to make segmenta-
tion decisions. Therefore, it can be effectively modeled as an
MDP, as depicted in Fig. 5. In this process, the RL agent inter-
acts with the environment by taking a sequence of actions.
Each action changes the state of the RL agent, which receives
a corresponding reward. We formulate the trajectory segmen-
tation problem as an MDP with the following components:
States A state reflects the environment where decision-
making is conducted. For instance, in a scenario with k
cluster centers and a trajectory T , we denote the state at
the tth point of T as st . The challenge is determining the
most appropriate features to represent this state accurately.
The intuition is to include comprehensive features related to
the trajectory clustering scenario, such as the cluster centers
and trajectories. However, our experimental results show that
this approach significantly slows down the training process
and fails to yield satisfactory clustering results. An alterna-
tive design approach is to leverage the clustering quality to
st (OD)=
x∈C1 d(x,c1) + ··· + x∈Ck d(x,ck )
numst
i=1
min
j=1,
···
numst
,k
d(xi ,c j )
(14)
=
,
numst
where numst is the number of sub-trajectories produced until
st . We define two variables, ODs and ODn , to represent the
OD produced by segmenting or not segmenting at the point
currently being scanned.
Additionally, we employ an incremental approach that uti-
lizes previously obtained OD values to enhance the efficiency
of the computation process. The ODs value of st can be com-
puted as:
st−1(OD)· numst−1
st (ODs )=
numst−1 + 1
min
j=1,
···
,k
d(x,c j )
(15)
+
numst−1 + 1,
123
Sub-trajectory clustering with deep… 691
where x is the sub-trajectory produced by segmenting at pt.
The ODn of st can be computed as:
st (ODn )= st−1(OD). (16)
However, relying solely on this information might make
the RL agent shortsighted. In other words, the agent might
choose to segment at point pt simply because st (ODs ) <
st (ODn ), potentially overlooking better results that could be
achieved by segmenting at subsequent points after pt . To
address this, we introduce two variables, Lb and L f , denot-
ing the length of the generated sub-trajectory and the length of
the remaining sub-trajectory, respectively. They are defined
as:
st (Lb)=
m1
|T |, st (L f )=
m2
|T |, (17)
where m1 is the number of points in the generated sub-
trajectory by segmenting pt , and m2 is the number of points
in the sub-trajectory that starts at pt and ends at the last point
of T .
The convergence of the RL model can be expedited by
integrating the expert knowledge, represented as ODb gen-
erated by TRACLUS. The ODb value can be calculated as:
num′
i=1
min
j=1,
···
ODb =
,k
d(x′
i ,c j )
num′ , (18)
where num′ denotes the total number of sub-trajectories pro-
duced by segmenting trajectories in dataset D, and x′
i is the
i th sub-trajectory. Finally, the state st is defined as:
st = (st (ODs ),st (ODn ),ODb,st (Lb),st (L f )). (19)
Actions An action represents a decision made by the RL
agent. Let pt denote the current point to be considered. The
action at ∈ {0,1} indicates whether to perform a segmen-
tation at pt . If at = 1, the agent executes a segmentation
operation at pt , thereby creating a new sub-trajectory and
increasing the total number of sub-trajectories by one. In
addition, the value of OD is updated accordingly. Conversely,
if at = 0, the agent does not perform segmentation at pt and
instead proceeds to consider the next point pt+1.
Transitions A transition refers to a change in the state
resulting from an action taken by the agent. In trajectory
segmentation, when the agent takes an action at in the cur-
rent state st , it leads to a new state st+1. Since the absence
of a known probability for the transition from st to st+1, we
employ a model-free method to solve the MDP, which is
particularly effective in scenarios where the transition prob-
abilities are unknown.
Rewards The reward function assigns a value to a transition,
reflecting the quality of the action that caused the transi-
tion. In order to guide the trajectory segmentation process
using clustering quality, we utilize the OD value to define
our reward function. Assuming a state transition from st to
st+1 after taking an action at , we define an immediate reward
as the difference in OD values between successive states, i.e.,
rt = st (OD)−st+1(OD). For each trajectory, when process-
ing a sequence of states s1,s2,
···
,s|T |, the accumulative
reward is defined as:
R=
|T |−1
rt
t=1
=
|T |−1
(st (OD)− st+1(OD))
t=1
= s1(OD)− s|T |(OD).
(20)
where s1(OD)= 0, and s|T |(OD) corresponds to the OD of
the terminated state. The objective of the MDP solver is to
maximize the accumulative reward, which is consistent with
minimizing OD.
4.3 Learning the optimal policy
RL aims to determine an optimal policy that specifies the
action a to be taken in each state s to maximize the accu-
mulative reward. The Q-value [40] is utilized to represent
the total expected reward. The optimal Q-value, denoted
as Q∗(s,a), signifies the maximum accumulative reward
achieved by following the optimal policy. According to the
Bellman Expectation Equation [31], we have:
Q∗(s,a)= E rt + γ max
a′
Q∗(st+1,a
′)|st ,at , (21)
which means that the maximum expected accumulative
reward is the sum of the agent’s immediate and the maximum
future rewards. Here, γ is a discount factor that balances the
importance of immediate and future rewards. We adopt the
DQN method to approximate the optimal Q∗(s,a) using a
deep neural network Q∗(s,a;θ), where θ represents the set
of parameters for the neural network.
Figure 6 shows the procedure of our DQN algorithm, and
Algorithm 2 presents the pseudocode. The algorithm begins
by initializing the main network Q(s,a;θ), the target net-
workˆ
Q(s,a;θ′), a fixed-size replay memory pool M, and
k cluster centers derived by the k-means++ method (Lines
1–4). The main network estimates the Q-value, and the tar-
get network computes the loss function for training the main
network. Replay memory M stores recent transitions, which
are uniformly sampled during training to reduce the correla-
tion between consecutive transitions. The subsequent steps
123
692 A. Liang et al.
Algorithm 2: Deep-Q-Network (DQN) learning
Fig. 6 Illustration of DQN learning
involve a series of episodes, each representing the segmen-
tation process of a trajectory. Specifically, the algorithm
shuffles trajectories at the start of the training epoch, pro-
cesses each trajectory sequentially from the training dataset,
and sets the initial state s1 of the first trajectory point (Lines
6–15). The algorithm then proceeds with |Ti | time steps. At
each tth time step, it checks point pt and decides on an action
using the ϵ-greedy strategy based on the main network (Line
17). If at = 1, segmentation is performed at pt , and vari-
ables sp , num, and st (OD) are updated accordingly (Lines
22–25). If at = 0, the algorithm proceeds to point pt+1.
The algorithm observes a new state st+1 and calculates the
immediate reward rt (Lines 26–28). Each experience tuple
(st ,at ,rt ,st+1) is then added to replay memory M (Line
29). Periodically, the algorithm samples a random minibatch
of experiences from M to update the main network using a
stochastic gradient descent algorithm, aiming to minimize
the mean squared error (MSE) (Lines 30–31), defined as:
MSE(θ)= (y− Q(st ,at ;θ))2
, (22)
Input : A trajectory dataset D
Output: Learned Q-value function Q(s,a; θ)
1 Initialize the main network Q(s,a; θ) with random parameters θ;
2 Initialize the target networkˆ
Q(s,a; θ′) with parameters θ′
= θ;
3 Initialize the replay memory as a empty list M ← [ ];
4 Initialize k cluster centers by k-means++ method;
5 for epoch= 1,2,
· · ·
,m do
6 Shuffle the trajectory dataset;
7 for i= 1,2,
· · ·
, N do
8 Sequentially access trajectory Ti ;
9 num ← 0;
10 /* sp is the latest segmentation point, and subtraj is the
latest sub-trajectory */
11 12 13 14 15 16 17 sp ← p1, subtraj ← ⟨ ⟩;
ODs ← 0, ODn ← 0;
Compute ODb by Eq. (18);
Lb ← 1
|Ti |, L f ← 1;
s1 ← (ODs ,ODn ,ODb , Lb , L f );
for t = 1,2,
· · ·
,|Ti | do
Choose an action by ε-greedy strategy based on the
main network;
18 20 if t = |Ti | then
19 subtraj ← sp ,
· · ·
, pt ;
break;
21 if at = 1 then
22 subtraj ← sp ,
· · ·
23 sp ← pt ;
24 num ← num + 1;
25 st (OD) ← st (ODs );
, pt ;
26 27 28 29 30 31 32 33 Update ODs , ODn , Lb , L f by Eq. (15)–Eq. (17);
Next state st+1 ← (ODs ,ODn ,ODb , Lb , L f );
rt ← st (OD)− st+1(OD);
M.append((st ,at ,rt ,st+1));
Uniformly sample a minibatch from M;
Perform a stochastic gradient descent algorithm on the
loss function;
Periodically update the target networkˆ
Q(s,a; θ′);
Validate the model on validation dataset; Update cluster
centers;
where Q(st ,at ;θ) is the Q-value predicted by the main net-
work, and y is the target Q-value, computed as:
y=
⎧ ⎨ ⎩
rt , if st+1 is the terminated state
ˆ
rt + γ max
Q(st+1,a
′;θ′), otherwise.
a′
(23)
Here,ˆ
Q(st+1,a′;θ′) represents the output of the target net-
work. The target networkˆ
Q(s,a;θ′) is periodically updated
using the soft update method (Line 32). The optimal policy is
learned by selecting the action a that maximizes Q(s,a;θ)
for a given state s.
Moreover, we employ several strategies to minimize the
impact of the traversal order of trajectories: (1) To mitigate
the impact of the data accessing order, we shuffle the trajec-
tories before each training epoch. (2) During training, we
introduce randomness through an exploration-exploitation
parameter, which allows the model to use its learned expe-
riences while still trying unexplored actions, thus avoiding
local optima. (3) We initialize the clustering centers using
the k-means++ method to reflect the data distribution com-
prehensively. Additionally, we update the clustering centers
after each training epoch to adapt to the evolving data.
4.4 Computation of cluster center
In this section, we propose a method for generating the
representative trajectory, i.e., the cluster center. The cluster
center captures the collective movement pattern of trajec-
tories within the cluster. To achieve this, we determine the
average coordinate at each timestamp to generate a represen-
tative trajectory. We scan the timestamps in chronological
order and record the number of trajectories containing each
timestamp. If the number of trajectories at a given timestamp
is no less than a predefined threshold MinNum, we compute
123
Sub-trajectory clustering with deep… 693
Fig. 7 Example of center computation
Algorithm 3: RLSTC algorithm
Input : A trajectory dataset D, a threshold τ, a learned model
Q(s,a; θ), k
Output: k clusters
1 Initialize k cluster centers;
2 while true do
3 for episode= 1,2,
· · ·
, N do
4 Randomly sample a trajectory T from D;
5 /∗ Segmentation component ∗/
6 for t = 1,2,· · · |T | do
7 at ← arg max
Q(st ,a; θ);
a
8 Execute Lines 18-27 of Algorithm 2;
9 /∗ Clustering component ∗/
10 11 12 Assign each sub-trajectory to its nearest cluster;
Update the cluster centers of the k clusters;
/∗ the new cluster centers are represented by ci , the previous
cluster centers in the last iteration are c′
i ∗/
13 14 15 16 17 18 centerdists ← [ ];
for i= 1,2,
· · ·
,k do
centerdists.append(d(ci ,c′
i ));
maxdist ← max(centerdists);
if maxdist ≥ τ then
Forward the updated cluster centers and the current OD to
the segmentation component;
19 20 if maxdist < τ then
return k clusters;
the average coordinate for that timestamp. We derive syn-
chronous points through linear interpolation for trajectories
that lack a sampled point at that timestamp. Figure 7 pro-
vides a simple example of this approach, where the solid
circles indicate the original sampled points, and the hollow
triangles represent the synchronous points added through
interpolation. The average coordinates at timestamps 4, 5,
and 6 are computed because the number of trajectories with
these timestamps meets or exceeds the threshold MinNum.
This process results in the generation of the representative
trajectory.
4.5 The RLSTC algorithm
In this section, we propose the RLSTC algorithm, outlined in
Algorithm 3. Initially, we utilize the k-means++ method to
initialize k cluster centers (Line 1). Subsequently, the learned
policy is applied to segment each trajectory (Lines 6–8). The
sub-trajectories are then assigned to their nearest cluster, and
the cluster centers are updated accordingly (Lines 10–11).
Following this, we calculate the maximum distance between
the updated and previous cluster centers (Lines 14–16). If
this maximum distance exceeds a predefined threshold, we
forward the updated cluster centers and the clustering quality
to the segmentation component (Lines 17–18). The algo-
rithm terminates if the maximum distance does not exceed
the threshold (Lines 19–20).
In Algorithm 3, the cluster centers will gradually converge
after multiple iterations, regardless of the traversal order,
thereby bringing a stable clustering result eventually. There-
fore, the traversal order of trajectories has a limited impact
on the clustering result.
Time Complexity The time complexity of the RLSTC algo-
rithm is O(n N ), where n is the average length of trajectories,
and N is the total number of trajectories in the dataset. Seg-
menting trajectories and assigning each sub-trajectory to its
nearest cluster are the most time-consuming parts (Lines 3–
10 in Algorithm 3). Assuming that the number of clusters,
k, is a constant, the time cost for processing each trajectory
point involves three components. First, computing the state
requires O(1) time, as the distance between a sub-trajectory
and a cluster center is incrementally computed. Second,
choosing an action, segmenting at a point, and assigning a
sub-trajectory to its nearest cluster also take O(1) time, as
the minimum distance between the sub-trajectories and clus-
ter centers is recorded while constructing the state. Finally,
updating the state requires O(1) time. Thus, the total time
cost for processing all trajectories in the dataset is O(n N ).
The time cost for updating the cluster centers (Line 11 in
Algorithm 3), as described in Sect. 4.4, is O(m M), where m
is the average length of sub-trajectories, and M is the total
number of sub-trajectories. Additionally, checking the dif-
ference between cluster centers of consecutive iterations has
a time complexity of O(m′) (Line 16 in Algorithm 3), where
m′ is the average length of cluster centers. Therefore, the time
complexity of one iteration is O(n N + m M + m′), and the
total time complexity of RLSTC is O(c(n N + m M + m′)),
with c representing the number of iterations. Since O(n N )
and O(m M) are of the same order of magnitude, and O(m′)
is less than O(n N ), and considering that RLSTC converges
within a few iterations, the overall time complexity can be
approximated as O(n N ).
123
694 A. Liang et al.
5 Experiments
5.1 Experimental setup
Dataset We evaluate the performance of our algorithm on
two widely used real-world trajectory datasets, Geolife and
T-Drive.
1. Geolife [51] records the outdoor GPS trajectories of 182
users over three years. It contains 17,364 trajectories, cov-
ering a total distance of about 1.2 million kilometers and a
duration of over 48,000 h. The GPS records were collected
at a sampling rate of 1–5 s. To enhance computational
efficiency, we constrain the number of points within each
trajectory to between 10 and 500. Trajectories exceeding
500 points are randomly reduced to 500 points, and those
with fewer than 10 points are discarded, resulting in a total
of 17,070 trajectories.
2. T-Drive [47] contains the trajectories of 10,289 taxis in
Beijing over a week. It comprises about 17 million points,
with the total trajectory distance exceeding 9 million
kilometers. The trajectory points were collected every
3–5 min. For trajectories with more than 500 points, we
randomly select 500 points from them. Trajectories with
fewer than 10 points are discarded. This process yields a
total of 9,937 trajectories.
We divide the dataset into n parts, one for testing and the
remaining n− 1 parts for training. 10% of the training set is
allocated as the validation set, while the remaining data are
used for training. In our experiments, we set n to 5.
Evaluation platform We conduct all experiments on a Dell
server with an NVIDIA P100 GPU, a 48-core Intel(R)
Xeon(R) CPU E5-2678 v3 @ 2.50GHz, and 128GB mem-
ory. All methods are implemented using Python 3.6, and the
implementation of our method is based on TensorFlow 2.2.0.
5.2 Baselines
We compare RLSTC with eight competitor methods in terms
of clustering quality and efficiency.
– TRACLUS [18] is a widely adopted classical framework
for trajectory segmenting and clustering, which utilizes
artificial partitioning rules alongside density-based clus-
tering to achieve effective results.
– S2T-Clustering [25] employs a global voting algorithm
that effectively adopts local density to segment and clus-
ter trajectories.
– Sub-trajectory Clustering [1] (For convenience, we will
refer to it as SubCLUS below.) SubCLUS is the state-of-
the-art method that models the sub-trajectory clustering
problem as a set-cover problem and proposes an approx-
imation solution to achieve the optimization objective.
– Greedy Method is a degraded version of RLSTC that
follows an optimal local strategy, segmenting at a point
only if its ODs is smaller than its ODn.
– CLUSTMOSA [8] uses three clustering objective criteria
and applies archived multi-objective simulated annealing
for optimal clustering.
– HyperCLUS [42] segments each trajectory into a series
of hypercubes and identifies the common sub-trajectories
(clusters) by evaluating intersections among hypercubes.
– RTCLUS [28] introduces a novel distance metric to
improve segmentation and clustering accuracy, using an
R-tree to enhance efficiency.
– RLSTC w/o simplification employs the same segmen-
tation and clustering method as RLSTC but omits the
trajectory preprocessing procedure.
5.3 Hyperparameter setting
The main network in DQN consists of a two-layer feedfor-
ward neural network. The first layer involves 64 neurons and
uses the ReLU activation function. The second layer involves
two neurons as the output, corresponding to the capacity of
the action space. The target network has the same structure
as the main network.
We utilize stochastic gradient descent (SGD) with a
learning rate of 0.001 based on empirical findings. Hyper-
parameters were set following guidelines from RL-based
trajectory processing literature [38, 39] and our prelimi-
nary experiments. The exploration-exploitation balance is
controlled by the parameter ε, initially set to 1.0 and then
reduced to 0.99ε after each step, with a lower bound of 0.1
to maintain a certain degree of exploration in the training.
The reward discount factor γ is set as 0.99. The size of the
replay memory and the sample size at each iteration are set as
5,000 and 32, respectively. The target network updates follow
θ′
= ωθ + (1− ω)θ′ at the end of each episode, with ω set
to 0.001. According to the preliminary experimental results,
we set the cluster number k to 10 and the maximum distance
threshold τ between cluster centers in successive iterations
to 0.1.
To set the parameters of the baseline method TRACLUS,
we use the heuristic method in [18]. If it fails to estimate
the values of the parameters, we determine the parameters
by manually searching over a range of values. The parame-
ter settings for S2T-Clustering, SubCLUS, CLUSTMOSA,
HyperCLUS, and RTCLUS follow the strategies outlined in
their respective studies [1, 8, 25, 28, 42].
123
Sub-trajectory clustering with deep… 695
5.4 Distance measurements
In addition to using dIED, the default distance measurement
presented in Definition 6, we evaluate RLSTC with three
commonly used distance measurements. These are the dis-
crete Fréchet distance dDF [2], DTW distance dDTW [45],
and Weighted distance dWD [18]. The equations for these
distance measurements are as follows:
dDF(Q, T )=
⎧ ⎪ ⎪ ⎪ ⎪ ⎪ ⎪ ⎨ ⎪ ⎪ ⎪ ⎪ ⎪ ⎪ ⎩
dED(p1,q1), if |Q| and |T | is 1
∞, if Q= ∅ or T= ∅
max(dED(p1,q1),
min(dDF(Qh , Th ),dDF(Q, Th ),
dDF(Qh , T ))), otherwise,
dED(p1,q1), if |Q| and |T | is 1
∞, if Q= ∅ or T= ∅
dED(p1,q1)+
dDTW(Q, T )=
⎧ ⎪ ⎪ ⎪ ⎪ ⎪ ⎪ ⎪ ⎨ ⎪ ⎪ ⎪ ⎪ ⎪ ⎪ ⎪ ⎪ ⎪ ⎪ ⎪ ⎩
min(dDTW(Qh , Th ),
dDTW(Q, Th ),
dDTW(Qh , T )), otherwise,
(24)
(25)
where Q and T are trajectories, dED(·
,
·)computes Euclidean
distance between two points, and Qh (resp. Th ) represents the
sub-trajectory starting from the second point of Q (resp. T ).
dWD(Q, T )= ω⊥d⊥(Q, T ) + ω||d||(Q, T )
+ ωθdθ(Q, T ). (26)
Here, d⊥(·
,
·), d||(·
,
·), and dθ(·
,
·) represent the perpendic-
ular, parallel, and angular distances between trajectories,
respectively, and ω⊥, ω||, and ωθ are the corresponding
weights set to 1.
5.5 Evaluation metrics
We utilize two widely recognized clustering metrics, namely
OD (Eq. (1)) and SSE (Eq. (27)), to evaluate the effec-
tiveness of sub-trajectory clustering. OD reflects the degree
of agglomeration between objects and their corresponding
cluster centers, which is highly relevant for center-based
clustering methods such as agglomerative hierarchical clus-
tering or k-means. On the other hand, SSE measures the
proximity of objects within a cluster, making it more per-
tinent for density-based clustering methods like DBSCAN.
By adopting these two metrics, we ensure a comprehensive
and fair comparison between various clustering methods. In
this paper, we calculate OD and SSE using trajectory distance
measurements. Therefore, lower values of both metrics indi-
cate higher clustering quality.
SSE=
k
i=1
⎛ ⎝1
2|Ci |
x∈Ci y∈Ci
d(x, y)2⎞ ⎠. (27)
5.6 Experimental results
Effectiveness We conducted experiments on T-Drive and
Geolife using various sub-trajectory clustering methods and
different distance measurements. To be noted, SubCLUS
leverages properties more specific to the discrete Fréchet dis-
tance, enabling it to achieve an approximate solution while
significantly accelerating the algorithm without compromis-
ing the clustering quality. Therefore, we exclusively evaluate
SubCLUS using dDF in our experiments. Since the large num-
ber of sub-trajectories generated by TRACLUS makes it less
practical, e.g., 33,168 sub-trajectories for 1,000 trajectories,
we evaluate these methods on 1,000 trajectories from the
testing dataset. We apply the N-fold validation method and
report the average clustering performance. Figures 8 and 9
show the OD and SSE of the clustering result for differ-
ent methods, respectively. Our experiments demonstrate that
RLSTC consistently outperforms baselines across different
datasets and distance measurements in terms of all clustering
quality metrics.
On average, RLSTC outperforms other methods by 36%
on T-Drive and 39% on Geolife in terms of OD. For SSE,
the improvements are 57% and 44% on T-Drive and Geo-
life, respectively. Notably, RLSTC outperforms TRACLUS,
CLUSTMOSA, HyperCLUS, and RTCLUS, which separate
segmentation and clustering phases and do not leverage clus-
tering results to improve the trajectory segmentation. Further-
more, RLSTC outperforms S2T-Clustering and SubCLUS,
which rely on complex artificial rules. RLSTC also avoids
the local optima caused by the shortsighted decision-making
property of Greedy Method, demonstrating the effectiveness
of reinforcement learning in sub-trajectory clustering. Addi-
tionally, the minor performance difference between RLSTC
with and without simplification indicates that the trajectory
simplification procedure has a minor impact on clustering
quality.
Efficiency To evaluate the efficiency of different methods,
we measured their running times on T-Drive and Geolife
datasets, accounting for both segmentation and clustering
time. In this experiment, we utilized the dIED distance mea-
surement. As shown in Fig. 10, RLSTC consistently runs
faster than TRACLUS, S2T-Clustering, SubCLUS, Greedy
Method, and CLUSTMOSA, with at least 20% perfor-
mance improvement. The efficiency improvement is even
more apparent with larger trajectory sizes. For example,
RLSTC outperforms TRACLUS and S2T-Clustering by
37% and 24% on T-Drive, respectively, and it outperforms
123
696 A. Liang et al.
Fig. 8 DF DTW WD IED
Clustering quality (OD) of different sub-trajectory clustering methods
Fig. 9 DF DTW WD IED
Clustering quality (SSE) of different sub-trajectory clustering methods
SubCLUS and Greedy Method by 32% and 30% on Geo-
life, respectively. HyperCLUS and RTCLUS are slightly
faster than RLSTC for the following reasons: HyperCLUS
segments trajectories into hypercubes and performs cluster-
ing by checking intersections among hypercubes, which is
faster than computing distances between raw trajectories.
RTCLUS employs Minimum Bounding Rectangles (MBRs)
to represent sub-trajectories and uses an R-tree index, accel-
erating the distance computation during clustering. However,
both HyperCLUS and RTCLUS rely on approximate tra-
jectory representations, neglecting precise calculations on
trajectories. Consequently, they result in worse clustering
performance, as shown in Figs. 8 and 9. It is worth not-
ing that RLSTC w/o simplification runs much slower than
RLSTC but remains faster than TRACLUS on Tdrive and
faster than TRACLUS, S2T-Clustering, CLUSTMOSA, and
Greedy Method on Geolife, which demonstrates that the pre-
processing step can significantly improve efficiency while
maintaining satisfactory clustering quality. Considering the
significant improvement in the clustering quality, RLSTC is
a better choice than Greedy Method. Therefore, we do not
include Greedy Method in the following experiments.
5.7 Effectiveness of segmentation methods
Our research primarily focuses on proposing a clustering-
friendly trajectory segmentation method that can adjust the
segmentation strategy based on clustering quality, applicable
to both center- and density-based clustering methods. There-
fore, our experiments aim to evaluate the performance of
different segmentation methods regarding clustering quality.
Fig. 10 Running time
To achieve this, we apply different segmentation approaches
to generate sub-trajectories and then employ the same clus-
tering algorithm on them. Specifically, we adopt the segmen-
tation methods in RLSTC, TRACLUS, S2T-Clustering, Sub-
CLUS, HyperCLUS, RTCLUS, and CLUSTMOSA, denoted
as RLSTC-S, TRACLUS-S, S2T-Clustering-S, SubCLUS-S,
HyperCLUS-S, RTCLUS-S, CLUSTMOSA-S, to produce
sub-trajectories. We then measure their OD after running the
same clustering algorithm. We run the above procedure for
three different clustering algorithms: Agglomerative Hierar-
chical Clustering (AHC), DBSCAN, and k-means, covering
approaches that specify a cluster number or are density-
based. We use four distance measurements, i.e., dIED, dDF,
dDTW, and dWD, during the segmentation and clustering
phases. The experimental framework is illustrated in Fig. 11.
Notably, the large number of sub-trajectories generated
by TRACLUS-S makes it less practical, e.g., 33,168 sub-
trajectories for 1,000 trajectories. Therefore, we evaluated
different segmentation methods on small datasets. The results
123
Sub-trajectory clustering with deep… 697
Fig. 11 Procedure of evaluating the segmentation methods
on Geolife are omitted because they yield similar observa-
tions.
As shown in Fig. 12, RLSTC-S consistently outperforms
other segmentation methods across different clustering algo-
rithms and distance measurements. In particular, RLSTC-
S surpasses HyperCLUS-S, RTCLUS-S, CLUSTMOSA-S,
TRACLUS-S, S2T-Clustering-S, and SubCLUS-S by 46%,
47%, 45%, 44%, 33%, and 11%, respectively, implying that
RLSTC-S outperforms the methods separating segmentation
and clustering phases, e.g., TRACLUS-S, HyperCLUS-S,
RTCLUS-S and CLUSTMOSA-S, and the methods depend-
ing on hand-crafted heuristics, e.g., S2T-Clustering-S and
SubCLUS-S. The results demonstrate the effectiveness of
our RL-based segmentation approach, which can leverage
clustering quality to guide trajectory segmentation.
Furthermore, the results indicate that RLSTC-S is adapt-
able to various clustering methods, whether they require
specifying the number of clusters or are density-based.
Therefore, RLSTC-S can be seamlessly integrated into
different clustering algorithms, thereby enhancing their per-
formance.
5.8 Parameter study
Impact of ODb In Sect. 4.2, we defined the concept of the
state and introduced the expert knowledge ODb into it. This
variable, ODb, represents OD produced by TRACLUS. We
conducted experiments using RLSTC on T-Drive and Geo-
life, comparing scenarios with and without ODb included in
the state. The results, shown in Fig. 13, demonstrate that clus-
tering performance is significantly better with ODb included
in the state for both datasets. This improvement can be
attributed to the fact that including ODb accelerates the con-
vergence, leading to enhanced clustering performance.
Impact of training data size To evaluate the impact of
training data size, we constructed five training datasets ran-
domly sampled from T-Drive, consisting of 1,000, 2,000,
3,000, 4,000, and 5,000 trajectories, respectively. Addition-
ally, we selected 1,000 trajectories from the remaining data
for testing. We recorded each model’s training time and per-
formance, as shown in Figures 14a and b. The clustering
quality slightly improves (i.e., the OD value decreases) when
the training size increases, but so does the training cost.
The OD value becomes stable as the training size further
increases. Since our objective is achieving low OD within
short training time, we define a metric TR in Eq. (28) to
represent the trade-off between training time and OD:
TR= tnorm + OD, (28)
OD
where tnorm = t·
t. (29)
Eq. (29) normalizes the training time to the scale of OD,
with t representing the training time. The optimal trade-off
is achieved when TR is minimized. As shown in Figure 14c,
TR is minimized when the training set size is 2,000. Conse-
quently, we set the training set size as 2,000 in the remaining
experiments.
Value of k We trained models using different k, i.e., k ∈
{5,8,10,12,14}, and evaluated them on the test dataset using
the same k. Table 2 shows the results. For both datasets,
the model trained using k= 10 achieves the best clustering
quality.
Furthermore, we conducted an experiment to measure the
clustering quality of a pretrained model against different k,
which was trained using a certain k. Table 3 shows the results
for a model trained using k= 10. The results reveal that when
the difference between the tested k and the k for training is
no more than 5, e.g., k= 5, 8, 10, 12, or 15, the variation in
OD is small. In contrast, when the difference exceeds 5, e.g.,
k= 20, 40, 60, 80, or 100, there is a significant increase in the
OD value. The results indicate that when the actual k value
differs from the k for model training by more than 50%, the
model’s performance would deteriorate substantially. Based
on our experimental findings, it is better to conduct a model
retraining when the actual number of clusters differs from
the k for model training by over 50%.
Additionally, as shown in Tables 3 and 4, we observe that
a small sample can produce similar trends to the result on a
complete dataset. Therefore, we can use a small dataset to
check whether the model works well for the actual k. If the
result reveals a substantial deviation in OD, e.g., over 40%,
then we should conduct model retraining on the entire dataset.
This approach offers a practical method for determining the
usability of a model and when it is necessary to retrain the
model.
Number of iterations We conducted experiments to explore
the impact of the number of iterations by randomly sam-
pling 1,000 trajectories from datasets. Figure 15a shows
123
698 A. Liang et al.
DF DTW WD IED
DF DTW WD IED
Fig. 12 DF DTW WD IED
Performance of different segmentation methods
Table 2 Impact of k k values 5 8 10 12 14
OD in T-Drive testing dataset 0.49 0.46 0.21 0.44 0.42
OD in Geolife testing dataset 0.052 0.051 0.023 0.051 0.04
Table 3 Clustering quality for a pretrained model (k= 10) against different k for testing
k= 5 k= 8 k= 10 k= 12 k= 15 k= 20 k= 40 k= 60 k= 80 k= 100
OD in T-Drive
dataset (k= 10)
OD in Geolife
dataset (k= 10)
0.25 0.22 0.21 0.26 0.26 0.51 0.59 0.61 0.60 0.62
0.04 0.04 0.02 0.03 0.04 0.08 0.09 0.10 0.09 0.12
Table 4 Clustering quality for a pretrained model (k= 10) against different k on a small dataset containing 200 sampled trajectories
k= 5 k= 8 k= 10 k= 12 k= 15 k= 20 k= 40 k= 60 k= 80 k= 100
OD in the small T-Drive dataset (k= 10) 0.14 0.19 0.18 0.18 0.19 0.32 0.34 0.34 0.35 0.37
OD in the small Geolife dataset (k= 10) 0.03 0.03 0.02 0.03 0.04 0.06 0.07 0.08 0.08 0.11
123
Sub-trajectory clustering with deep… 699
Fig. 13 Impact of ODb
Fig. 16 Visualization on geolife
Fig. 14 Impact of training data size
sub-trajectories of these clusters. Notably, the clustering cen-
ters exhibit a radial pattern, extending from the center to the
outer areas, which is consistent with the distribution of the
road network. Therefore, the result is reasonable and can be
used to identify hot areas.
Fig. 15 Varying the number of iterations
the results on T-Drive, while results on Geolife are omitted
because they yield similar trends. Initially, the OD fluctuates
during the first nine iterations but stabilizes after the 10th
iteration, indicating that the clustering performance gradu-
ally converges with increasing iterations. Furthermore, we
examined the changes in cluster centers during the iterations.
We used maxdist, denoted as τ in Algorithm 3, to represent
the maximum distance between the cluster centers of consec-
utive iterations. The result is shown in Figure 15b. We can
observe that the centers drastically change at the beginning,
and then, they stabilize after the 10th iteration, meaning that
the clustering process has converged. The results demonstrate
that RLSTC efficiently converges within a small number of
iterations. Additionally, our experiments indicate that OD
stabilizes when τ is set to 0.1.
5.9 Case study by visualization
The sub-trajectory clustering results on Geolife are visual-
ized in Fig. 16, where thin blue lines depict trajectories, and
thick red lines represent representative trajectories, i.e., clus-
ter centers. The visualization shows that five clusters have
been identified, with the cluster centers capturing common
5.10 Discussion
The computation cost of trajectory distance is a bottleneck
in our algorithm. Whenever a point pi is scanned, it neces-
sitates computing the distance between each cluster center
and the sub-trajectory sp ,
···
, pi (where sp denotes the
previous segmentation point), resulting in high computation
costs, particularly for lengthy trajectories. To address this
challenge, we explore potential improvements to our algo-
rithm. First, implementing a grid index can map trajectory
points into grid cells, effectively reducing trajectory length
and computational complexity. By approximating trajectory
distance with the distance between sequences of grid cells,
we can adjust grid cell size to achieve a balance between
efficiency and accuracy. Second, to reduce the computational
cost of evaluating each point as a candidate for segmentation,
we propose augmenting the MDP with additional actions for
skipping certain points. This approach leverages the fact that
successive points often exhibit similar motion trends. For
example, introducing a new action S into the action space
would enable the algorithm to skip points pi +1,
···
, pi +S
when scanning point pi and proceed directly to scanning
point pi +S+1.
123
700 A. Liang et al.
6 Related work
In this section, we briefly review the related work on
trajectory clustering, sub-trajectory clustering, and deep rein-
forcement learning, respectively.
6.1 Trajectory clustering
Trajectory clustering groups trajectories into several clusters
so that trajectories in the same cluster are more similar to each
other than to those in different clusters. It plays a fundamen-
tal role in trajectory mining and analysis tasks [7, 17, 50]
and has been extensively studied for decades. Most existing
trajectory clustering algorithms define a similarity measure-
ment and then employ classical clustering techniques, such
as k-means [21], BIRCH [49], DBSCAN [9], and OPTICS
[4], to group the trajectories. Gaffney et al. [14] proposed
a clustering algorithm based on regression mixture mod-
els, which uses the Expectation-Maximization algorithm to
cluster trajectories. Ferreira et al. [12] presented a trajec-
tory clustering method called Vector Field k-means. They
utilize vector fields to define and represent clusters and to
indicate the similarity between trajectories. Afterward, they
apply the classical k-means method for clustering. Wang et
al. [34] investigated trajectory clustering on road networks
and proposed a novel distance measurement and a scalable
clustering method.
With the rapid development of deep learning, some
researchers utilize deep learning methods to solve the tra-
jectory clustering problem. For example, Yao et al. [44]
transform each trajectory into a fixed-length feature vector
via a sequence-to-sequence auto-encoder and then apply the
k-means algorithm over the feature vectors. Wang et al. [37]
employ network representation learning to learn the low-
dimensional representation vectors of vehicle trajectories and
then utilize machine learning methods to cluster the vehicle
vectors. Fang et al. [11] introduced an end-to-end deep tra-
jectory clustering framework, which is self-training without
extracting manual features. The trajectory clustering problem
does not consider trajectory segmentation, which is essential
to the sub-trajectory clustering problem.
6.2 Sub-trajectory clustering
There are two main categories of work on the sub-trajectory
clustering problem. The first line of work segments tra-
jectories into sub-trajectories according to specific criteria
such as location, direction, speed, and shape [3, 6, 10,
30], ensuring homogeneity within each sub-trajectory. Sub-
sequently, a clustering algorithm is used to group these
sub-trajectories. For instance, Lee et al. [18] proposed the
TRACLUS framework, which includes partition and group
phases. In the partition phase, trajectories are divided into
sub-trajectories based on the MDL principle. In the group
phase, density-based clustering method is performed over
these sub-trajectories. Similarly, Li et al. [20] developed an
incremental clustering algorithm building on TRACLUS.
Dutta et al. [8] employed bearing measurements for tra-
jectory segmentation to identify significant turning points
based on the directional degree at each point. They then
leverage an archived multi-objective simulated annealing
approach for optimal clustering. Xia et al. [42] applied a tra-
jectory segmentation approach considering location, time,
and motion direction variations. They transform each tra-
jectory into a sequence of hypercubes and identify common
sub-trajectories by checking intersections among hypercubes
for clustering. Qiao et al. [28] propose enhancements to
TRACLUS, introducing motion direction as a segmenta-
tion criterion and devising a novel distance measurement
while leveraging the R-tree index to improve efficiency in
clustering sub-trajectories. In these methods, trajectory seg-
mentation and clustering are independent, where clustering
quality does not guide the segmentation. In addition, segmen-
tation criteria are often designed for specific applications,
resulting in a lack of generality. Buchin et al. [5] presented
a sub-trajectories clustering algorithm to detect commuting
patterns in trajectories. However, this method requires users
to define additional cluster parameters, which can be chal-
lenging to set in advance.
Another research approach considers clustering quality
during trajectory segmentation. Pelekis et al. [25] proposed
the S2T-Clustering method, which uses a global voting
algorithm to indicate the local density of sub-trajectories.
Trajectories are then segmented based on this local den-
sity, followed by clustering of the resulting sub-trajectories.
However, this algorithm requires sophisticated preprocess-
ing and a set of complex predefined parameters, leading
to high computational costs. Tampakis et al. [33] expanded
the S2T-Clustering algorithm for distributed environments.
Agarwal et al. [1] introduced pathlet to represent each sub-
trajectory cluster, aiming to find the optimal collection of
pathlets that best represent the trajectories. They present an
objective function for this purpose but show that finding the
optimal pathlet collection is an NP-hard problem. Hence,
they proposed fast approximation algorithms. However, these
algorithms rely on complex artificial rules and are limited
to specific trajectory distance measurements. Zygouras et al.
[52] explored identifying common paths frequently followed
by the given trajectories. Tampakis et al. [32] and Zhang et al.
[48] further researched the distributed sub-trajectory join and
query algorithms, which aim at finding similar parts of tra-
jectories with the query trajectory in real-time. Notably, the
works of [1, 32, 48, 52] focus on identifying shared portions
among trajectories, which differs from our research focus.
123
Sub-trajectory clustering with deep… 701
6.3 Reinforcement learning
Acknowledgements This work was supported by the NSFC (61832017),
Alibaba Group through Alibaba Innovative Research (AIR) Program,
the National Key Research and Development Program of China
(2020YFB1710200), and Hangzhou Qianjiang Distinguished Expert
Program.
Reinforcement learning (RL) is a method that enables an
agent to learn from feedback through trial-and-error interac-
tions with the environment. In RL, the problem to solve is
described as an MDP. RL has been widely applied in various
fields, such as database optimization, traffic light manage-
ment, index construction, and trajectory analysis. In database
optimization, Yu et al. [46] presented a novel learned opti-
mizer utilizing RL with Tree-structured LSTM for join order
selection. Marcus et al. [22] combined tree convolutional
neural networks with an RL algorithm to design a query
optimizer. In index construction, Yang et al. [43] developed
a qd-tree index based on RL methods to reduce I/O costs.
Gu et al. [16] applied RL methods to optimize subtree selec-
tion and node splitting when building an R-tree. For traffic
signal control, Wei et al. [41] reviewed existing RL-based
approaches used in this domain.
In recent years, RL-based solutions have been investigated
for trajectory analysis. For instance, Wang et al. [39] pro-
posed an RL-based algorithm called RLS for sub-trajectory
similarity search. Wang et al. [38] presented an RL-based
solution called RLTS to address the trajectory simplifica-
tion problem. These works differ fundamentally from ours
in terms of MDP design and corresponding algorithm. For
example, RLS models a state by considering the similarity
between the query trajectory and the sub-trajectories, while
RLTS considers the error caused by dropping a point. In
contrast, our work focuses on the distance between cluster
centers and trajectories, making their algorithms cannot be
applied to our problem. Therefore, we do the pioneering work
of proposing an RL-based algorithm for the sub-trajectory
clustering problem.
References
7 Conclusion
In this paper, we study the sub-trajectory clustering prob-
lem and propose a novel RL-based approach. Specifically,
our model is trained to consider clustering quality in guiding
trajectory segmentation, enabling the segmentation and clus-
tering components to cooperate closely and enhance each
other for better clustering results. Compared with existing
methods, our approach is more general and avoids devel-
oping complicated hand-crafted rules. We have conducted
extensive experiments on two real-world trajectory datasets
to evaluate our method. The results demonstrate that our
RL-based algorithm outperforms state-of-the-art methods in
terms of both effectiveness and efficiency. Some open prob-
lems for future work include: (1) How to extend this work
for multi-modal trajectories, such as semantic trajectories
that contain spatial, temporal, and semantic features; and (2)
How to accelerate the training process.
1. Agarwal, P.K., Fox, K., Munagala, K., Nath, A., Pan, J., Taylor, E.:
Subtrajectory clustering: Models and algorithms. In: SIGMOD, pp.
75–87 (2018)
2. Alt, H., Godau, M.: Computing the fréchet distance between two
polygonal curves. Int. J. Comput. Geom. Appl 5(01–02), 75–91
(1995)
3. Anagnostopoulos, A., Vlachos, M., Hadjieleftheriou, M., Keogh,
E., Yu, P.S.: Global distance-based segmentation of trajectories. In:
SIGKDD, pp. 34–43 (2006)
4. Ankerst, M., Breunig, M.M., Kriegel, H.P., Sander, J.: Optics:
Ordering points to identify the clustering structure. ACM SIGMOD
Rec. 28(2), 49–60 (1999)
5. Buchin, K., Buchin, M., Gudmundsson, J., Löffler, M., Luo, J.:
Detecting commuting patterns by clustering subtrajectories. Int. J.
Comput. Geom. Appl. 21(03), 253–282 (2011)
6. Buchin, M., Driemel, A., Van Kreveld, M., Sacristán, V.: Segment-
ing trajectories: a framework and algorithms using spatiotemporal
criteria. J. Spatial Inf. Sci. 3, 33–63 (2011)
7. Chen, L., Gao, Y., Fang, Z., Miao, X., Jensen, C.S., Guo, C.:
Real-time distributed co-movement pattern detection on stream-
ing trajectories. In: Proceedings of the VLDB Endowment (2019)
8. Dutta, S., Das, A., Patra, B.K.: Clustmosa: Clustering for gps trajec-
tory data based on multi-objective simulated annealing to develop
mobility application. Appl. Soft Comput. 130, 109655 (2022)
9. Ester, M., Kriegel, H.P., Sander, J., Xu, X., et al.: A density-based
algorithm for discovering clusters in large spatial databases with
noise. In: SIGKDD, pp. 226–231 (1996)
10. Etemad, M., Júnior, A.S., Hoseyni, A., Rose, J., Matwin, S.: A
trajectory segmentation algorithm based on interpolation-based
change detection strategies. In: EDBT/ICDT Workshops (2019)
11. Fang, Z., Du, Y., Chen, L., Hu, Y., Gao, Y., Chen, G.: E 2 dtc: An
end to end deep trajectory clustering framework via self-training.
In: ICDE, pp. 696–707 (2021)
12. Ferreira, N., Klosowski, J.T., Scheidegger, C.E., Silva, C.T.: Vec-
tor field k-means: Clustering trajectories by fitting multiple vector
fields. In: Computer Graphics Forum, vol. 32, pp. 201–210. Wiley
Online Library (2013)
13. Frentzos, E., Gratsias, K., Theodoridis, Y.: Index-based most sim-
ilar trajectory search. In: ICDE (2007)
14. Gaffney, S., Smyth, P.: Trajectory clustering with mixtures of
regression models. In: SIGKDD, pp. 63–72 (1999)
15. Grünwald, P.D., Myung, I.J., Pitt, M.A.: Advances in minimum
description length: theory and applications (2005)
16. Gu, T., Feng, K., Cong, G., Long, C., Wang, Z., Wang, S.: A rein-
forcement learning based r-tree for spatial data indexing in dynamic
environments. arXiv preprint arXiv:2103.04541 (2021)
17. Lee, J.G., Han, J., Li, X., Gonzalez, H.: Traclass: trajectory
classification using hierarchical region-based and trajectory-based
clustering. VLDB 1(1), 1081–1094 (2008)
18. Lee, J.G., Han, J., Whang, K.Y.: Trajectory clustering: a partition-
and-group framework. In: SIGMOD, pp. 593–604 (2007)
19. Li, Y., Luo, J., Chow, C.Y., Chan, K.L., Ding, Y., Zhang, F.: Grow-
ing the charging station network for electric vehicles with trajectory
data analytics. In: ICDE, pp. 1376–1387 (2015)
123
702 A. Liang et al.
20. Li, Z., Lee, J.G., Li, X., Han, J.: Incremental clustering for trajecto-
ries. In: International conference on database systems for advanced
applications, pp. 32–46 (2010)
21. Lloyd, S.: Least squares quantization in pcm. IEEE Trans. Inf.
Theory 28(2), 129–137 (1982)
22. Marcus, R., Negi, P., Mao, H., Tatbul, N., Alizadeh, M., Kraska, T.:
Bao: Making learned query optimization practical. In: SIGMOD,
pp. 1275–1288 (2021)
23. Meratnia, N., et al.: Spatiotemporal compression techniques for
moving point objects. In: EDBT (2004)
24. Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I.,
Wierstra, D., Riedmiller, M.: Playing atari with deep reinforcement
learning. arXiv preprint arXiv:1312.5602 (2013)
25. Pelekis, N., Tampakis, P., Vodas, M., Panagiotakis, C., Theodoridis,
Y.: In-dbms sampling-based sub-trajectory clustering. In: EDBT,
pp. 632–643 (2017)
26. Puterman, M.L.: Markov decision processes: discrete stochastic
dynamic programming. Wiley, New York (2014)
27. Qian, W.N., Zhou, A.Y.: Analyzing popular clustering algorithms
from different viewpoints. J. Softw. 13(8), 1382–1394 (2002)
28. Qiao, D., Yang, X., Liang, Y., Hao, X.: Rapid trajectory clustering
based on neighbor spatial analysis. Pattern Recogn. Lett. 156, 167–
173 (2022)
29. Schiller, P.L., Kenworthy, J.R.: An introduction to sustainable
transportation: Policy, planning and implementation. Routledge
(2017)
30. Soares Júnior, A., Moreno, B.N., Times, V.C., Matwin, S., Cabral,
L.D.A.F.: Grasp-uts: an algorithm for unsupervised trajectory seg-
mentation. Int. J. Geogr. Inf. Sci. 29(1), 46–68 (2015)
31. Sutton, R.S., Barto, A.G.: Reinforcement learning: An introduc-
tion. MIT Press, Cambridge (2018)
32. Tampakis, P., Doulkeridis, C., Pelekis, N., Theodoridis, Y.: Dis-
tributed subtrajectory join on massive datasets. ACM Trans. Spatial
Algorith. Syst. (TSAS) 6(2), 1–29 (2020)
33. Tampakis, P., Pelekis, N., Doulkeridis, C., Theodoridis, Y.: Scalable
distributed subtrajectory clustering. In: 2019 IEEE international
conference on big data (Big Data), pp. 950–959. IEEE (2019)
34. Wang, S., Bao, Z., Culpepper, J.S., Sellis, T., Qin, X.: Fast large-
scale trajectory clustering. VLDB 13(1), 29–42 (2019)
35. Wang, S., Bao, Z., Culpepper, J.S., Sellis, T., Sanderson, M., Qin,
X.: Answering top-k exemplar trajectory queries. In: ICDE, pp.
597–608 (2017)
36. Wang, S., Shen, Y., Bao, Z., Qin, X.: Intelligent traffic analyt-
ics: from monitoring to controlling. In: Proceedings of the Twelfth
ACM International Conference on Web Search and Data Mining,
pp. 778–781 (2019)
37. Wang, W., Xia, F., Nie, H., Chen, Z., Gong, Z., Kong, X., Wei,
W.: Vehicle trajectory clustering based on dynamic representation
learning of internet of vehicles. IEEE Trans. Intell. Transp. Syst.
22(6), 3567–3576 (2020)
38. Wang, Z., Long, C., Cong, G.: Trajectory simplification with rein-
forcement learning. In: ICDE, pp. 684–695 (2021)
39. Wang, Z., Long, C., Cong, G., Liu, Y.: Efficient and effective sim-
ilar subtrajectory search with deep reinforcement learning. VLDB
13(12), 2312–2325 (2020)
40. Watkins, C.J., Dayan, P.: Q-learning. Mach. Learn. 8(3), 279–292
(1992)
41. Wei, H., Zheng, G., Gayah, V., Li, Z.: Recent advances in reinforce-
ment learning for traffic signal control: A survey of models and
evaluation. ACM SIGKDD Explorat. Newsl. 22(2), 12–18 (2021)
42. Xia, Y., Zhou, L.: Improved clustering algorithm based on hyper-
cube. In: 2022 International Conference on Machine Learning,
Control, and Robotics (MLCR), pp. 32–37 (2022)
43. Yang, Z., Chandramouli, B., Wang, C., Gehrke, J., Li, Y., Minhas,
U.F., Larson, P.Å., Kossmann, D., Acharya, R.: Qd-tree: Learn-
ing data layouts for big data analytics. In: SIGMOD, pp. 193–208
(2020)
44. Yao, D., Zhang, C., Zhu, Z., Huang, J., Bi, J.: Trajectory clustering
via deep representation learning. In: IJCNN, pp. 3880–3887 (2017)
45. Yi, B.K., Jagadish, H.V., Faloutsos, C.: Efficient retrieval of similar
time sequences under time warping. In: ICDE, pp. 201–208 (1998)
46. Yu, X., Li, G., Chai, C., Tang, N.: Reinforcement learning with
tree-lstm for join order selection. In: ICDE, pp. 1297–1308 (2020)
47. Yuan, J., Zheng, Y., Zhang, C., Xie, W., Xie, X., Sun, G., Huang, Y.:
T-drive: driving directions based on taxi trajectories. In: SIGSPA-
TIAL, pp. 99–108 (2010)
48. Zhang, D., Chang, Z., Yang, D., Li, D., Tan, K.L., Chen, K., Chen,
G.: Squid: subtrajectory query in trillion-scale gps database. VLDB
J. pp. 1–18 (2023)
49. Zhang, T., Ramakrishnan, R., Livny, M.: Birch: an efficient data
clustering method for very large databases. ACM SIGMOD Rec.
25(2), 103–114 (1996)
50. Zhang, X., Meng, F., Xu, J.: Perfinsight: A robust clustering-based
abnormal behavior detection system for large-scale cloud. In: IEEE
CLOUD, pp. 896–899 (2018)
51. Zheng, Y., Xie, X., Ma, W.Y., et al.: Geolife: A collaborative social
networking service among user, location and trajectory. IEEE Data
Eng. Bull. 33(2), 32–39 (2010)
52. Zygouras, N., Gunopulos, D.: Corridor learning using individual
trajectories. In: MDM, pp. 155–160 (2018)
Publisher’s Note Springer Nature remains neutral with regard to juris-
dictional claims in published maps and institutional affiliations.
Springer Nature or its licensor (e.g. a society or other partner) holds
exclusive rights to this article under a publishing agreement with the
author(s) or other rightsholder(s); author self-archiving of the accepted
manuscript version of this article is solely governed by the terms of such
publishing agreement and applicable law.
123