# Multi-armed Bandit Experimental Design: Online Decision-making and Adaptive Inference 

David Simchi-Levi, Chonghuan Wang<br>Laboratory for Information and Decision Systems, MIT


#### Abstract

Multi-armed bandit has been well-known for its efficiency in online decision-making in terms of minimizing the loss of the participants' welfare during experiments (i.e., the regret). In clinical trials and many other scenarios, the statistical power of inferring the treatment effects (i.e., the gaps between the mean outcomes of different arms) is also crucial. Nevertheless, minimizing the regret entails harming the statistical power of estimating the treatment effect, since the observations from some arms can be limited. In this paper, we investigate the trade-off between efficiency and statistical power by casting the multiarmed bandit experimental design into a minimax multi-objective optimization problem. We introduce the concept of Pareto optimality to mathematically characterize the situation in which neither the statistical power nor the efficiency can be improved without degrading the other. We derive a useful sufficient and necessary condition for the Pareto optimal solutions. Additionally, we design an effective Pareto optimal multiarmed bandit experiment that can be tailored to different levels of the trade-off between the two objectives.


## 1 Introduction
The minimax optimal regret of stochastic MAB has been well understood to be $\widetilde{\Theta}(\log n)$, which can be achieved by the famous upper confidence bound (UCB) based algorithms (see, Lai et al. 1985) and Thompson sampling (TS) based algorithms (see, Thompson 1933).




<details>
<summary><b> Crucial challenges </b></summary>


- MAB algorithms collect data adaptively and consequently, it is not appropriate to consider the collected data as i.i.d. MAB algorithms are effective in learning the optimal online decision-making policy, but lack statistical power. 

- In contrast, some existing well-established ATE estimation methods are recognized for their strong statistical power, but will incur significant regrets. 
    - Specifically, since a predetermined percentage of the population will be assigned to the suboptimal arm, the regret of a traditional RCT will increase linearly with the number of samples. 

- Moreover, the relationship between the two tasks, adaptive inference and online decision-making, may change throughout the experiment. 

    - In order to describe such a trade-off, we introduce the term "Pareto optimality" to characterize the circumstance where neither regret nor estimating error of ATE can be made better off without making the other worse off. 

</details>

### 1.1 Preliminaries
<details>
<summary><b> Basic Setting </b></summary>

- $\mathcal{A}$ of arms, $a \in \mathcal{A}$ with $|\mathcal{A}|=K$. 
- $n$ is the total number of experimental units (or the time horizon). 
- At each time $t$, the environment generates a reward $r_{t}(a)$. 
- After choosing arm $a_{t}$, only the reward of the chosen arm $r_{t}:=r_{t}\left(a_{t}\right)$ can be observed. 
- The expectation $\mathbb{E}\left[r_{t}(a)\right]=$ $\mu_{a} \in[-1,1]$
    - $\mu_{a}$ is the unknown true reward of arm $a$ which is disturbed by an i.i.d. noise to generate $r_{t}(a) \in$ $[-1,1]$
- A stochastic MAB instance can be denoted by $\nu=\left(P_{1}, \cdots, P_{K}\right)$, where $P_{i}$ is the distribution of the rewards of $\operatorname{arm} i$. 
- The optimal arm is the arm with the maximum mean reward denoted by $a^{*}:=\arg \max _{a \in \mathcal{A}} \mu_{a}$
    - The gap between arm $i$ and $\operatorname{arm} j$ as $\Delta^{(i, j)}:=\mu_{i}-\mu_{j}$. 
    - Difference between $\mu_{a^{*}}$ and $\mu_{a}$ as the suboptimality gap, i.e., $\Delta(a):=\mu_{a^{*}}-\mu_{a}$ for $a \in \mathcal{A} \backslash\left\{a^{*}\right\}$. 
- Specifically, when $K=2$, we can define arm 1 to be the treatment of interest and arm 2 to be a control, and thus $\Delta^{(1,2)}$ is the ATE. 
- In this paper, we will elaborate on $\left|\Delta^{(i, j)}\right|=\Theta(1)$ for all $i \neq j \in[K]$, which is arguably the most fundamental case. 
- Denote all stochastic MAB instances satisfying the mentioned assumptions to constitute a feasible set $\mathcal{E}_{0}$. 


- At every time $t$, the decision maker observes the history $\mathcal{H}_{t}=\left(a_{1}, r_{1}, \cdots, a_{t}, r_{t}\right)$. 
- An admissible policy $\pi=$ $\left\{\pi_{t}\right\}_{t \geq 1}$ maps the history $\mathcal{H}_{t-1}$ to an action $a_{t}$. 
    - $\pi_{t}(a)=\mathbb{P}\left(a_{t}=a \mid \mathcal{H}_{t-1}\right)$ under policy $\pi$
- accumulative regret
    - $\mathcal{R}(n, \pi)=\mathbb{E}^{\pi}\left[n \mu_{a^{*}}-\sum_{i=1}^{n} r_{i}\left(a_{i}\right)\right]$
- An admissible adaptive estimator $\hat{\Delta}^{(i, j)}=\left\{\hat{\Delta}_{t}^{(i, j)}\right\}_{t \geq 1}$ maps the history $\mathcal{H}_{t}$ to an estimation of $\Delta^{(i, j)}$ at each time $t$. 
- $\left.e\left(t, \hat{\Delta}^{(i, j)}\right)=\mathbb{E}\left[\left|\Delta^{(i, j)}-\hat{\Delta}_{t}^{(i, j)}\right|\right]\right)$ 
    - measure the quality of the estimation
- $\hat{\Delta}:=\{\hat{\Delta}(i, j)\}_{i<j \leq K}$: all the estimators on the gap between any two arms
- A design of an MAB experiment can then be represented by an admissible pair $(\pi, \hat{\Delta})$. 
</details>


<details>
<summary><b> minimax multi-objective optimization problem </b></summary>

- The optimal design of MAB experiments in this paper is solving the following minimax multi-objective optimization problem:

$$
\min _{(\pi, \hat{\Delta})} \max _{\nu \in \mathcal{E}_{0}}\left(\mathcal{R}_{\nu}(n, \pi), \max _{i<j \leq K} e_{\nu}\left(n, \hat{\Delta}^{(i, j)}\right)\right)
$$
- 
    - minimizing the regret and the inference error under the worst case. 
    - For traditional MAB problems, $\min _{\pi} \max _{\nu \in \mathcal{E}_{0}} \mathcal{R}(n, \pi)$ is usually the only objective. 
    - The asymptotic behavior of $\hat{\Delta}$ is one of the central focuses of ATE literature. 
    - Note that $\pi$ and $\hat{\Delta}$ are complicatedly correlated through the history $\mathcal{H}$.

- start from the inner maximization over all $\nu \in \mathcal{E}_{0}$ 
    - Intuitively, under a given $(\pi, \hat{\Delta})$, the pairs $\left(\mathcal{R}_{\nu}, \max _{i<j \leq K} e_{\nu}\right)$ for all $\nu \in \mathcal{E}_{0}$ can constitute an accessible region, and the front of the accessible region can be defined to be the optimal values of the inner maximum problem (see, Figure 1).  
</details>


<details>
<summary><b> Front </b></summary>

> Definition 1 (Front). The front for $(\pi, \hat{\Delta})$, denoted by $\mathcal{F}(\pi, \hat{\Delta})$, consists of all pairs of $(R, e)$ satisfying: 
> - $(i) \exists \nu \in$ $\mathcal{E}_{0},\left(\mathcal{R}_{\nu}(n, \pi), \max _{i<j \leq K} e_{\nu}\left(n, \hat{\Delta}^{(i, j)}\right)\right)=(R, e)$; 
> - (ii) $\nexists \nu \in \mathcal{E}_{0}, \max _{i<j \leq K} e_{\nu}\left(n, \hat{\Delta}^{(i, j)}\right)>e$ and $\mathcal{R}_{\nu}(n, \pi) \geq$ $R$; 
> - (iii) $\exists \nu \in \mathcal{E}_{0}, \max _{i<j \leq K} e_{\nu}\left(n, \hat{\Delta}^{(i, j)}\right) \geq e$ and $\mathcal{R}_{\nu}(n, \pi)>R$.

- The first condition ensures the achievability of the front. 
- The second and the third conditions
    - there does not exist any instance that will incur no fewer (more) values on both objectives and a strictly larger value on at least one. 
- In Figure 1 , the yellow region and its boundary are an example of the accessible region and front, respectively. 
- RCTs. 
    - Since $\left|\Delta^{(i, j)}\right|=\Theta(1)$ for the instance class $\mathcal{E}_{0}$, the RCTs will usually incur linear regrets.
    - By the theoretical results for RCTs, the best achievable accuracy is $\Theta\left(n^{-\frac{1}{2}}\right)$
    - Thus the front for RCTs is at the point $\left(n, n^{-\frac{1}{2}}\right)$ 
    - The worst-case regret of any policy is no smaller than $\log (n)$, and thus any accessible region will inevitably have some parts on or above the $\log (n)$ line.

</details>



<details>
<summary><b> Pareto dominance </b></summary>


> Definition 2 (Pareto dominance). A feasible solution $\left(\pi_{1}, \hat{\Delta}_{1}\right)$ Pareto dominates another solution $\left(\pi_{2}, \hat{\Delta}_{2}\right)$ 
> - if $\forall\left(R_{1}, e_{1}\right) \in \mathcal{F}\left(\pi_{1}, \hat{\Delta}_{1}\right), \exists\left(R_{2}, e_{2}\right) \in \mathcal{F}\left(\pi_{2}, \hat{\Delta}_{2}\right)$, such that at least one of the following two conditions holds: 
> - (i) $R_{1} \leq R_{2}$ and $e_{1}<e_{2}$ 
> - or (ii) $R_{1}<R_{2}$ and $e_{1} \leq e_{2}$.

The definition formally describes that $\left(\pi_{1}, \hat{\Delta}_{1}\right)$ is Pareto better than $\left(\pi_{2}, \hat{\Delta}_{2}\right)$ if for any point $\left(R_{1}, e_{1}\right)$ on the front of $\left(\pi_{1}, \hat{\Delta}_{1}\right)$, there exists some point $\left(R_{2}, e_{2}\right)$ on the front of $\left(\pi_{2}, \hat{\Delta}_{2}\right)$ such that $\left(R_{1}, e_{1}\right)$ is no larger than $\left(R_{2}, e_{2}\right)$ on both coordinates and is strictly better on at least one coordinate. 
![](https://cdn.mathpix.com/cropped/2023_11_09_dcce4180771ed0085af4g-03.jpg?height=290&width=832&top_left_y=2181&top_left_x=194)
- $\left(\pi_{b}, \hat{\Delta}_{b}\right)$ Pareto dominates $\left(\pi_{a}, \hat{\Delta}_{a}\right)$, and $\left(\pi_{c}, \hat{\Delta}_{c}\right)$ can neither Pareto dominate nor be Pareto dominated by $\left(\pi_{a}, \hat{\Delta}_{a}\right)$ or $\left(\pi_{b}, \hat{\Delta}_{b}\right)$. 

</details>

<details>
<summary><b> Pareto Optimality </b></summary>

> Definition 3 (Pareto Optimality). An admissible pair of $\left(\pi^{*}, \hat{\Delta}^{*}\right)$ is Pareto optimal in terms of the dependence on $n$, if it is not Pareto dominated by any other solution. Pareto frontier denoted as $\mathcal{P}$ is the envelop of the fronts of all the Pareto optimal solutions.

- How to solve the minimax multi-objective optimization problem to get the Pareto optimal solutions. 
- Moreover, $\pi$ and $\hat{\Delta}$ have the measurability constraints and are highly correlated through the history $\mathcal{H}_{t}$, which are hard to explicitly integrate into the optimization problem and endow the feasible region with complicated structures. 
- How to obtain the optimal Pareto optimal solutions given different levels of trade-off in these two objectives.
</details>

</details>


### 1.2 Contributions and Main Results
<details><summary><b> sufficient and necessary condition for the Pareto optimal </b></summary>

- First, we find a sufficient and necessary condition for the Pareto optimal solutions of the minimax multi-objective optimization problem. Specifically, an admissible pair $\left(\pi^{*}, \hat{\Delta}^{*}\right)$ is Pareto optimal if and only if

$$
\max _{\nu \in \mathcal{E}_{0}}\left[\left(\max _{i<j \leq K} e_{\nu}\left(n, \hat{\Delta}^{*(i, j)}\right)\right) \sqrt{\mathcal{R}_{\nu}\left(n, \pi^{*}\right)}\right]=\widetilde{\mathcal{O}}(1)
$$

- - An information theoretical minimax lower bound to portray the trade-off between these two objectives. Specifically,

$$
\inf _{(\pi, \hat{\Delta})} \max _{\nu \in \mathcal{E}_{0}}\left[\left(\max _{i<j \leq K} e_{\nu}\left(n, \hat{\Delta}^{(i, j)}\right)\right) \sqrt{\mathcal{R}_{\nu}(n, \pi)}\right]=\Omega(1)
$$

- - This lower bound tells that no solution can do better on $\left(\max _{i<j \leq K} e_{\nu}\left(n, \hat{\Delta}^{(i, j)}\right)\right) \sqrt{\mathcal{R}_{\nu}(n, \pi)}$ than a constant order in the worst case. 
    - Particularly, since UCB/TS algorithms have the regret upper bound of $\mathcal{O}(\log n)$, then any ATE estimator based on the UCB/TS algorithm cannot avoid an error of $\Omega\left(\frac{1}{\sqrt{\log n}}\right)$ in the worst case. 
    - This explicitly shows the lack of statistical power of UCB/TS algorithms on ATE inference. 
- - On the other hand, we also show that the Pareto frontier is the curve that satisfies $\left(\max _{i<j \leq K} e_{\nu}\left(n, \hat{\Delta}^{(i, j)}\right)\right) \sqrt{\mathcal{R}_{\nu}(n, \pi)}=\widetilde{\Theta}(1)$.
</details>

<details>
<summary><b> efficient Pareto optimal algorithm </b></summary>

- Second, we propose an efficient Pareto optimal algorithm for stochastic MAB experiments which can adapt to different levels of trade-off between the two objectives. 
    - Specifically, we combine the well-known EXP3 algorithm (see, e.g., Auer et al. 2002a, Seldin et al. 2013) 
    - and the idea of extra forced exploration which means that the algorithm purposely plays the other arms after it identifies the best one. 
    - For any given $\alpha \in[0,1]$ as input, we prove that the regret of our algorithm is $\widetilde{\mathcal{O}}\left(n^{1-\alpha}\right)$ and the estimation error of ATE is $\widetilde{\mathcal{O}}\left(\frac{1}{\sqrt{t^{1-\alpha}}}\right)$ for all $t \leq n$, showing its Pareto optimality. 
    - Note that $\alpha$ balances the two objectives. If $\alpha$ is large, the practitioners emphasize the control of regret. Small $\alpha$ will lead to a more accurate ATE estimation. 
    - Technically, a common practice when using the EXP3 algorithm is to estimate the expected rewards of each arm with inverse propensity weighted (IPW) estimators, which imposes the challenge on careful variance control.
</details>

### 1.3 Related Work
<details>
<summary><b>1. Learning Efficiency in MABs.</b></summary>

- A main body of literature in MABs has focused on the online learning efficiency, i.e., minimizing regret.
    - Two representative classes of algorithms that can provide optimal regret bounds, i.e., $\Theta(\log n)$ regret bounds, are UCB-based algorithms and TS-based algorithms. 
    - Both UCB/TS algorithms have been extended to the setting where contextual information of actions exists. 
    - Fan and Glynn (2021) and Simchi-Levi et al. (2022) reveal that efficiency-optimized bandit algorithms may suffer from serious heavy-tailed risk. 
    - In this paper, our design is based on the idea of EXP3, which was initially designed for adversarial MABs Auer et al. (2002b). 
        - Recently, it has gradually gained its own popularity in the stochastic setting and the mixed stochastic-adversarial setting. 
        - The version of Bernstein's inequality we used is inspired by Seldin et al. (2013).    
    - These mentioned works only focus on minimizing the regret. 
- Another growing body of MAB literature is aiming at identifying the best arm.
    - Zhong et al. (2021) carefully study the trade-off between regret minimization and best-arm identification, which is different from our objective. 
    - An emerging field is the multitasking bandit, where minimizing regret is not the only objective. 
    - Erraqabi et al. (2017) also want to balance the trade-off between regret and estimation error. They redefine a new reward function based on the observed rewards and the error bounds. By such a new reward to guide online decision-making, they formulate the problem into a single objective optimization, integrating the two objectives into one. In this way, they do not explicitly capture the trade-off as we do, and thus cannot describe the optimality of their design.
</details>

 
<details>
<summary><b>2. Adaptive experimental design. </b></summary>

- Experimental design is becoming more and more popular in operations research, econometrics, and statistics.
    - MAB itself can also be seen as a type of adaptive experimental design, but here we focus on the designs different from traditional MAB. 
    - Kato et al. (2020) investigate adaptive experiments for ATE when contexts can be observed. 
    - Glynn et al. (2020) propose a theoretical model to study optimal experimental design when temporal interference exists by transforming it into to a Markov decision problem. 
    - Adusumilli (2021) investigates the asymptotic Bayes and minimax risk for bandit experiments. 
    - Farias et al. (2022a) combine synthetic control and MAB to study the settings where experimental units are coarse due to interference or other concerns. 
    - Different from our stationary treatment effect, Qin and Russo (2022) investigate bandit experiments where a potentially nonstationary sequence of contexts influences arms' performance.

</details>

<details>
<summary><b>3. Inference in MABs. </b></summary>

- One of the central tasks along this line is the evaluation of a new policy given historic/observational data which cannot be seen as i.i.d. samples. 
- Bareinboim et al. (2015) study the issue of unobserved confounding in MAB, and consider how the observational data can be used to empower TS algorithms. 
- Dimakopoulou et al. (2021) focus on conducting inference on the true mean of each arm based on data collected by stochastic MAB so far at each step. 
- They incorporate the adaptively weighted doubly robust estimator into TS algorithms, which is proved to achieve the optimal regret and has outstanding empirical performances. 
- Dimakopoulou et al. (2017) and Dimakopoulou et al. (2019) consider the case where context exists and estimate the conditional expectation of each action's reward under different contexts.
</details>

## 2 MAB Experimental Design for $K=2$

We focus on $K=2$ to illustrate our ideas. 
- We first establish the crucial lower bound and the sufficient condition for the Pareto optimality. 
- Then, we propose a series of Pareto optimal designs and show the necessity of the condition based on the constructed Pareto optimal solutions. - For brevity, we adopt the $\Delta$ instead of $\Delta^{(1,2)}$, since there is no ambiguity when $K=2$.

### 2.1 A Lower Bound and A Sufficient Condition


<details>
<summary><b> A lower bound  </b></summary>
 

> Theorem 1. For any admissible pair $\left(\pi, \hat{\Delta}_{n}\right)$, there always exists a hard instance $\nu \in \mathcal{E}_{0}$ that $e_{\nu}\left(n, \hat{\Delta}_{n}\right) \sqrt{\mathcal{R}_{\nu}(n, \pi)}$ is no less than a constant order, i.e.,
> $$
\inf _{\left(\pi, \hat{\Delta}_{n}\right)} \max _{\nu \in \mathcal{E}_{0}}\left[e_{\nu}\left(n, \hat{\Delta}_{n}\right) \sqrt{\mathcal{R}_{\nu}(n, \pi)}\right]=\Omega(1)
$$

- Theorem 1 states that for any admissible pair $\left(\pi, \hat{\Delta}_{n}\right)$, there usually exists a challenging instance $\nu \in \mathcal{E}$ such that the product of estimation error and regret is lower-bounded by $n^{p}$ for some positive value of $p$. 
- This mathematically highlights the trade-off between the two objectives. 
    - A small regret will inevitably have a large error on the ATE estimation.
    -  Roughly speaking, the expected error is almost lower bounded by the inverse of the square root of the regret in the worst case, i.e., $e_{\nu}\left(n, \hat{\Delta}_{n}\right)=\Omega\left(\frac{1}{\sqrt{\mathcal{R}_{\nu}(n, \pi)}}\right)$. 
    - In particular, since $\mathcal{R}_{\nu}(n, \pi)=\mathcal{O}(\log (n))$ for UCB and TS algorithms, no estimators can not achieve smaller error than the order $\Omega\left(\frac{1}{\sqrt{\log (n)}}\right)$ consistently over all the possible instances. 
    - Although $\log (n)$ increases with $n$, the speed is rather slow which explicitly shows the limitation of regretoptimal policies in terms of statistical power for estimating the ATE.

- In Theorem 1, we have shown that no solution can perform better than a constant order in terms of $e_{\nu}\left(n, \hat{\Delta}_{n}\right) \sqrt{\mathcal{R}_{\nu}(n, \pi)}$ in the worst case.
</details>

<details>
<summary><b> A sufficient condition  </b></summary>
 

- The following theorem states one policy is Pareto optimal if it can achieve the constant order on $e_{\nu}\left(n, \hat{\Delta}_{n}\right) \sqrt{\mathcal{R}_{\nu}(n, \pi)}$ in terms of the dependence on $n$.

> Theorem 2. An admissible pair $(\pi, \hat{\Delta})$ is Pareto optimal if
> $$
\max _{\nu \in \mathcal{E}_{0}}\left[e_{\nu}(n, \hat{\Delta}) \sqrt{\mathcal{R}_{\nu}(n, \pi)}\right]=\widetilde{\mathcal{O}}(1)
$$

- Consider the traditional RCTs where a half of experimental units are treated and controlled, respectively. 
    - For any $\nu \in \mathcal{E}_{0}, e_{\nu}\left(n, \hat{\Delta}_{\mathrm{RCT}}\right)=\widetilde{O}(1 / \sqrt{n})$ 
    - and $\mathcal{R}_{\nu}\left(n, \pi_{\mathrm{RCT}}\right)=\Theta(n)$, 
    - and thus they are Pareto optimal.

</details>

### 2.2 An Algorithm and A Necessary Condition



### 2.2.1 Algorithm and regret upper bound

- We adopt the idea of the famous EXP3 algorithm for adversarial MAB 
    - together with the idea to force the algorithm to actively explore the suboptimal arm, 
    - to design our EXP3 with exploration (EXP3E) algorithm shown in Algorithm 1.

- We first define a set of random variables $\hat{R}_{t}(a)$ for $a \in$ $\{1,2\}$ based on inverse propensity score weight (IPW) as: 
    - $\hat{R}_{t}(a)=\hat{R}_{t-1}(a)+\frac{R_{t}}{\pi_{t}(a)} \mathbb{I}_{a=A_{t}}$, 
    - which can provide an unbiased estimation of $\mu_{a}$ after being divided by $t$, 
    - i.e., $\mathbb{E}\left[\hat{R}_{t}(a)\right]=\mu_{a} t$. 
- We also define $\hat{R}_{t}^{\max }$ as $\max _{a \in\{1,2\}} \hat{R}_{t}(a)$. 
    - One may think a more straightforward way to estimate $\mu_{a}$ is the simple sample average $\frac{\sum_{s=1}^{t} \mathbb{I}_{a=A_{t}} R_{t}}{\sum_{s=1}^{t} \mathbb{I}_{a=A_{t}}}$. 
    - However, such an estimator is neither unbiased nor asymptotically normal 
        - because whether we take action $a$ at time $t$ is highly correlated with the past history as is pointed by the recent works.
    - Thus, the ATE based on the simple sample average will inevitably be biased. 
    - The first phase: identifying the best arm with well-controlled regret. 
        - In this phase, the algorithm is adaptively polishing its decision policy to gain confidence about which arm is the optimal one,
        - according to the estimated reward $\hat{R}_{t}(a)$. 
        - There are many ways to map $\hat{R}_{t}(a)$ into probabilities,
            - among which a popular choice is exponential weighting as $\pi_{t}(a)=\frac{e^{\varepsilon_{t-1} \hat{R}_{t-1}(a)}}{\sum_{a \in \mathcal{A}} e^{\varepsilon_{t-1} \hat{R}_{t-1}(a)}}$. 
        - Note that the decision maker knows the exactly $\pi_{t}(a)$,
        - different from the classical offline ATE inference. 
    - If at time $t$ there exists an arm $a$ such that $\hat{R}_{t}(a)$ is larger than the other by at least $\Omega(\sqrt{t})$, 
        - the algorithm believes $a$ is the optimal arm and eliminates the other arm. 
    - Formally, our elimination rule is 
        - $\mathcal{A}_{t+1}=\mathcal{A}_{t} \backslash\left\{a \in \mathcal{A}_{t}: \hat{R}_{t}^{\max }-\hat{R}_{t}(a)>2 \sqrt{C t}\right\}$, 
        - where $C$ is a constant defined in Algorithm 1. 
    - Note that when the first phase ends is a stopping time with respect to the history $\mathcal{H}_{t}$. 
        - We define two stopping times as $\tau(a)=$ $\max \left\{t: a \in \mathcal{A}_{t}\right\}$ for $a \in\{1,2\}$, 
        - and then the first phase ends after $\min _{a \in\{1,2\}} \tau(a)$ periods. 
        - By a careful analysis, the length of the first phase can be shown in the order $1 / \Delta^{2}$.

![](https://cdn.mathpix.com/cropped/2023_11_09_dcce4180771ed0085af4g-06.jpg?height=726&width=805&top_left_y=346&top_left_x=1067)

- After eliminating the suboptimal arm, EXP3E operates into the second phase. 
    - The algorithm is forced to play the arm which was identified as the suboptimal one in the first phase with a carefully controlled probability $\alpha_{t}=\frac{1}{2 t^{\alpha}} $.
    - $\alpha$ is an important input parameter that balances our two tasks. 
        - a small $\alpha$ can help the algorithm to have a more accurate estimator of $\Delta$, 
        - while sacrificing the regret.

> Theorem 3. Let Algorithm 1 runs with any given $\alpha \in[0,1]$ and $\delta=\frac{1}{2 n^{2}}$. The regret is
> $$
\mathcal{O}\left(\frac{\log (n)}{\Delta}+n^{1-\alpha} \Delta \log (n)\right)
$$

- The regret bound in Theorem 3 decreases with $\alpha$, 
    - which is consistent with our intuition that a large $\alpha$ restricts the probability to play the suboptimal arm in the second phase. When $\alpha=1$, the regret bound in Theorem 3 becomes $\mathcal{O}\left(\frac{\log (n)}{\Delta}+\Delta \log (n)\right)$, which matches with the optimal regret bound of MAB in current literature (see, e.g., Lattimore and Szepesvári 2020) up to logarithmic factors. This means that if minimizing the accumulative regret is the only objective (i.e., ignoring the inference task), by setting $\alpha=1$, the performance of our EXP 3E is unimprovable in terms of the dependency on the learning horizon $n$. Another extreme case is when $\alpha=0$, the regret upper bound grows linearly with the learning horizon $T$. When $\alpha$ is set to be 0 , in the second phase, the exploration probability remains to be $\frac{1}{2}$. This indicates that in the second phase EXP 3E is doing random control trials. Moreover, if $|\Delta|=\Theta(1)$, Theorem 3 has an immediate corollary.

Corollary 1. With any given $\alpha \in[0,1], \delta=\frac{1}{2 n^{2}}$ and $|\Delta|=$ $\Theta(1)$, the regret of Algorithm 1 is $\widetilde{\mathcal{O}}\left(n^{1-\alpha}\right)$.

### 2.2.2 Inference for ATE

Now, we are going to focus on the problem of inference of $\Delta$. Since we have shown $\mathbb{E}\left[\hat{R}_{t}(a)\right]=\mu_{a} t$, we can define a set of martingales as $M_{t}^{a}=\hat{R}_{t}(a)-\mu_{a} t$ for $a \in \mathcal{A}$, and $M_{t}^{(1,2)}:=M_{t}^{1}-M_{t}^{2}=t \hat{\Delta}_{t}-t \Delta$. We can directly have,

Theorem 4. For $t \in[n], \hat{\Delta}_{t}$ is unbiased, i.e., $\mathbb{E}\left[\hat{\Delta}_{t}\right]=\Delta$.

Then, how well our $\hat{\Delta}_{t}$ can estimate $\Delta$ becomes the central problem that we need to solve. For any $t \in[n]$, the martingale difference of $M_{t}^{(1,2)}$ can be bounded by $\left|M_{t}^{(1,2)}-M_{t-1}^{(1,2)}\right|=2+1 / \pi_{t}(1)+1 / \pi_{t}(2)$. Moreover, the variance of the martingale $M_{t}^{(1,2)}$ can be bounded as $\sum_{t=1}^{n} \mathbb{E}\left[\left(\frac{R_{t}}{\pi_{t}(1)} \mathbb{I}_{A_{t}=1}-\frac{R_{t}}{\pi_{t}(2)} \mathbb{I}_{A_{t}=2}-\Delta\right)^{2} \mid \mathcal{H}_{t-1}\right] \leq$ $\sum_{t=1}^{n} \frac{1}{\pi_{t}(1)}+\frac{1}{\pi_{t}(2)}$. A common concern about IPW-based method is its large variance, especially when $\pi_{t}(a)$ is small (e.g., Dimakopoulou et al. 2021). The blessing of online learning is that we can control the propensity score. However, it also imposes an important challenge. Assigning a small probability to the suboptimal arm can bring us a small regret but a large variance of the estimators which may harm the inference. In the second phase of our EXP 3E, $\frac{1}{\pi_{t}(1)}+\frac{1}{\pi_{t}(2)} \leq 2 t^{\alpha}$ by the design of our algorithm. As for the first phase, from the proof of Theorem 3, our elimination rules contribute to securing $\frac{1}{\pi_{t}(a)} \leq 1+e^{2}$ for $a \in \mathcal{A}$. In this way, we control the variance of $M_{t}^{(1,2)}$. By Bernstein's inequality (Freedman 1975), we can have the following theorem.

Theorem 5. If Algorithm 1 runs with $\alpha \in[0,1]$ and $\delta<$ $2 / e$, with probability at least $1-\delta$, for all $t \in[n]$,

$$
\left|\hat{\Delta}_{t}-\Delta\right| \leq \frac{\left(8 e^{2}+16\right) \log \frac{2}{\delta}}{\sqrt{t^{1-\alpha}}}
$$

Furthermore, since we can take $\delta=\frac{1}{2 n^{2}}, e(n, \hat{\Delta})=$ $\mathbb{E}\left[\left|\hat{\Delta}_{n}-\Delta\right|\right]=\widetilde{\mathcal{O}}\left(\frac{1}{\sqrt{n^{1-\alpha}}}\right)$.

Different from most existing results on adaptive estimators in the inference literature that are asymptotic in nature, Theorem 5 is a finite sample properties which has its own advantages when the number or samples are relatively small. Another important difference with offline inference lies in that Theorem 5 is an any-time bound since it holds for all $t \in[n]$. In addition, the RHS of Eq. (5) increases with the value of $\alpha$. Intuitively, when $\alpha$ is small, $\operatorname{EXP} 3 \mathrm{E}$ is more likely to explore during the second phase of our algorithm, which will help to estimate of the reward of the suboptimal arm, and thus improve our inference. However, there is no free lunch. In contrast, the regret upper bound in Theorem 3 will be large. Such observations illustrate the role that $\alpha$ plays in balancing online learning and inference.

Together with Corollary 1 and Theorem 5, we can safely draw the following statement.
Corollary 2. For any instance $\nu \in \mathcal{E}_{0}$ and $\alpha \in[0,1]$, Algorithm 1 can guarantee $e_{\nu}(n, \hat{\Delta}) \sqrt{\mathcal{R}_{\nu}(n, \pi)}=\widetilde{\mathcal{O}}(1)$. Furthermore, by Theorem 1, Algorithm 1 achieves the Pareto optimality for any $\alpha \in[0,1]$.

For simplicity, we denote the online decision-making policy and ATE estimator with input parameter $\alpha$ in Algorithm 1 as $\left(\pi_{\alpha}, \hat{\Delta}_{\alpha}\right)$. By Theorem 1, the front of $\left(\pi_{\alpha}, \hat{\Delta}_{\alpha}\right)$ is $\mathcal{F}\left(\pi_{\alpha}, \hat{\Delta}_{\alpha}\right)=\left\{\left(n^{1-\alpha}, \frac{1}{\sqrt{n^{1-\alpha}}}\right)\right\}$. With different input of $\alpha \in[0,1]$, the front of our $\left(\pi_{\alpha}, \hat{\Delta}_{\alpha}\right)$ will cover the line $e(n, \hat{\Delta}) \sqrt{\mathcal{R}_{\nu}(n, \pi)}=1$, which is the Pareto frontier. The front for $\alpha=0$ coincides with that of RCTs.

### 2.2.3 The sufficient and necessary condition

Based on the series of $\left(\pi_{\alpha}, \hat{\Delta}_{\alpha}\right)$, the following theorem illustrates the necessity of the condition (3). The main idea is that any policy violating condition (3) will be Pareto dominated by $\left(\pi_{\alpha}, \hat{\Delta}_{\alpha}\right)$ for some $\alpha$, and thus can not be Pareto optimal.

Theorem 6. Any Pareto optimal $\left(\pi^{*}, \hat{\Delta}^{*}\right)$ satisfies $\max _{\nu \in \mathcal{E}_{0}}\left[e_{\nu}\left(n, \hat{\Delta}^{*}\right) \sqrt{\mathcal{R}_{\nu}\left(n, \pi^{*}\right)}\right]=\widetilde{\mathcal{O}}(1)$.

Together with Theorems 2 and 6 , we formally confirm the condition (3) is sufficient and necessary for the Pareto optimal solutions for $K=2$.

Corollary 3. The admissible pair $\left(\pi^{*}, \hat{\Delta}^{*}\right)$ is Pareto optimal to the minimax multi-objective optimization (1) if and only if $\max _{\nu \in \mathcal{E}_{0}}\left[e_{\nu}\left(n, \hat{\Delta}^{*}\right) \sqrt{\mathcal{R}_{\nu}\left(n, \pi^{*}\right)}\right]=\widetilde{\mathcal{O}}(1)$.

In Figure 3, we present several examples of the possible fronts of the admissible policies. First, the lower bound in Eq. (2) tells that the front of any admissible policy has intersection of the region $\mathcal{R}(n, \pi) \gtrsim 1 /(e(n, \widehat{\Delta}))^{2}$ (the blue region). The sufficient and necessary condition indicates that the Pareto optimal solutions will only intersect with the blue region on the boundary $\mathcal{R}(n, \pi) \simeq 1 /(e(n, \hat{\Delta}))^{2}$. In turn, any policy that intersects with the region on the boundary are Pareto optimal. The red curve is non Pareto optimal, since it partly falls into the interior of the region.

![](https://cdn.mathpix.com/cropped/2023_11_09_dcce4180771ed0085af4g-07.jpg?height=452&width=691&top_left_y=1991&top_left_x=1151)

Figure 3: Examples of Pareto (non)optimal solutions.

## 3 Extension to General $K$

In this section, we extend our model, algorithm and analysis to a general $K \geq 2$. The main ideas of EXP3EG in Algorithm 2 follow from those of EXP 3E. However, since the suboptimal arms are usually unlikely to be eliminated at the same time, EXP 3EG can not be divided into two phases explicitly anymore. Following the same notatition as before, we assign $a \notin \mathcal{A}_{t}$ a time-varying but fixed probability $\alpha_{t}=\frac{1}{K t^{\alpha}}$ to be played. For $a \in \mathcal{A}_{t}$, we assign the probability $\left(1-\left|\mathcal{A}_{t}^{c}\right| \alpha_{t}\right) \frac{e^{\varepsilon_{t-1} \hat{R}_{t-1}(a)}}{\sum_{a^{\prime} \in \mathcal{A}_{t}} e^{\varepsilon_{t-1} \hat{R}_{t-1}\left(a^{\prime}\right)}}$. The elimination rule is still $\mathcal{A}_{t+1}=\mathcal{A}_{t} \backslash\left\{a: \hat{R}_{t}^{\max }-\hat{R}_{t}(a)>2 \sqrt{C t}\right\}$. Moreover, notably, EXP 3EG is powerful enough to output $\hat{\Delta}_{i, j}$ for all $i \neq j \in[K]$ at the same time, which means EXP3EG does not need to know which $\Delta_{i, j}$ is of interest in advance. We can have the following theorem on regret.

Theorem 7. Let Algorithm 2 run with $\alpha \in[0,1]$ and $\delta=\frac{1}{2 n^{2}}$. The regret is $\mathcal{O}\left(\sum_{a \in[K] /\left\{a^{*}\right\}} \frac{\log (n)}{\Delta(a)}+\right.$ $\left.\Delta(a) n^{1-\alpha} \log (n)\right)$.

When $\alpha=1$, the regret upper bound in Theorem 7 matches with the minimax lower bounds for MAB problems up to a logarithmic factor. Also, since we mainly care about $|\Delta(a)|=\Theta(1)$, the regret bounds becomes $\mathcal{O}\left(n^{1-\alpha}\right)$. In Theorem 7, the regret upper bound is only dependent on the gaps with the optimal arm $\Delta(a)$ instead of all $\left|\Delta^{(i, j)}\right|$. Intuitively, $\Delta(a)$ has a more important role than $\left|\Delta^{(i, j)}\right|$, since the regret is defined to compete with the optimal arm.

For inference, following the notation defined in Section 2.2, we introduce a series of martingales $M_{t}^{(i, j)}:=M_{t}^{i}-M_{t}^{j}$ for any $i \neq j \in[K]$, where recall that $M_{t}^{i}=\hat{R}_{t}(i)-$ $\mu_{i} t$. An immediate result is unbiasedness, i.e., $\mathbb{E}\left[\hat{\Delta}_{t}^{(i, j)}\right]=$ $\Delta^{(i, j)}$ for $t \in[n]$. We extend the result in Theorem 5 as following to a fixed pair of $i, j$.

![](https://cdn.mathpix.com/cropped/2023_11_09_dcce4180771ed0085af4g-08.jpg?height=621&width=753&top_left_y=1809&top_left_x=205)

Theorem 8. If Algorithm 2 runs with $\alpha \in[0,1]$, for any fixed $i, j \in[k], i \neq j$, and $\delta<2 / e$, with probability at least $1-\delta$, for all $t \in[n]$,

$$
\left|\hat{\Delta}_{t}^{(i, j)}-\Delta^{(i, j)}\right| \leq \frac{4\left(2+2 K^{2}\left(1+e^{2}\right)\right) \log \frac{2}{\delta}}{\sqrt{t^{1-\alpha}}} .
$$

Particularly, after taking $\delta=\frac{1}{2 n^{2}}$, we can derive $\max _{i<j \leq K} e\left(n, \hat{\Delta}_{n}^{(i, j)}\right)=\mathcal{O}\left(\frac{1}{\sqrt{n^{1-\alpha}}}\right)$.

Then, together with Theorem 7, $\left(\max _{i<j \leq K} e_{\nu}\left(n, \hat{\Delta}_{n}^{(i, j)}\right)\right)$ $\sqrt{\mathcal{R}_{\nu}(n, \pi)}=\widetilde{O}(1)$ holds for any $\nu \in \mathcal{E}_{0}$. Taking $i^{*}$ and $j^{*}$ to be the best and the second best arm respectively, we always have $\max _{i<j \leq K} e\left(n, \hat{\Delta}_{n}^{(i, j)}\right) \geq e\left(n, \hat{\Delta}_{n}^{\left(i^{*}, j^{*}\right)}\right)$. By such a fact, we can reduce the problem with $K>2$ to $K=2$. Theorem 2 can be easily generalized, and thus the sufficient condition for Pareto optimality can be $\max _{\nu \in \mathcal{E}_{0}}\left(\max _{i<j \leq K} e_{\nu}\left(n, \hat{\Delta}_{n}^{(i, j)}\right)\right) \sqrt{\mathcal{R}_{\nu}(n, \pi)}=\widetilde{O}(1)$. Hence, Algorithm 2 is Pareto optimal for all $\alpha \in[0,1]$. Then the necessity of the condition follows similarly from Theorem 6 . We can naturally extend Corollary 3 as follows.

Theorem 9. The admissible pair $\left(\pi^{*}, \hat{\Delta}^{*}\right)$ is Pareto optimal to the optimization problem (1) if and only if $\max _{\nu \in \mathcal{E}_{0}}\left[\left(\max _{i<j \leq K} e_{\nu}\left(n, \hat{\Delta}_{n}^{*(i, j)}\right)\right) \sqrt{\mathcal{R}_{\nu}\left(n, \pi^{*}\right)}\right]=\widetilde{\mathcal{O}}(1)$.

## 4 Discussion

In the previous sections, we restrict ourselves to the instance class $\mathcal{E}_{0}$, where $\Delta^{(i, j)}=\Theta(1)$ for all $i, j \in[K]$, which is usually referred to as the "well-separated" instance class (see, e.g., Kalvit and Zeevi 2021). Such an instance class allows us to see the magnitude of $\Delta^{(i, j)}$ as a universal constant independent of $n$ and ignore its influence when deriving the necessary and sufficient condition for Pareto optimality. In this section, we first discuss about the case where $\Delta^{(i, j)}$ is extremely small comparing with the time horizon $n$ or can even shrink with $n$ (i.e., $\Delta^{(i, j)}=\mathcal{O}\left(n^{-p}\right)$ for some strictly positive $\left.p>0\right)$. For simplicity, we will focus on $K=2$ since $K>2$ naturally follows.

Case 1: $\Delta=\mathcal{O}\left(n^{-1 / 2}\right)$. As known in current literature (see, e.g., Lattimore and Szepesvári 2020 and Kalvit and Zeevi 2021), approximately $\frac{1}{\Delta^{2}}$ samples are unavoidable to distinguish between two distributions with means separated by $\Delta$. This indicates that if $\Delta=\mathcal{O}\left(n^{-1 / 2}\right)$, even the most efficient adaptive algorithm which only cares about the regret rate will spend $\Theta(n)$ samples on the suboptimal arm and thus the regret is doomed to be roughly $n \Delta$. No one can expect to increase the statistical power by sacrificing the online decision-making efficiency, which itself has no room to be sacrificed. Therefore, the main question becomes what level the estimation error can be controlled. By slight modifications of the proof of Theorem 5, we can show our estimator $\hat{\Delta}$ can achieve $e(n, \hat{\Delta})=\widetilde{O}\left(n^{-1 / 2}\right)$ when $\Delta=\mathcal{O}\left(n^{-1 / 2}\right)$, which can not be further improved
either. In this case, we have the strongest statistical power and an unavoidably large regret.

Case 2: $\Delta=\Omega\left(n^{-1 / 2}\right)$ and $\Delta=\mathcal{O}\left(n^{-(1-\alpha) / 2}\right)$. Recall that $\alpha$ is the input of EXP3E controlling the trade-off between the two objectives. In this case, the regret upper bound in Theorem 3 reduces to $\widetilde{\mathcal{O}}\left(\frac{\log (n)}{\Delta}\right)$, which is not influenced by $\alpha$ and matches with the regret lower bound in the worst case (see, e.g., Lattimore and Szepesvári 2020). This means that when $n^{-1 / 2} \lesssim \Delta \lesssim n^{-(1-\alpha) / 2}$, our EXP3E always has the optimal efficiency in online decision-making. Note that Theorem 5 still holds here, i.e., $e(n, \hat{\Delta})=\widetilde{O}\left(n^{-(1-\alpha) / 2}\right)$. We want to point out that by a simple modification of Lemma ??, in this case one feasible lower error bound is $\Omega\left(\sqrt{\frac{\Delta}{\log (n)}}\right)$, which we do not match with. Such kind of mismatch may be caused by the large variance of IPW-based estimator, especially during the second phase of the algorithm. It is also possible that $\Omega\left(\sqrt{\frac{\Delta}{\log (n)}}\right)$ underestimates the difficulty of the problem with $n^{-1 / 2} \lesssim \Delta \lesssim n^{-(1-\alpha) / 2}$. We leave this issue to our future research.

Case 3: $\Delta=\Omega\left(n^{-(1-\alpha) / 2}\right)$ and $\Delta<\Theta(1)$. In this case, Theorem 3 offers a regret upper bound of $\widetilde{\mathcal{O}}\left(n^{1-\alpha} \Delta\right)$, whose proof inplies that the algorithm plays the suboptimal arm exact $\Theta\left(n^{1-\alpha}\right)$ times. And thus, by an easy extension of Lemma ??, the $\widetilde{O}\left(n^{-(1-\alpha) / 2}\right)$ error bound offered by Theorem 5 is rate optimal. The problem here is that since we do not know the order of the magnitude of $\Delta$ in advance, we can hardly control the regret to the level that we want in the order of $n$. That is the price for the strong statistical power when $\Delta=\Omega\left(n^{-(1-\alpha) / 2}\right)$.

Up till now, we have discussed that EXP3E is somewhat optimal in some other senses under different orders of the small $\Delta$. However, what is the sufficient and necessary condition for the desired Pareto optimal for extremely small $\Delta$ is still unknown and we leave it to our future work.

The second aspect to consider is the neglect of constants and logarithm terms in defining Pareto optimality. While this is a common practice in $\mathrm{MAB}$, ignoring these terms can be significant, especially in cases where sample collection is costly or the number of samples is limited. Admittedly, such a simplification is another limitation of our work. It is very challenging to get optimal rates on the dependence of both constant and $n$. Even for the traditional MAB problem without the added complexity of inference, there is a lack of research addressing the best achievable dependence on constants. Another important consideration is the choice of the parameter $\alpha$, which plays a crucial role in our design. Choosing the appropriate $\alpha$ can be challenging in some cases, and there may be scenarios where a dynamic $\alpha$ would be more appropriate, especially in experiments that have a long duration. Understanding how people make decisions about choosing $\alpha$ is also important, but it is outside the scope of this work. Finally, a crucial next step is extending our work to continuous arm bandit problems, as our design and analysis are currently limited to discrete arms.

## 5 Concluding Remarks

In this paper, we statistically investigate the trade-off between efficiency in decision-making and statistical power of ATE in MAB experiments. We novelly introduce the general minimax multi-objective optimization framework and Pareto optimality to formally describe and theoretically analyze such a trade-off. Moreover, we derive a useful sufficient and necessary condition for Pareto optimal designs, i.e., $\left(\max _{i<j \leq K} e_{\nu}\left(n, \hat{\Delta}^{*(i, j)}\right)\right) \sqrt{\mathcal{R}_{\nu}\left(n, \pi^{*}\right)}=\widetilde{\mathcal{O}}(1)$ for any instance $\nu \in \mathcal{E}_{0}$. Additionally, we propose an efficient Pareto optimal design with $\max _{i<j \leq K} e_{\nu}\left(n, \hat{\Delta}^{(i, j)}\right)=$ $\mathcal{O}\left(n^{-(1-\alpha) / 2}\right)$ and $\mathcal{R}_{\nu}(n, \pi)=\mathcal{O}\left(n^{1-\alpha}\right)$ for any give $\alpha \in[0,1]$ controlling the desired level of trade-off.

## References

Adusumilli, K. (2021). Risk and optimal policies in bandit experiments. arXiv preprint arXiv:2112.06363.

Agrawal, R. (1995). Sample mean based index policies by o $(\log \mathrm{n})$ regret for the multi-armed bandit problem. Advances in Applied Probability, 27(4):1054-1078.

Agrawal, S., Koolen, W. M., and Juneja, S. (2021). Optimal best-arm identification methods for tail-risk measures. Advances in Neural Information Processing Systems, 34:25578-25590.

Angrist, J. and Imbens, G. (1995). Identification and estimation of local average treatment effects.

Aronow, P. M. and Samii, C. (2017). Estimating average causal effects under general interference, with application to a social network experiment. The Annals of Applied Statistics, 11(4):1912-1947.

Atan, O., Zame, W. R., and Schaar, M. (2019). Sequential patient recruitment and allocation for adaptive clinical trials. In The 22nd International Conference on Artificial Intelligence and Statistics, pages 1891-1900. PMLR.

Athey, S., Eckles, D., and Imbens, G. W. (2018). Exact pvalues for network interference. Journal of the American Statistical Association, 113(521):230-240.

Athey, S. and Wager, S. (2021). Policy learning with observational data. Econometrica, 89(1):133-161.

Auer, P. (2002). Using confidence bounds for exploitationexploration trade-offs. Journal of Machine Learning Research, 3(Nov):397-422.

Auer, P., Cesa-Bianchi, N., and Fischer, P. (2002a). Finitetime analysis of the multiarmed bandit problem. Machine learning, 47(2-3):235-256.

Auer, P., Cesa-Bianchi, N., Freund, Y., and Schapire, R. E. (2002b). The nonstochastic multiarmed bandit problem. SIAM journal on computing, 32(1):48-77.

Bareinboim, E., Forney, A., and Pearl, J. (2015). Bandits with unobserved confounders: A causal approach. Advances in Neural Information Processing Systems, 28.

Bhat, N., Farias, V. F., Moallemi, C. C., and Sinha, D. (2020). Near-optimal ab testing. Management Science, 66(10):4477-4495.

Bibaut, A., Dimakopoulou, M., Kallus, N., Chambaz, A., and van Der Laan, M. (2021). Post-contextual-bandit inference. Advances in neural information processing systems, 34:28548-28559.

Bojinov, I., Rambachan, A., and Shephard, N. (2021). Panel experiments and dynamic causal effects: A finite population perspective. Quantitative Economics, 12(4):1171-1196.

Bojinov, I., Simchi-Levi, D., and Zhao, J. (2020). Design and analysis of switchback experiments. arXiv preprint arXiv:2009.00148.

Bubeck, S. and Slivkins, A. (2012). The best of both worlds: Stochastic and adversarial bandits. In Conference on Learning Theory, pages 42-1. JMLR Workshop and Conference Proceedings.

Carpentier, A., Lazaric, A., Ghavamzadeh, M., Munos, R., and Auer, P. (2011). Upper-confidence-bound algorithms for active learning in multi-armed bandits. In International Conference on Algorithmic Learning Theory, pages 189-203. Springer.

Chan, H. P. and Lai, T. L. (2006). Sequential generalized likelihood ratios and adaptive treatment allocation for optimal sequential selection. Sequential Analysis, 25(2):179-201.

Chapelle, O. and Li, L. (2011). An empirical evaluation of thompson sampling. Advances in neural information processing systems, 24.

Chen, N., Gao, X., and Xiong, Y. (2022). Debiasing samples from online learning using bootstrap. In International Conference on Artificial Intelligence and Statistics, pages 8514-8533. PMLR.

Chu, W., Li, L., Reyzin, L., and Schapire, R. (2011). Contextual bandits with linear payoff functions. In Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics, pages 208-214. JMLR Workshop and Conference Proceedings.

Deshmukh, A. A., Dogan, U., and Scott, C. (2017). Multitask learning for contextual bandits. Advances in neural information processing systems, 30 .

Dimakopoulou, M., Ren, Z., and Zhou, Z. (2021). Online multi-armed bandits with adaptive inference. Advances in Neural Information Processing Systems, 34.
Dimakopoulou, M., Zhou, Z., Athey, S., and Imbens, G. (2017). Estimation considerations in contextual bandits. arXiv preprint arXiv:1711.07077.

Dimakopoulou, M., Zhou, Z., Athey, S., and Imbens, G. (2019). Balanced linear contextual bandits. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pages 3445-3453.

Dudík, M., Erhan, D., Langford, J., and Li, L. (2014). Doubly robust policy evaluation and optimization. Statistical Science, 29(4):485-511.

Dudík, M., Langford, J., and Li, L. (2011). Doubly robust policy evaluation and learning. arXiv preprint arXiv:1103.4601.

Eckles, D., Karrer, B., and Ugander, J. (2017). Design and analysis of experiments in networks: Reducing bias from interference. Journal of Causal Inference, 5(1).

Erraqabi, A., Lazaric, A., Valko, M., Brunskill, E., and Liu, Y.-E. (2017). Trading off rewards and errors in multiarmed bandits. In Artificial Intelligence and Statistics, pages 709-717. PMLR.

Fan, L. and Glynn, P. W. (2021). The fragility of optimized bandit algorithms. arXiv preprint arXiv:2109.13595.

Farajtabar, M., Chow, Y., and Ghavamzadeh, M. (2018). More robust doubly robust off-policy evaluation. In International Conference on Machine Learning, pages 1447-1456. PMLR.

Farias, V., Moallemi, C., Peng, T., and Zheng, A. (2022a). Synthetically controlled bandits. arXiv preprint arXiv:2202.07079.

Farias, V. F., Li, A. A., Peng, T., and Zheng, A. T. (2022b). Markovian interference in experiments. arXiv preprint arXiv:2206.02371.

Filippi, S., Cappe, O., Garivier, A., and Szepesvári, C. (2010). Parametric bandits: The generalized linear case. In Advances in Neural Information Processing Systems, pages 586-594.

Freedman, D. A. (1975). On tail probabilities for martingales. the Annals of Probability, pages 100-118.

Gabillon, V., Ghavamzadeh, M., and Lazaric, A. (2012). Best arm identification: A unified approach to fixed budget and fixed confidence. Advances in Neural Information Processing Systems, 25.

Garivier, A. and Cappé, O. (2011). The kl-ucb algorithm for bounded stochastic bandits and beyond. In Proceedings of the 24th annual conference on learning theory, pages 359-376. JMLR Workshop and Conference Proceedings.

Garivier, A. and Kaufmann, E. (2016). Optimal best arm identification with fixed confidence. In Conference on Learning Theory, pages 998-1027. PMLR.

Garivier, A. and Moulines, E. (2011). On upper-confidence bound policies for switching bandit problems. In International Conference on Algorithmic Learning Theory, pages 174-188. Springer.

Glynn, P. W., Johari, R., and Rasouli, M. (2020). Adaptive experimental design with temporal interference: A maximum likelihood approach. Advances in Neural Information Processing Systems, 33:15054-15064.

Hadad, V., Hirshberg, D. A., Zhan, R., Wager, S., and Athey, S. (2021). Confidence intervals for policy evaluation in adaptive experiments. Proceedings of the $\mathrm{Na}$ tional Academy of Sciences, 118(15).

Hahn, J., Hirano, K., and Karlan, D. (2011). Adaptive experimental design using the propensity score. Journal of Business \& Economic Statistics, 29(1):96-108.

Jennison, C., Johnstone, I. M., and Turnbull, B. W. (1982). Asymptotically optimal procedures for sequential adaptive selection of the best of several normal means. In Statistical decision theory and related topics III, pages 55-86. Elsevier.

Johari, R., Li, H., Liskovich, I., and Weintraub, G. Y. (2022). Experimental design in two-sided platforms: An analysis of bias. Management Science.

Johari, R., Pekelis, L., and Walsh, D. J. (2015). Always valid inference: Bringing sequential analysis to $\mathrm{a} / \mathrm{b}$ testing. arXiv preprint arXiv:1512.04922.

Kallus, N. and Zhou, A. (2018). Confounding-robust policy improvement. Advances in neural information processing systems, 31 .

Kalvit, A. and Zeevi, A. (2021). A closer look at the worst-case behavior of multi-armed bandit algorithms. Advances in Neural Information Processing Systems, 34:8807-8819.

Kasy, M. and Sautmann, A. (2021). Adaptive treatment assignment in experiments for policy choice. Econometrica, 89(1):113-132.

Kato, M. and Ariu, K. (2021). The role of contextual information in best arm identification. arXiv preprint arXiv:2106.14077.

Kato, M., Ishihara, T., Honda, J., Narita, Y., et al. (2020). Efficient adaptive experimental design for average treatment effect estimation. arXiv preprint arXiv:2002.05308.

Kaufmann, E., Korda, N., and Munos, R. (2012). Thompson sampling: An asymptotically optimal finite-time analysis. In International conference on algorithmic learning theory, pages 199-213. Springer.

Khan, S. and Ugander, J. (2021). Adaptive normalization for ipw estimation. arXiv preprint arXiv:2106.07695.

Lai, T. L., Robbins, H., et al. (1985). Asymptotically efficient adaptive allocation rules. Advances in applied mathematics, 6(1):4-22.
Lattimore, T. and Szepesvári, C. (2020). Bandit algorithms. Cambridge University Press.

Li, L., Lu, Y., and Zhou, D. (2017). Provably optimal algorithms for generalized linear contextual bandits. In Proceedings of the 34th International Conference on Machine Learning-Volume 70, pages 2071-2080. JMLR. org.

Li, L., Munos, R., and Szepesvári, C. (2015). Toward minimax off-policy value estimation. In Artificial Intelligence and Statistics, pages 608-616. PMLR.

Luedtke, A. R. and Van Der Laan, M. J. (2016). Statistical inference for the mean outcome under a possibly nonunique optimal treatment strategy. Annals of statistics, 44(2):713.

Mannor, S. and Tsitsiklis, J. N. (2004). The sample complexity of exploration in the multi-armed bandit problem. Journal of Machine Learning Research, 5(Jun):623-648.

Nie, X., Tian, X., Taylor, J., and Zou, J. (2018). Why adaptively collected data have negative bias and how to correct for it. In International Conference on Artificial Intelligence and Statistics, pages 1261-1269. PMLR.

Offer-Westort, M., Coppock, A., and Green, D. P. (2021). Adaptive experimental design: Prospects and applications in political science. American Journal of Political Science, 65(4):826-844.

Qin, C. and Russo, D. (2022). Adaptivity and confounding in multi-armed bandit experiments. arXiv preprint arXiv:2202.09036.

Russo, D. and Van Roy, B. (2014). Learning to optimize via posterior sampling. Mathematics of Operations Research, 39(4):1221-1243.

Russo, D. and Van Roy, B. (2016). An informationtheoretic analysis of thompson sampling. The Journal of Machine Learning Research, 17(1):2442-2471.

Seldin, Y., Auer, P., Shawe-taylor, J., Ortner, R., and Laviolette, F. (2011). Pac-bayesian analysis of contextual bandits. Advances in neural information processing systems, 24.

Seldin, Y., Cesa-Bianchi, N., Auer, P., Laviolette, F., and Shawe-Taylor, J. (2012). Pac-bayes-bernstein inequality for martingales and its application to multiarmed bandits. In Proceedings of the Workshop on On-line Trading of Exploration and Exploitation 2, pages 98-111. JMLR Workshop and Conference Proceedings.

Seldin, Y., Szepesvári, C., Auer, P., and Abbasi-Yadkori, Y. (2013). Evaluation and analysis of the performance of the exp3 algorithm in stochastic environments. In European Workshop on Reinforcement Learning, pages 103116. PMLR.

Simchi-Levi, D., Zheng, Z., and Zhu, F. (2022). A simple and optimal policy design with safety against
heavy-tailed risk for multi-armed bandits. arXiv preprint arXiv:2206.02969.

Sutton, R. S. and Barto, A. G. (2018). Reinforcement learning: An introduction. MIT press.

Swaminathan, A. and Joachims, T. (2015). Batch learning from logged bandit feedback through counterfactual risk minimization. The Journal of Machine Learning Research, 16(1):1731-1755.

Thompson, W. R. (1933). On the likelihood that one unknown probability exceeds another in view of the evidence of two samples. Biometrika, 25(3/4):285-294.

Villar, S. S., Bowden, J., and Wason, J. (2015). Multiarmed bandit models for the optimal design of clinical trials: benefits and challenges. Statistical science: a review journal of the Institute of Mathematical Statistics, 30(2):199.

Wager, S. and Xu, K. (2021). Experimenting in equilibrium. Management Science, 67(11):6694-6715.

Wainwright, M. J. (2019). High-dimensional statistics: A non-asymptotic viewpoint, volume 48. Cambridge University Press.

Wang, Y.-X., Agarwal, A., and Dudık, M. (2017). Optimal and adaptive off-policy evaluation in contextual bandits. In International Conference on Machine Learning, pages 3589-3597. PMLR.

Xiong, R., Athey, S., Bayati, M., and Imbens, G. (2019). Optimal experimental design for staggered rollouts. arXiv preprint arXiv:1911.03764.

Xu, M., Qin, T., and Liu, T.-Y. (2013). Estimation bias in multi-armed bandit algorithms for search advertising. Advances in Neural Information Processing Systems, 26.

Yang, F., Ramdas, A., Jamieson, K. G., and Wainwright, M. J. (2017). A framework for multi-a (rmed)/b (andit) testing with online fdr control. Advances in Neural Information Processing Systems, 30.

Yao, J., Brunskill, E., Pan, W., Murphy, S., and DoshiVelez, F. (2021). Power constrained bandits. In Machine Learning for Healthcare Conference, pages 209259. PMLR.

Zhan, R., Hadad, V., Hirshberg, D. A., and Athey, S. (2021). Off-policy evaluation via adaptive weighting with data from contextual bandits. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery \& Data Mining, pages 2125-2135.

Zhang, K., Janson, L., and Murphy, S. (2020). Inference for batched bandits. Advances in neural information processing systems, 33:9818-9829.

Zhang, K., Janson, L., and Murphy, S. (2021). Statistical inference with m-estimators on adaptively collected data. Advances in neural information processing systems, 34:7460-7471.
Zhong, Z., Cheung, W. C., and Tan, V. Y. (2021). On the pareto frontier of regret minimization and best arm identification in stochastic bandits. arXiv preprint arXiv:2110.08627.

Zhou, Z., Athey, S., and Wager, S. (2022). Offline multiaction policy learning: Generalization and optimization. Operations Research.

