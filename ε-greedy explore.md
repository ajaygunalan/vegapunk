algorithm_step_template: |
  ## Understanding the ε-Greedy Algorithm: Balancing Exploration and Exploitation
  
  ### Background
  The ε-greedy algorithm step exists to address the *exploration-exploitation dilemma* in sequential decision-making problems, such as reinforcement learning or multi-armed bandits. This dilemma arises because an agent must decide between exploiting the best-known action based on current knowledge to maximize immediate reward and exploring less certain actions to discover potentially better rewards in the future.
  
  Previous methods either exploited too much, risking getting trapped with suboptimal actions, or explored excessively, sacrificing immediate rewards. The ε-greedy approach provides a simple yet effective strategy to trade off these conflicting goals, allowing sustained learning and improving long-term performance.
  
  ### Given
  - **Exploration rate** ($\epsilon$): Probability value in [0,1] that controls how often the algorithm explores, i.e., selects a random action instead of the best estimated one.
  - **Estimated action-values** ($Q(a)$): Current estimates of the expected rewards for each possible action $a$.
  - **Action set** ($\mathcal{A}$): The finite set of possible actions at each decision point.
  
  We know that:
  - With probability $\epsilon$, the algorithm chooses an action uniformly at random from $\mathcal{A}$ (exploration).
  - With probability $1-\epsilon$, it chooses the action $a^* = \arg\max_{a \in \mathcal{A}} Q(a)$ (exploitation).
  
  ### To Find
  The **action selection policy** $\pi$ that balances exploration and exploitation at each step:
  
  From equation:
  $$
  \pi(a) = \begin{cases}
  \frac{\epsilon}{|\mathcal{A}|} & \text{if } a \neq a^* \\
  1 - \epsilon + \frac{\epsilon}{|\mathcal{A}|} & \text{if } a = a^*
  \end{cases}
  $$
  
  Here, $\pi(a)$ denotes the probability of selecting action $a$.
  
  ### Challenge
  The core difficulty is to ensure every action is sampled sufficiently often to accurately estimate its value (exploration), while still primarily choosing the best-known action to gain reward (exploitation). Achieving the right balance mathematically requires a carefully controlled probability $\epsilon$ that neither converges too quickly (which may cause premature exploitation) nor remains too high (which may waste time exploring poor actions). Additionally, in contextual or linear settings, ensuring convergence to optimal actions under limited feedback adds complexity.
  
  ### Approach
  
  #### Core Insight
  > The fundamental insight of ε-greedy is to introduce *randomized choice* with fixed probability $\epsilon$ to guarantee continual exploration, while mostly exploiting by selecting the current best action.
  
  This insight is powerful because it ensures even suboptimal actions are selected infinitely often (in theory), enabling their true value to be discovered over time and avoiding local optima.
  
  #### Method
  Step-by-step procedure to choose an action at time $t$:
  
  1. Generate a random number $r \sim \text{Uniform}(0,1)$.
  2. If $r < \epsilon$, select an action uniformly at random from the full action set $\mathcal{A}$ (exploration).
  3. Else, select the action $a^*_t$ with the highest estimated value $Q_t(a)$ (exploitation).
  4. Execute the selected action and observe the reward and next context/state.
  5. Update $Q_t(a)$ based on the observed outcome.
  6. Repeat for the next decision step.
  
  Mathematically:
  $$
  a_t = \begin{cases}
  \text{random action} & \text{with probability } \epsilon \\
  \arg \max_{a \in \mathcal{A}} Q_t(a) & \text{with probability } 1 - \epsilon
  \end{cases}
  $$
  
  ### Results
  - **Guaranteed exploration:** Each action is sampled infinitely often in the limit, enabling learning of true reward values.
  - **Simple regret bound:** In contextual settings such as linear bandits, ε-greedy achieves a regret bound of order $O(T^{2/3})$ where $T$ is the number of rounds, which is suboptimal compared to more sophisticated algorithms like LinUCB but still meaningful for simple settings.
  - **Impact of decreasing ε:** Making $\epsilon$ decrease over time (e.g., $\epsilon_t \to 0$ as $t \to \infty$) helps convergence by gradually shifting from exploration to exploitation, potentially improving regret rates and matching or closing the gap with algorithms like LinUCB.
  
  ### Summary
  By combining a fixed probability of random exploration with greedy exploitation, the ε-greedy algorithm step balances the conflicting goals of learning new information and maximizing rewards. This transform a pure exploration or pure exploitation approach into a randomized policy ensuring continual improvement and convergence even in complex contextual decision problems.
  
  ---
  Navigation: [[Previous Step]] | [[Epsilon-Greedy Algorithm]] | [[LinUCB Algorithm]]
  Prerequisites: [[Exploration-Exploitation Dilemma]], [[Multi-Armed Bandits]]
  Enables: [[Regret Analysis in Contextual Bandits]], [[Adaptive Exploration Schedules]]