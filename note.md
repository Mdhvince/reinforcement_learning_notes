
## How to describe an RL problem 

### The reward hypothesis
All goals can be framed as the maximization of __expected cumulative reward__.  

The cumulative reward or the __return G__ a time step `t` is: 

__`Gₜ = Rₜ₊₁ + Rₜ₊₂ + Rₜ₊₃ + Rₜ₊₄ + ...`__  
  
In order to give more importance to the earlier reward. It is common to multiply the reward by a discount factor &nbsp;&nbsp; `γ [0, 1]` &nbsp;&nbsp; so the __Agent__ can maximize the __discounted return__.

__`Gₜ = Rₜ₊₁ + γRₜ₊₂ + γ²Rₜ₊₃ + γ³Rₜ₊₄ + ...`__  

__`Gₜ =  Σ γ⁽⁽ᵗ⁺¹⁾⁻¹⁾ * Rₜ`__  

- The more __`γ`__ approach __`0`__, the more the agent care about immediate rewards
- The more __`γ`__ approach __`1`__, the more the agent care about future rewards
- It is common to set it at `0.9`

&nbsp;&nbsp;
### MDP: Markov Decision Processe
A (finite) MDP is defined by:  
- a (finite) set of states &nbsp;&nbsp; __𝓢__ &nbsp;&nbsp; (_known by the agent_)  
- a (finite) set of actions &nbsp;&nbsp; __𝓐__ &nbsp;&nbsp; (_known_)  
- a (finite) set of rewards &nbsp;&nbsp; __𝓡__ &nbsp;&nbsp; (_not known, used to describe how the environment works_)  
- a one-step dynamics of the environment &nbsp;&nbsp; __`p(s', r|s, a)`__ &nbsp;&nbsp; (_not known_)  
- A discount rate &nbsp;&nbsp; __γ__ &nbsp;&nbsp; (_known_)  

&nbsp;&nbsp;
---
## How to solve an RL problem 

&nbsp;&nbsp;  
### Policy
A simple policy &nbsp;&nbsp;__`π`__&nbsp;&nbsp; is a mapping of a set of states, to a set of actions: &nbsp;&nbsp; __`π: 𝓢 -> 𝓐`__  
This simple policy is a __deterministic policy__ (input 𝓢 and output 𝓐).  
  
_Example: When in state __s__  choose action __a___

&nbsp;&nbsp;  
The __stochastic policy__ is denoted by &nbsp;&nbsp; __`π(a | s)`__&nbsp;&nbsp; (It takes as input a state-action pair and output the probability that the agent will take this action while in state s)
  
_Example: When in state __s__ choose action __a1__ with __x%__ probability, __a2__ with __y%__ probability..._  

&nbsp;&nbsp; 
### State-Value Functions 
To make sure the agent choose the best Policy, we keep track of all the returns that are likely to happen when starting at state __s__ and following policy __π__ for all time steps.  
  
This mapping between __states__ and __expected returns__ is called a __State-Value function__ of the policy __π__,&nbsp;&nbsp; noted &nbsp;&nbsp; __v_π(s)__ .

__`v_π(s) = E[Gₜ | Sₜ]`__  
  
After knowing the state-value function (all expected returns associated with a state), the agent will choose an action using its policy.  
  
In this case the policy can be : chose always the action that yield the best score __(Greedy policy)__ or explore another option x% of the time __(ε-greedy policy)__.  
  
&nbsp;&nbsp; 
#### Bellman equation 
The value of any state (expected returns starting in that states) is equal to the immediate reward + the discounted value of the state that follow &nbsp;&nbsp; This is called the _- Bellman expectation equation_.  

__```v_π(s) = E[Rₜ₊₁ + γ*v_π(sₜ₊₁) | s]```__  
__EXPECTED__ because :
- Rₜ₊₁ &nbsp;&nbsp; and &nbsp;&nbsp; sₜ₊₁ &nbsp;&nbsp; cannot be known with certainty
  
&nbsp;&nbsp; 
### Action-Value Functions 
This mapping between __states + action__ and __expected returns__ is called a __Action-Value function__ of the policy __π__,&nbsp;&nbsp; noted &nbsp;&nbsp; __q_π(s, a)__ .

__`q_π(s, a) = E[Gₜ | Sₜ. Aₜ]`__  
  
Value functions are estimates so we need a way to make it accuracte so we have the optimal value function `q*` or `v*`
  
&nbsp;&nbsp; 
## Algorithms to help finding optimal policy
  
### Monte-Carlo Methods
A useful question for solving the RL problem using MC methods is to ask:  
__For each state, which action is best ?__  
- 1) Have a table with 1 row per states, 1 column per action  
- 2) For each episode, store the return we've got for each action selected in a state  
__`q(sₜ, aₜ) = E[Gₜ]`__  
- 3) We average the value if we got differents returns with same state-action pair __(Every-visit MC)__ or we can only consider the first value __(First-visit MC)__  
__`q(sₜ, aₜ) += E[Gₜ] / n`__ &nbsp;&nbsp; with __`n`__ being the number of time this state-action pair has been selected  
- 4) After collecting many episodes to fill the table, the Agent can refer to this table to chose the action that yield the max return  
  
The table the Agent is referring to is called a __Q-table__  
here is how it works  

>- initialize dict __Q__ with keys=(state, action) and values=0 by default - _values here are the expected returns_  
>- initialize dict __N__ with keys=(state, action) and values=0 by default - _values here are the number of times the (state, action) pair has been selected_  
>- initialize dict __G__ with keys=(state, action) and values=0 by default - _values here are the sum of all returns collected from a (state, action) pair_  

___an episode is a list of successive (state, action, rewards), so the agent is playing to get those___  
  
>- generate an episode __[(sₜ, aₜ, rₜ₊₁), (sₜ₊₁, aₜ₊₁, rₜ₊₂), (sₜ₊₂, aₜ₊₂, rₜ₊₃)]__  
>- for each time step t of the episode  
&nbsp;&nbsp;&nbsp;&nbsp; __state, action, reward = episode[t]__  
&nbsp;&nbsp;&nbsp;&nbsp; __N[(state, action)] += 1__  
&nbsp;&nbsp;&nbsp;&nbsp; __G[(state, action)] += reward * γ⁽⁽ᵗ⁺¹⁾⁻¹⁾__  
&nbsp;&nbsp;&nbsp;&nbsp; __Q[(state, action)] = G[(state, action)] / N[(state, action)]__  
>- __REDO__ the 2 above steps for a certain number of episodes  
>- __return Q__  
  
We can update the value of __Q[(state, action)]__  in a more efficient way, called __Incremental Mean__:  

&nbsp;&nbsp;&nbsp;&nbsp; __`Q = Q + 1/N * (G - Q)`__  
Here is how it works  

&nbsp;&nbsp;&nbsp;&nbsp; __state, action, reward = episode[t]__  
&nbsp;&nbsp;&nbsp;&nbsp; __N[(state, action)] += 1__  
&nbsp;&nbsp;&nbsp;&nbsp; __G[(state, action)] += reward * γ⁽⁽ᵗ⁺¹⁾⁻¹⁾__  
&nbsp;&nbsp;&nbsp;&nbsp; __Q[(state, action)] += 1/ N[(state, action)] * (G[(state, action)] - Q[(state, action)])__  

  
We can improve even more the update using what we call the __constant alpha α__  
Here is how it works  
>- we have __`Q += 1/N * (G - Q)`__  
>- the term __`(G - Q)`__ can be seen as an __error term δ__  
>- if __δ > 0__ : our estimate lower than expected, so we increase the estimate  
>- if __δ < 0__ : our estimate higher than expected, so we decrease the estimate  
>- decrease/increase by how much ? by &nbsp;&nbsp;__`1/N`__&nbsp;&nbsp; in our algorithm  
>- a better way in to decrease/increase by a constant step-size &nbsp;&nbsp; __`α`__  
  
Our update becomes &nbsp;&nbsp; __`Q += α(G - Q)`__  


This ends the process of __populating the Q-table__, this is called __Policy Evaluation__  

### Temporal-Difference Methods
MC methods needs the episode to end in order to calculate the return and then use it to update the Q table.  
  
The idea with __TD Learning__ is to estimate after each steps the probability of reaching the goal. This concept help in both cases __continuous__ and __episodic__ tasks.  
  
TD Method update using __Sarsa(0)__  
__`Q(Sₜ, Aₜ) += α(Rₜ₊₁ +  γQ(Sₜ₊₁, Aₜ₊₁) - Q(Sₜ, Aₜ))`__ 

How to find `Q(Sₜ₊₁, Aₜ₊₁)` we we are at time step t ?  
We generating an episode we know the sequence of State, Action, Rewards, Next state, action, reward ...  
So we will proceed as follow  
>- get the value of the current state-action paire in the Q-table: `Q(Sₜ, Aₜ)`  
>- get the value of the current reward from the episode: `Rₜ₊₁`  
>- get the next state-action paire from the episode list: `Sₜ₊₁, Aₜ₊₁`  
>- look for its value in the Q-table: `Q(Sₜ₊₁, Aₜ₊₁)`  
>- update: __`Q(Sₜ, Aₜ) += α(Rₜ₊₁ +  γQ(Sₜ₊₁, Aₜ₊₁) - Q(Sₜ, Aₜ))`__  


TD Method update using __SarsaMax__ or __Q-Learning__   
__`Q(Sₜ, Aₜ) += α(Rₜ₊₁ +  γ*maxQ(Sₜ₊₁, Aₜ₊₁) - Q(Sₜ, Aₜ))`__  
The only diffrence is  
>- instead of choosing __`Q(Sₜ₊₁, Aₜ₊₁)`__, we choose the action A that yield the max return when in state  __`Sₜ₊₁`__

## Deep Reinforcement learning: RL in continuous space  

__Discrete Spaces__  
>- State space is finite : {s1, s2, s3 ... sn}  
>- Action space is finite : {a1, a2, a3 ... am}  
>- So their value functions can be reprensented in a table/ dictionnary etc...  


__Continuous Spaces__  
>- Connot represent their value functions in a table/ dictionnary etc...  
So we need to modify our algorithm to deal with continuous spaces. For this, there is 2 techniques:  
>- __Discretization__  
>- __Function approximation__  

#### Discretization
It is simply a method to __convert__ a continuous space into a discrete one. Some techniques to achieve this are:  
>- Tiles and adaptive tiles coding (converting states space into overlapping grid cells)  
>- Coarse codeing (converting states space into overlapping circles/ellipses or spheres in the 3D case)  

#### Function Approximation  
Approximating a __State-Value__ function using a weight parameter (__𝐖__) &nbsp;&nbsp; __`v_hat(s, 𝐖)`__  
  
How ?  
>- Get a feature vector representing the state __`𝐗(s)`__  
>- compute the dot product __`𝐗(s)ᵀ.𝐖`__ to get the scalar value __`v_hat(s, 𝐖)`__   
>- This is called __Linear function approximation__  
  
How to estimate __𝐖__ ?  
We can use gradient descent  
We want to minimize the error __J(𝐖)__ between the __true value function v_π and the estimate 𝐗(s)ᵀ.𝐖__  
Minimize &nbsp;&nbsp; __`J(𝐖) = (v_π(s) - 𝐗(s)ᵀ.𝐖)²`__  

To do so  
>- Find the __gradient of the J(𝐖)__, that means find the derivative of __J(𝐖)__ with respect to __𝐖__.  
Error Gradient of J(𝐖):  __`⛛𝐰J(𝐖) = -2(v_π(s) - 𝐗(s)ᵀ.𝐖) * 𝐗(s)`__  
>- Update rule: __`⛛𝐰 = α(v_π(s) - 𝐗(s)ᵀ.𝐖) * 𝐗(s)`__ with __α__ being the learning rate  
>- We do this for each sample states in order to reduce the error.

Approximating a __Action-Value `q_hat(s, a, 𝐖)`__  
>- Get a feature vector representing the state, action pairs __`𝐗(s, a)`__  
>- Use the same gradient descent method as above  
  

These cases works only for linear problem. To overcome this, we will need to look at __non-linear approximation methods__  
  
For this we can use __kernel functions__ in order to transform our input state __`𝐗(s)`__ or __`𝐗(s, a)`__ into a different space. This way we can catch non-linear relationship between the input state and output value.  
>- A common Kernel function is RBF: Radial basis functions.  

#### Non-Linear Function Approximation: Deep Neural Networks  
Same, but we will use what we call __activation functions 𝓕__  
We can write it as __`v_hat(s, 𝐖) = 𝓕(𝐗(s)ᵀ.𝐖)`__  

### Value-Based Methods  

#### Deep Q-Network (DQN)

__Experience Replay__  

The idea is to interact first with the environment before any kind of learning. At each time step, the agent is in a state __S__,  perform an action __A__, receive the reward __Rₜ₊₁__, end up in a new state __Sₜ₊₁__.
>- interract in order to get several experience tuples: __`<S, A, Rₜ₊₁, Sₜ₊₁>`__   
>- Then all these interraction are stored in a __Replay Buffer__, like a database storing our interractions  
>- Then Sample the buffer (to remove bias and correlation between tuples), get a small batch of it and learn from it.  

So in fact we are __transforming an RL problem to a supervised learning scenario__  

__Fixed Q-Targets__  
In order to add more stability during traning, we need to decouple the target (the weight we are updating) and the parameter we are changing.  
For this, we need to fix the function parameter used to generate the target:  
>- We copy __`𝐖`__ inro __`𝐖⁻`__ (our fixed param, that is not changed during the learning step)  
>- Use of __`𝐖⁻`__ to generate targets while changing __`𝐖`__ during a certain number of epochs
>- Then we update __`𝐖⁻`__ with the latest version of __`𝐖`__ after n epochs  
>- Repeat  



### Some notes on Double-DQN Paper  
The problem: Q-learning or even DQN tend to learn too high action values, because it include a maximization step over estimated action values.
If an error is done during estimation of Q(s, a), it is likely to be overestimated.  
  
if all are Q(s, a) are overestimated, it is not a problem because the relative action preference will stay the same.

#### How is Double-DQN solves this ?
in DQN, the target network action value estimate is written as follow:  
__`Yₜ = Rₜ₊₁ + γ * maxQ_target(Sₜ₊₁, a; 𝐖⁻)`__  
  
so here `Q_target(Sₜ₊₁, a; 𝐖⁻)` in an inference using the target network, then `maxQ_target(Sₜ₊₁, a; 𝐖⁻)` is selecting the action, then all of this is used to update the Q_target(s, a; 𝐖⁻) = `Yₜ` value estimate.  
for short, the DQN uses the same values both to select and to evaluate an action. An this makes it more likely to select overestimated values.  

The idea behind Double Q-learning is to decouple the selection from the evaluation.  
  
- Two value functions are learned by assigning them random experience to update one of the two. So we hawe 2 set of weight `𝐖` and `𝐖⁻`.  
- for each update, one run the inference to get the __action that yield max value__ : __`argmax(Q, (Sₜ₊₁, a; 𝐖))`__  
- and the other run the inference using that action in order the get the action value __`Q(Sₜ₊₁, argmax(Q, (Sₜ₊₁, a; 𝐖), 𝐖⁻)`__  

