# Reinforcement Learning in LLMs

Words wander in meaning over time. Tech terms at the height of the hype cycle do so more than most. Reinforcement Learning was initially appropriated by the AI world for RLHF, but now it's starting to move back to its roots.

Today's LLM RL teaches models *how to think* instead of *what to answer*. Traditional fine-tuning shows models examples of good outputs: RL teaches the *process* of generating good outputs. Early CoT prompt engineering showed models *could* think, inspiring RL approaches to train models to think *automatically* and think *well*. Reasoning models have grown out of RL.

Traditional RL (TradRL)[^0] involves continuous environment interaction and actual learning during episodes. LLM RL is more like batch learning from completed episodes - you run the whole thing, get a score, then update the model for next time.

Generalizing, Reinforcement Learning is useful when a system takes a bunch of intermediate steps leading eventually to a valuable end result. "Valuable" meaning "able to be valued" - the reward typically can't be calculated until the entire system is finished.

Since the development of thinking models, a lot of LLM RL applications have been found: reasoning enhancement, codegen, tool use, agentness, game playing, etc. All have in common a pattern where they walk a long path before reaching the destination, and it's hard to trace back what steps were crucial to reaching the end.

## LLM RL in contrast

I have just enough of a classical ML background to be very confused when I started hearing about LLM RL. This started me on a deep dive to discover exactly what makes LLM RL different from everything else.

### LLM pre-training vs LLM RL

Traditional LLM training provides reward per-token. It trains to produce the next token accurately, and to an extent each next-token prediction is graded/optimized for independently of all the others. If the 2nd token is predicted accurately, it doesn't matter what the 10th token prediction is.

RL rewards the model based on the end token(s). The model outputs a bunch of tokens, and then the reward is granted based on what the *end* result is - not the intermediate steps. The reward is then spread out among all the intermediate steps/tokens.

### Traditional RL vs LLM RL

These are kinda different things. Traditional RL (TradRL) involves continuous environment interaction and, well, learning. LLM RL batch learns and then stops learning (usually) - you just use the resultant models.

Reward distribution is also different - TradRL tends to have rewards given continuously at every step, LLM RL runs the entire episode and only then gives you the reward. This relates to the above point. TradRL learns *during* an episode, and the learning is somewhat but not reliably generalizable to other episodes. LLM RL learns *from* an episode, then generalizes to other episodes. TradRL helps within the current episode, LLM RL helps with future episodes.

TradRL has feedback loops within each step (action -> sense response -> next action). LLM RL treats each token as a step, so it skips the 'sense' part (token -> next token).

You do get TradRL heritage in the value and reward functions, credit assignment, and policy gradients. I'm not sure that's enough to steal the label - I think Sutton would be mad, but maybe that's just me.[^1]


### Gradient Descent vs LLM RL

There are conceptual similarities and conceptual differences between these. The objectives are different (minimize loss vs maximize reward) but the model update mechanics are pretty similar.

**Similar**

They both use gradient-based parameter updates (it's in the name).

**Different**

Gradient Descent generally has *exact, differentiable* rewards. The goal is a number, and the ML system changes the number in predictable ways. LLM RL typically has *stochastic, non-differentiable* goals. E.g., "funny" is a kinda vague non-numerical goal, and you can't solve a math equation that will tweak the LLM to be "more funny", because "funniness" isn't a differentiable numerical equation.

Basically gradient descent is a nice comfortable algebra equation with actual solutions and LLM RL is a stochastic probabilistic process where you can estimate it by trying things and seeing what happens.

### Supervised Learning vs LLM RL

The RLHF thread of LLM RL is mechanistically different from supervised learning, but recently researchers have discovered they're effectively equivalent - under certain conditions.

This is a pretty big deal.

**Direct Preference Optimization**

Instead of the classic RLHF pipeline (train reward model, run RL, pray everything works), DPO skips straight to supervised classification on preference data. Instead of a reward model estimating "A is good, and B is gooder", preference data just needs to know `A < B`. Add enough of the alphabet and you start to get something meaningful.

It makes no equivalence claims on anything but the optimal policy. That's great, since we only care about the optimal policy.

This is life-changing for RLHF[^2] and similar problems. Supervised learning is much, *much* easier and cheaper than LLM RL. It's like realizing you can get to the same destination by taking the highway instead of Hannibal'ing over the Alps.

Tragically it can't be used everywher. LLM RL is equivalent to DPO supervised learning when you have preference data of "the right sort".[^20] This is often the case for RLHF, when you're optimizing for alignment/human preferences, since that's intrinsically preference data. It falls apart when you need actual outcome rewards (like "this code compiles" vs "this code doesn't") or when you're doing complex multi-step reasoning where the journey matters more than the destination. Teaching models how to think remains an RL-ish problem.

**No but for real**

Supervised learning is not the same as LLM RL. At all. Except for the problems when they're equivalent, yes, but still. The philosophical problem is different (decision making vs pattern matching) and the objectives are polar opposite (reward maxxing vs loss minning).

That said, the parameter update mechanic of gradients is *kinda* similar, but calculating the gradient for LLM RL can get messy and stochastic if the reward function isn't differentiable. 

## Reward Propagation

There are a few major problems with propagating the reward for RL LLMs:

1. Reward Function: What should the *total* reward be?
2. Credit Assignment: How much of the reward should each *token* get?
3. Propagation Policy: How do we update the model based on this result?

The first two are what makes LLM RL more RL than anything else. RL is a field that has put a lot of thought into solving the problem of actions having an impact on a reward many steps down the chain. Estimating and assigning rewards are generally deeply intertwined.

Gear up, we're about to embark on a series of "it depends" explanations.

### Reward Function 

Some LLM RL applications have pretty straightforward reward functions: binary or numeric. This can be implemented with math solving (right or wrong), tool use (completion + efficiency), and codegen (correctness + style + efficiency + readability).

Reward *estimates* come into play when we're trying to optimize something that can't be computationally defined. LLMs are a human-ish system serving humans, and human desires are generally vague and non-mathematical. Good, funny, efficient, safe, useful - these are tricky to pin down. We can (and do) give them scores/numbers, but in doing so we compress out the complexity of the reaction. It's also typically unclear what actions (tokens/steps) had what impact - or how changing an action would impact the reward. Yay, more stats.

**Reward Estimate Approaches**

A lot of vaguer goals are *preference* related, and thus can be simplified into DPO supervised learning - skipping the reward estimate problem entirely.

When that doesn't apply, TradRL gives a bunch of stochastic approaches to try: Monte Carlo methods are a classic, but nowadays we've also got Evals and can train separate reward models.[^3] There's a bunch of competing methods, the field hasn't settled yet.

**Hierarchical Rewards**

Quick mention that some applications will have sub-goals with their own separate rewards. This is particularly common with agentic/tool use applications that should solve multiple small problems in the process of solving an overarching problem. There are various ways to math out how you want to calculate your total reward based on all the different things that are done.

### Credit Assignment

Each episode has one reward, but many steps(/tokens/actions). To propagate the result properly to a LLM, each token should get its own reward assignment.

Basic approach: fully attribute the end reward to all tokens.[^4]

The more complex approaches require the introduction of a new concept.

**Value Functions**

The *reward* is the final score that's actually received. This isn't received until the episode actually ends. TradRL generally depends on making decisions based on the *expectation* of reward at each step of the episode, calculated with a value function.

In LLM RL, a value function gives an estimate of what the reward will be given the current step.[^5] Since this is LLM-world, a value function is usually a specially-trained predictive model. This can then be used for more complex credit assignment approaches.

**Advantage Assignment**

Advantage calculates how much better than expected you did based on that action. Give each step a score based on how the end result changed from the previous expectation:

`advantage_x = reward - VF_x`

This is a key TradRL idea: estimating how much each token contributed to the final reward. Tokens that typically lead to better outcomes get reinforced more strongly.

**Temporal Difference**

TD is a granular-update method that is very popular in TradRL and kinda useless in LLM RL. When you ask an LLM to explain LLM RL to you, it may try to convince you that TD is used. This is a lie. Your LLM got confused because of how much the RL-related training data referred to TD.[^6]

### Process Rewards

Alternatively, skip the credit attribution and give rewards directly for intermediate steps. This is a new and rather intensive approach.

This breaks from most LLM RL by considering a 'step' to be a *reasoning step* instead of a single token. Humans then give a score to each *reasoning* step.[^7] This score is distributed evenly over all the tokens.

Wonderful for explicitly guiding thinking process, but depends on human judgment of what good thinking is.

### Propagation Policy

The reward is a training thing, the entire point of it is to change the model. Run an iteration, iteration gives you a reward, there's math that tells you how to update the model based on that reward.

That math comes in two parts:

1. The gradient
2. The constraints

The gradient math tells you how the model would move to fulfill the latest iteration. The constraints debounce that movement, to stop the model from jumping all over the place.

**The Gradient**

Some LLM RL can use classic gradient descent methods. 

That math becomes harder when rewards aren't differentiable.[^8] LLM RL solves this with the **Policy Gradient Estimator**, a stats thing that estimates gradients based on Monte Carlo sampling:

`∇J(θ) = E[∇ log π(a|s) × A(s,a)]`

It's very statistical/stochastic, which means that it makes a guess and does math things to try and improve the reliability of the guess by reducing variance. This is in the 'science' area of math - we believe it works because we tested it a lot and it worked kinda well.[^9]

**The Constraints**

It's great to update your ideas when new information is available, but everyone hates that spineless guy who always repeats the views of the last person they talked to.[^10] 

So! The model needs to update a bit, but not too much. The updates are *constrained.* The basic-bitch gradient descent mechanism is to update a percentage; either a fixed percentage, or based on how many rewards have been seen so far.[^11] LLM RL also uses fancier constraints: clipping or KL divergence.

**Proximal Policy Optimization**

PPO is the most popular propagation policy algorithm. Regular PGE gradient descent can make unstable updates when training policies. PPO solves this by "clipping" the policy updates to prevent them from being too large:

`L(θ) = min(ratio × advantage, clip(ratio, 1-ε, 1+ε) × advantage)`

Note that `ratio` can be negative, so we're bounding both the positive and negative directions.

**TRPO (Kullback-Leibler Divergence)**

The most popular algorithm for combining PGE with a KL-based constraint is TRPO.

KL divergence comes up a bunch in LLM RL. It's a measure of how different two probability distributions are.[^12] LLM models are pretty much probability distributions, so we use KL-divergence to figure out how big the gap is between the existing model and the one we'd have if we optimized for this specific iteration. It tells us how big the change would be if we *didn't* constrain it.

We then use this information to proportionately constrain the change. This dampens the effect of *really different* changes. It's a pretty common stats/ML thing ever since Bayes: we have some amount of trust in our existing model,[^13] so if one episode results in a reward that's pretty far removed from what our existing model predicts/expects, we trust it less. 

How much we trust the existing model vs the changes can be varied with a math parameter:

```
Objective = Reward - β * KL_divergence(new_model || original_model)
```

- If β is high: Model stays very close to original (safe but limited improvement)
- If β is low: Model can change more (risky but potentially bigger gains)

The KL constraint controls the prioritization of local-reward-maxing vs consistency.

<hr> 

[^0]: It's a pun.
[^1]: There's an entire section in Reinforcement Learning: An introduction dedicated to making it clear that Supervised Learning Is Not Reinforcement Learning
[^2]: The name gets more ironic by the day
[^3]: These are generally complex and expensive. Monte Carlo involves running the same episode many times to get an average, separate reward models often need to be specially trained.
[^4]: Of course the basic method isn't enough - you don't get a new paper published without inventing something new!
[^5]: Step *generally* means 'token' in LLM RL, but there are some methods that class it differently, e.g.: reasoning step.
[^6]: Temporal-differential methods rely on the system getting an immediate reward for each action. Since this isn't a thing for LLM RL, it's not really applicable.
[^7]: I'm sure we'll have LLM-as-judge in here soon enough, if Constitutional AI isn't already making its way over.
[^8]: Because the math involves a derivative, and derivatives only work on differentiable functions. As in, the definition of a differentiable function is "a function that has a derivative." Delightful!
[^9]: Also because there are math papers proving it mathematically, but those math papers confuse me.
[^10]: It's worst when they're organizing dinner. You say you want pizza, A says they want pizza, B says they want burgers, and somehow you're getting burgers just because B spoke last? Annoying.
[^11]: This is pretty similar to mean-average, but there's a different way to calculate that is less computationally/memory intensive when performed iteratively.
[^12]: Yes, it's another math thing. LLM RL is a collection of math things, and stats things we turn into math things because math things are easier to compute.
[^13]: Or probability distribution, or neural network, or whatever
[^20]: Draco Malfoy's preferred preference data fits the Bradley-Terry model