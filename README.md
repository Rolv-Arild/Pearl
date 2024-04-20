# PEARL

## Player Evaluation and Analysis for Rocket League

Heavily inspired by [this blogpost](https://blog.calculated.gg/2022/06/situational-player-value/), this repo includes code for parsing replays and training a next-goal predictor (NGP) along with player masking to estimate situational player value. 

Different from the blogpost is that this repo's NGP is intended to be used to provide [Shapley values](https://en.wikipedia.org/wiki/Shapley_value), meaning it needs to support masking any combination of players. The masking is also done differently, using a mask flag in the player input (and setting the rest of the inputs to 0) instead of using the masking inside the attention mechanism (which this implementation still includes to support any gamemode).

Wandb logs are located [here](https://wandb.ai/rolv-arild/next-goal-predictor).
