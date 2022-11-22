import gym


gym.register(
    id='simpl-kitchen-v0',
    entry_point='simpl.env.kitchen:KitchenEnv'
)


gym.register(
    id='simpl-kitchen-v1',
    entry_point='simpl.env.kitchen:KitchenEnvOT'
)
