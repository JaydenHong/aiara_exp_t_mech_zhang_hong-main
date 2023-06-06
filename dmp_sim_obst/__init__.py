from gymnasium.envs.registration import registry, register, make, spec
register(
    id='dmp-v0',
    entry_point='dmp_sim_obst.envs:DMPEnv'
)
