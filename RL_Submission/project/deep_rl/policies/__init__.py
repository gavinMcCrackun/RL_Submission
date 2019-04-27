from project.deep_rl.policies.actor_critic import ActorCriticPolicy
from project.deep_rl.policies.ddpg import DDPGPolicy
from project.deep_rl.policies.dqn import DqnPolicy
from project.deep_rl.policies.ppo import PPOPolicy
from project.deep_rl.policies.qlearning import QlearningPolicy
from project.deep_rl.policies.reinforce import ReinforcePolicy

ALL_POLICIES = [
    ActorCriticPolicy,
    DDPGPolicy,
    DqnPolicy,
    PPOPolicy,
    QlearningPolicy,
    ReinforcePolicy
]
