import gym
from easydict import EasyDict
from tabulate import tabulate

from surreal.agent.q_agent import QAgent
from surreal.distributed import *
from surreal.distributed.ps import *
from surreal.env import *
from surreal.model.q_net import FFQfunc
from surreal.replay import *
from surreal.session import *

C = {
    'exploration': { # NOTE: piecewise schedule requires that fraction and final_eps
        # are lists of the same dim that specifies each "piece"
        'schedule': 'linear',
        'steps': 30000,
        'final_eps': 0.01,
    },
}
C = EasyDict(C)

client = RedisClient()
sender = ExpSender(
    client,
    'replay',
    obs_cache_size=5,
    pointers_only=True,
    save_exp_on_redis=False,
)

env = gym.make('CartPole-v0')
env = EpisodeMonitor(env, filename=None)

q_func = FFQfunc(
    input_shape=[4],
    action_dim=2,
    convs=[],
    fc_hidden_sizes=[64],
    dueling=False
)

q_agent = QAgent(
    model=q_func,
    action_mode='train',
    action_dim=2,
)

client = RedisClient()
listener = TorchListener(client, q_agent, debug=0)
listener.run_listener_thread()


def get_exploration_schedule(C):
    if C.exploration.schedule.lower() == 'linear':
        return U.LinearSchedule(
            initial_p=1.0,
            final_p=C.exploration.final_eps,
            schedule_timesteps=int(C.exploration.steps),
        )
    else:
        steps = C.exploration.steps
        final_epses = C.exploration.final_eps
        U.assert_type(steps, list)
        U.assert_type(final_epses, list)
        assert len(steps) == len(final_epses)
        endpoints = [(0, 1.0)]
        for step, eps in zip(steps, final_epses):
            endpoints.append((step, eps))
        return U.PiecewiseSchedule(
            endpoints=endpoints,
            outside_value=final_epses[-1]
        )

exploration = get_exploration_schedule(C)

info_print = PeriodicTracker(100)

obs = env.reset()
# q_agent.set_eval(stochastic=False)
for T in itertools.count():
    # print(binary_hash(q_agent.q_func.parameters_to_binary()))
    action = q_agent.act(
        U.to_float_tensor(obs),
        eps=exploration.value(T)
    )
    new_obs, reward, done, info = env.step(action)
    # replay_buffer.add(new_obs, action, reward, new_obs, float(done))
    sender.send([obs, new_obs], action, reward, done, info)
    obs = new_obs
    if done:
        obs = env.reset()
        if info_print.track_increment():
            # book-keeping, TODO should be done in Evaluator
            info_table = []
            avg_reward = np.mean(env.get_episode_rewards()[-10:])
            info_table.append(['Last 10 rewards', U.fformat(avg_reward, 3)])
            info_table.append(['Exploration',
                               str(int(100 * exploration.value(T)))+'%'])
            avg_speed = 1 / (float(np.mean(env.get_episode_times()[-10:])) + 1e-6)
            info_table.append(['Speed iter/s', U.fformat(avg_speed, 1)])
            info_table.append(['Total steps', env.get_total_steps()])
            info_table.append(['Episodes', len(env.get_episode_rewards())])
            print(tabulate(info_table, tablefmt='fancy_grid'))
