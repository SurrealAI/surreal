from easydict import EasyDict
from tabulate import tabulate

from surreal.agent.q_agent import QAgent
from surreal.distributed import *
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
client.flushall()
sender = ExpSender(client, 'replay', local_obs_cache_size=5)

env = wrap_deepmind(make_atari('Pong'))
env = EpisodeMonitor(env, filename=None)
action_dim = env.action_space.n

q_func = FFQfunc(
    input_shape=[4, 84, 84],
    action_dim=action_dim,
    convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
    fc_hidden_sizes=[512],
    dueling=False,
    is_uint8=True
)

q_agent = QAgent(
    model=q_func,
    agent_mode='train',
    action_dim=action_dim,
)

client = RedisClient()
listener = TorchListener(client, q_func, q_agent.get_lock(), debug=0)
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

# TODO make all reset return `info`
obs, info = env.reset()
frames = info.pop('frames')
# q_agent.set_eval(stochastic=False)
for T in itertools.count():
    # print(binary_hash(q_agent.q_func.parameters_to_binary()))
    action = q_agent.act(U.to_float_tensor(obs), vectorize=True)
    q_agent.eps = exploration.value(T)
    new_obs, reward, done, info = env.step(action)
    new_frames = info.pop('frames')
    sender.send(frames + new_frames, action, reward, done, info)
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
            avg_speed = 1 / (float(np.mean(env.get_episode_duration()[-10:])) + 1e-6)
            info_table.append(['Speed iter/s', U.fformat(avg_speed, 1)])
            info_table.append(['Total steps', env.get_total_steps()])
            info_table.append(['Episodes', len(env.get_episode_rewards())])
            print(tabulate(info_table, tablefmt='fancy_grid'))


ATARI_CONFIG = {
    'lr': 1e-4,
    'max_timesteps': int(5e7),
    'buffer_size': int(1e6),
    'train_freq': 4,
    'batch_size': 32,
    'gamma': .99,
    'target_network_update_freq': int(4e4),

    'exploration': {
        'schedule': 'piecewise',
        'fraction': [1/50., 1/5.],
        'final_eps': [0.1, 0.01],
    },
}

