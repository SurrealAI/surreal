from surreal.learner.ddpg import DDPGLearner
from surreal.main.ddpg.halfcheetah_configs import *
from surreal.replay import *

RedisClient().flushall()  # DEBUG ONLY

replay = UniformReplay(
    learn_config=halfcheetah_learn_config,
    env_config=halfcheetah_env_config,
    session_config=halfcheetah_session_config
)

dpg_learner = DDPGLearner(
    learn_config=halfcheetah_learn_config,
    env_config=halfcheetah_env_config,
    session_config=halfcheetah_session_config
)

for i, batch in enumerate(replay.sample_iterator()):
    dpg_learner.learn(batch)
    dpg_learner.writer.add_scalar('replay_size', len(replay), i)

    # if (i+1) % 10 == 0:
    #     dpg_learner.broadcast(message='batch '+str(i))
    dpg_learner.push_parameters(i, message='batch '+str(i))

    # book-keeping, TODO should be done in Evaluator
    if i % 100 == 0:
        print('model updates completed: {}'.format(i))