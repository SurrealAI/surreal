import itertools


def setup_network(*, agents,
                  evals,
                  ps,
                  replay,
                  learner,
                  tensorplex,
                  loggerplex,
                  tensorboard):
    """
        Sets up the communication between surreal
        components using symphony

        Args:
            agents, evals (list): list of symphony processes
            ps, replay, learner, tensorplex, loggerplex, tensorboard:
                symphony processes
    """
    for proc in itertools.chain(agents, evals):
        proc.connects('ps-frontend')
        proc.connects('collector-frontend')

    ps.binds('ps-frontend')
    ps.binds('ps-backend')
    ps.connects('parameter-publish')

    replay.binds('collector-frontend')
    replay.binds('sampler-frontend')
    replay.binds('collector-backend')
    replay.binds('sampler-backend')

    learner.connects('sampler-frontend')
    learner.binds('parameter-publish')
    learner.binds('prefetch-queue')

    tensorplex.binds('tensorplex')
    loggerplex.binds('loggerplex')

    for proc in itertools.chain(agents, evals, [ps, replay, learner]):
        proc.connects('tensorplex')
        proc.connects('loggerplex')

    tensorboard.exposes({'tensorboard': 6006})
