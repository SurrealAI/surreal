from surreal.launch.setup_network import setup_network


def create_surreal_containerized(exp,
                                 nonagent_image,
                                 agent_image,
                                 cmd_dict,
                                 batched=False):
    """
    TODO: document

    Args:
        exp: [description]
        nonagent_image: [description]
        agent_image: [description]
        cmd_dict: [description]
        batched: [description] (default: {False})
    """
    nonagent = exp.new_process_group('nonagent')
    learner = nonagent.new_process(
        'learner',
        container_image=nonagent_image,
        args=[cmd_dict['learner']])

    # For dm_control
    learner.set_env('DISABLE_MUJOCO_RENDERING', "1")

    replay = nonagent.new_process(
        'replay',
        container_image=nonagent_image,
        args=[cmd_dict['replay']])

    ps = nonagent.new_process(
        'ps',
        container_image=nonagent_image,
        args=[cmd_dict['ps']])

    tensorboard = nonagent.new_process(
        'tensorboard',
        container_image=nonagent_image,
        args=[cmd_dict['tensorboard']])

    tensorplex = nonagent.new_process(
        'tensorplex',
        container_image=nonagent_image,
        args=[cmd_dict['tensorplex']])

    loggerplex = nonagent.new_process(
        'loggerplex',
        container_image=nonagent_image,
        args=[cmd_dict['loggerplex']])
    nonagent.image_pull_policy('Always')

    agents = []
    for i, arg in enumerate(cmd_dict['agent']):
        agent_fmt = 'agent-{}'
        if batched:
            agent_fmt = 'agents-{}'
        agent = exp.new_process(
            agent_fmt.format(i),
            container_image=agent_image,
            args=[arg])

        agents.append(agent)

    evals = []
    for i, arg in enumerate(cmd_dict['eval']):
        eval_fmt = 'eval-{}'
        if batched:
            eval_fmt = 'evals-{}'
        eval_p = exp.new_process(
            eval_fmt.format(i),
            container_image=agent_image,
            args=[arg])

        evals.append(eval_p)

    setup_network(agents=agents,
                  evals=evals,
                  learner=learner,
                  replay=replay,
                  ps=ps,
                  tensorboard=tensorboard,
                  tensorplex=tensorplex,
                  loggerplex=loggerplex)
    return {
        'agents': agents,
        'evals': evals,
        'learner': learner,
        'replay': replay,
        'ps': ps,
        'tensorboard': tensorboard,
        'tensorplex': tensorplex,
        'loggerplex': loggerplex
    }
