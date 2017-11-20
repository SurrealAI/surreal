from surreal.session import TmuxCluster
from surreal.main.cartpole_configs import cartpole_session_config
import pprint


cluster = TmuxCluster(
    cluster_name='cartpole',
    session_config=cartpole_session_config,
    agent_script='surreal.main.run_cartpole_agent',
    learner_script='surreal.main.run_cartpole_learner',
    evaluator_script=None,
    dry_run=0,
)


i=1
if i==0:
    cluster.launch(
        agent_names=list(range(5)),
        agent_args=['yo'+str(i) for i in range(10, 15)],
    )
elif i==1:
    for win, out in cluster.get_stdout(
        group='redis',
        window=None,
        history=0
    ).items():
        print('='*20, win, '='*20)
        print(out)
    print(cluster.get_stdout(group='agent', window=1, history=350))
elif i==2:
    cluster.add_agents(
        agent_names=list(range(10,35)),
        agent_args=['yo'+str(i) for i in range(20, 45)],
    )
elif i==3:
    cluster.kill_agents(
        agent_names=list(range(5,35)),
    )
elif i==4:
    cluster.killall()
elif i==5:
    for win, out in cluster.check_error().items():
        print('='*20, win, '='*20)
        print(out)
