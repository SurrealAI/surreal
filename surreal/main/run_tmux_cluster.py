import surreal.utils as U
from surreal.session import TmuxCluster
from surreal.main.cartpole_configs import cartpole_session_config
from collections import OrderedDict
import pprint


cluster = TmuxCluster(
    cluster_name='cartpole',
    session_config=cartpole_session_config,
    agent_script='surreal.main.run_cartpole_agent',
    learner_script='surreal.main.run_cartpole_learner',
    evaluator_script=None,
    dry_run=0,
)

def get_stdout(group, window, history):
    for win, out in cluster.get_stdout(
            group=group,
            window=window,
            history=history
    ).items():
        print('='*20, win, '='*20)
        print(out)

i=0

if i==0:
    N = range(0, 35)
    cluster.launch(
        agent_names=['yo'+str(i) for i in N],
        agent_args=[str(i) for i in N],
    )
elif i==1:
    get_stdout('infras', None, 0)
    get_stdout('agent', 'yo2', 40)
elif i==2:
    N = range(5, 35)
    cluster.add_agents(
        agent_names=['yo'+str(i) for i in N],
        agent_args=[str(i) for i in N],
    )
elif i==3:
    cluster.kill_agents(
        agent_names=list(range(5,35)),
    )
elif i==4:
    cluster.killall()
    U.f_remove('~/Temp/cartpole')
elif i==5:
    for win, out in cluster.check_error().items():
        print('='*20, win, '='*20)
        print(out)
