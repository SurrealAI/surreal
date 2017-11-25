import surreal.utils as U
from surreal.main.dqn_cartpole.configs import cartpole_session_config
from surreal.session import TmuxCluster

cluster = TmuxCluster(
    cluster_name='cartpole',
    session_config=cartpole_session_config,
    agent_script='surreal.main.dqn_cartpole.run_agent',
    learner_script='surreal.main.dqn_cartpole.run_learner',
    eval_script='surreal.main.dqn_cartpole.run_eval',
    dry_run=0,
)

i=0

if i==0:
    N = range(0, 16)
    cluster.launch(
        agent_names=['yo'+str(i) for i in N],
        agent_args=[str(i) for i in N],
    )
elif i==1:
    cluster.print_stdout('infras', None, 0)
    cluster.print_stdout('agent', 'yo2', 40)
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
    cluster.print_error()
