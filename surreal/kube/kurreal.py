from surreal.kube.kubectl import *
from surreal.kube.generate_command import *
from surreal.kube.yaml_util import *
from surreal.utils.serializer import string_hash

# TODO experiment name needs to be passed into command line
# TODO temp yaml will be saved to surreal-dir/experiment/temp-aosdfjiosdj.yaml


experiment_name = 'myexper'
host_name = string_hash(experiment_name).lower()  # WARNING: must lower case
service_url = host_name + '.surreal'  # hostname.subdomain

if 1:
    kube = Kubectl(dry_run=0)
    gen = CommandGenerator('/mylibs/surreal/surreal/surreal/main/ddpg_configs.py', config_command="--env 'dm_control:cheetah-run' --savefile /root/", service_url=service_url)
    cmd_dict = gen.launch(3)
    print(cmd_dict['learner'])

    kube.create_surreal(
        './kurreal.yml',
        snapshot=0,
        NONAGENT_HOST_NAME=host_name,
        CMD_DICT=cmd_dict
    )
else:
    import pprint
    pp = pprint.pprint
    # 3 different ways to get a list of node names
    # pp(kube.query_jsonpath('nodes', '{.metadata.name}'))
    # pp(kube.query_jsonpath('nodes', "{.metadata.labels['kubernetes\.io/hostname']}"))
    # pp(kube.query_resources('nodes', 'name'))
    # y = kube.query_resources('nodes', 'yaml')
    # print(y.dumps_yaml())
    # pp(y.items[0].metadata)
    # print(YamlList(y).to_string())
    # print(kube.query_jsonpath('pods', '{.metadata.name}', labels='mytype=transient_component'))
    # print(kube.query_resources('pods', 'name', labels='mytype=persistent_component'))
    y = kube.query_resources('pods', 'yaml')
    for _it in y.items:
        print(_it.status.phase)
        # pp(y.items[0].status)
