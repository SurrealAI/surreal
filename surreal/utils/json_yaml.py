"""
JSON and YAML utility functioins
"""
import json
import yaml
from io import StringIO
import os.path as path
from collections import OrderedDict
from functools import partial


def json_load(file_path, **kwargs):
    file_path = path.expanduser(file_path)
    with open(file_path, 'r') as fp:
        return json.load(fp, **kwargs)


def json_loads(string, **kwargs):
    return json.loads(string, **kwargs)


def json_dump(data, file_path, **kwargs):
    file_path = path.expanduser(file_path)
    with open(file_path, 'w') as fp:
        json.dump(data, fp, indent=4, **kwargs)


def json_dumps(data, **kwargs):
    "Returns: string"
    return json.dumps(data, **kwargs)


json_ordered_load = partial(json_load, object_pairs_hook=OrderedDict)
json_ordered_loads = partial(json_loads, object_pairs_hook=OrderedDict)
json_ordered_dump = json_dump
json_ordered_dumps = json_dumps


def yaml_load(file_path, *, loader=yaml.load, **kwargs):
    file_path = path.expanduser(file_path)
    with open(file_path, 'r') as fp:
        return loader(fp, **kwargs)


def yaml_loads(string, *, loader=yaml.load, **kwargs):
    return loader(string, **kwargs)


def yaml_dump(data, file_path, *, dumper=yaml.dump):
    file_path = path.expanduser(file_path)
    with open(file_path, 'w') as fp:
        dumper(
            data,
            stream=fp,
            indent=2,
            default_flow_style=False
        )


def yaml_dumps(data, *, dumper=yaml.dump):
    "Returns: string"
    stream = StringIO()
    dumper(
        data,
        stream,
        default_flow_style=False,
        indent=2
    )
    return stream.getvalue()


def yaml_ordered_load_stream(stream,
                             Loader=yaml.Loader,
                             object_pairs_hook=OrderedDict):
    """
    https://stackoverflow.com/questions/5121931/in-python-how-can-you-load-yaml-mappings-as-ordereddicts
    """
    class OrderedLoader(Loader):
        pass
    def _construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))
    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        _construct_mapping)
    return yaml.load(stream, OrderedLoader)


def yaml_ordered_dump_stream(data, stream=None, Dumper=yaml.Dumper, **kwargs):
    class OrderedDumper(Dumper):
        pass
    def _dict_representer(dumper, data):
        return dumper.represent_mapping(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
            data.items())
    OrderedDumper.add_representer(OrderedDict, _dict_representer)
    return yaml.dump(data, stream, OrderedDumper, **kwargs)


yaml_ordered_load = partial(yaml_load, loader=yaml_ordered_load_stream)
yaml_ordered_loads = partial(yaml_loads, loader=yaml_ordered_load_stream)
yaml_ordered_dump = partial(yaml_dump, dumper=yaml_ordered_dump_stream)
yaml_ordered_dumps = partial(yaml_dumps, dumper=yaml_ordered_dump_stream)


if __name__ == '__main__':
    D = OrderedDict(
        [('z','y'), ('x','w'), ('a', 'b'), ('c', 'd')]
    )
    print(yaml_ordered_dumps(D))
    print(yaml_loads(yaml_ordered_dumps(D)))
    print(yaml_ordered_loads(yaml_ordered_dumps(D)))
    fpath = '~/Temp/kurreal/ordered.yml'
    yaml_ordered_dump(D, fpath)
    print(yaml_load(fpath))
    print(yaml_ordered_load(fpath))

    print('===== json =====')
    print(json_ordered_dumps(D))
    print(json_loads(json_ordered_dumps(D)))
    print(json_ordered_loads(json_ordered_dumps(D)))
    fpath = '~/Temp/kurreal/ordered.json'
    json_ordered_dump(D, fpath)
    print(json_load(fpath))
    print(json_ordered_load(fpath))
