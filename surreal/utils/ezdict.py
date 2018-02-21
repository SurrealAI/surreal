"""
Adapted from: https://github.com/makinacorpus/EzDict
This version adds new features to EzDict and fixes issues, like how to handle
builtin methods (items, update) when there's conflict
"""
import json
import yaml
from io import StringIO
from os.path import expanduser


def _get_special_methods():
    methods = ['keys', 'items', 'values', 'get', 'clear',
               'update', 'pop', 'popitem',
               'to_dict', 'deepcopy']
    for action in ['load', 'dump']:
        for mode in ['s', '']:  # 's' for string, '' for file
            for format in ['json', 'yaml']:
                methods.append(action + mode + '_' + format)
    protected = ['_builtin_' + m for m in methods]
    return methods + protected, protected


_EzDict_NATIVE_METHODS, _EzDict_PROTECTED_METHODS = _get_special_methods()


class EzDict(dict):
    """
    Notes:
      Use `dict.items()` if you know there might be conflict in the keys
      or `_builtin_` + method name

    Added methods: the version always prefixed by `builtin` is protected against
      changes. You can use the non-prefixed version if you know for sure that
      the name will never be overriden

    >>> d = EzDict({'foo':3})
    >>> d['foo']
    3
    >>> d.foo
    3
    >>> d.bar
    Traceback (most recent call last):
    ...
    AttributeError: 'EzDict' object has no attribute 'bar'

    Works recursively

    >>> d = EzDict({'foo':3, 'bar':{'x':1, 'y':2}})
    >>> isinstance(d.bar, dict)
    True
    >>> d.bar.x
    1

    Bullet-proof

    >>> EzDict({})
    {}
    >>> EzDict(d={})
    {}
    >>> EzDict(None)
    {}
    >>> d = {'a': 1}
    >>> EzDict(**d)
    {'a': 1}

    Set attributes

    >>> d = EzDict()
    >>> d.foo = 3
    >>> d.foo
    3
    >>> d.bar = {'prop': 'value'}
    >>> d.bar.prop
    'value'
    >>> d
    {'foo': 3, 'bar': {'prop': 'value'}}
    >>> d.bar.prop = 'newer'
    >>> d.bar.prop
    'newer'


    Values extraction

    >>> d = EzDict({'foo':0, 'bar':[{'x':1, 'y':2}, {'x':3, 'y':4}]})
    >>> isinstance(d.bar, list)
    True
    >>> from operator import attrgetter
    >>> map(attrgetter('x'), d.bar)
    [1, 3]
    >>> map(attrgetter('y'), d.bar)
    [2, 4]
    >>> d = EzDict()
    >>> d.keys()
    []
    >>> d = EzDict(foo=3, bar=dict(x=1, y=2))
    >>> d.foo
    3
    >>> d.bar.x
    1

    Still like a dict though

    >>> o = EzDict({'clean':True})
    >>> o.items()
    [('clean', True)]

    And like a class

    >>> class Flower(EzDict):
    ...     power = 1
    ...
    >>> f = Flower()
    >>> f.power
    1
    >>> f = Flower({'height': 12})
    >>> f.height
    12
    >>> f['power']
    1
    >>> sorted(f.keys())
    ['height', 'power']
    """
    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        if kwargs:
            dict.update(d, **kwargs)
        for k, v in dict.items(d):
            setattr(self, k, v)
        # Class attributes
        for k in self.__class__.__dict__.keys():
            if not (k.startswith('__') and k.endswith('__')
                    or k in _EzDict_NATIVE_METHODS):
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if name in _EzDict_PROTECTED_METHODS:
            raise ValueError('Cannot override `{}`: EzDict protected method'
                             .format(name))
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x)
                     if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict):
            # implements deepcopy if EzDict(EzDict())
            # to make it shallow copy, add the following condition:
            # ...  and not isinstance(value, self.__class__)):
            value = self.__class__(value)
        super(EzDict, self).__setattr__(name, value)
        super(EzDict, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def to_dict(self):
        """
        Convert to raw dict
        """
        return ezdict_to_dict(self)

    def deepcopy(self):
        return EzDict(self)

    @classmethod
    def load_json(cls, file_path):
        file_path = expanduser(file_path)
        with open(file_path, 'r') as fp:
            return cls(json.load(fp))

    @classmethod
    def loads_json(cls, string):
        return cls(json.loads(string))

    @classmethod
    def load_yaml(cls, file_path):
        file_path = expanduser(file_path)
        with open(file_path, 'r') as fp:
            return cls(yaml.load(fp))

    @classmethod
    def loads_yaml(cls, string):
        return cls(yaml.load(string))

    def dump_json(self, file_path):
        file_path = expanduser(file_path)
        with open(file_path, 'w') as fp:
            json.dump(ezdict_to_dict(self), fp, indent=4)

    def dumps_json(self):
        "Returns: string"
        return json.dumps(ezdict_to_dict(self))

    def dump_yaml(self, file_path):
        file_path = expanduser(file_path)
        with open(file_path, 'w') as fp:
            yaml.dump(
                ezdict_to_dict(self),
                stream=fp,
                indent=2,
                default_flow_style=False
            )

    def dumps_yaml(self):
        "Returns: string"
        stream = StringIO()
        yaml.dump(
            ezdict_to_dict(self),
            stream,
            default_flow_style=False,
            indent=2
        )
        return stream.getvalue()

    def __getstate__(self):
        """
        Support pickling.
        Warning:
          if this EzDict overrides dict builtin methods, like `items`,
          pickle will report error.
          don't know how to resolve yet
        """
        return self._builtin_to_dict()

    def __setstate__(self, state):
        self.__init__(state)

    def __str__(self):
        return str(ezdict_to_dict(self))

    _builtin_keys = dict.keys
    _builtin_items = dict.items
    _builtin_values = dict.values
    _builtin_get = dict.get
    _builtin_clear = dict.clear
    _builtin_update = dict.update
    _builtin_pop = dict.pop
    _builtin_popitem = dict.popitem
    _builtin_to_dict = to_dict
    _builtin_deepcopy = deepcopy
    _builtin_loads_json = loads_json
    _builtin_loads_yaml = loads_yaml
    _builtin_load_json = load_json
    _builtin_load_yaml = load_yaml
    _builtin_dumps_json = dumps_json
    _builtin_dumps_yaml = dumps_yaml
    _builtin_dump_json = dump_json
    _builtin_dump_yaml = dump_yaml


def ezdict_to_dict(easy_dict):
    """
    Recursively convert back to builtin dict type
    """
    d = {}
    for k, value in dict.items(easy_dict):
        if isinstance(value, EzDict):
            d[k] = ezdict_to_dict(value)
        elif isinstance(value, (list, tuple)):
            d[k] = type(value)(
                ezdict_to_dict(v)
                if isinstance(v, EzDict)
                else v for v in value
            )
        else:
            d[k] = value
    return d


def _add_protected_methods():
    for protected, normal in zip(_EzDict_PROTECTED_METHODS,
                                 _EzDict_NATIVE_METHODS):
        setattr(EzDict, protected, getattr(EzDict, normal))

_add_protected_methods()


def _print_protected_methods():
    "paste the generated code into EzDict class for PyCharm convenience"
    for i, (protected, normal) in enumerate(zip(_EzDict_PROTECTED_METHODS,
                                                _EzDict_NATIVE_METHODS)):
        if i < 8:
            normal = 'dict.' + normal
        print('{} = {}'.format(protected, normal))


if __name__ == '__main__':
    import pickle, traceback
    _print_protected_methods()
    if 1:
        a = EzDict({'keys': EzDict({'items': 100, 'get': 66})})
        b = a.deepcopy()
        b.keys.items = 120
        print(a.keys._builtin_items())
        print(b.keys)
        print(b.keys.get)
        # aib = pickle.dumps(b)
        # aib = pickle.loads(aib)
        # print(aib)
        # print(aib.keys)
    else:
        a = EzDict({'keys2': {'items2': 100, 'get2': 66, 'values':10}})
        b = a.deepcopy()
        b.keys2.items2 = 120
        aib = pickle.dumps(b)
        aib = pickle.loads(aib)
        print(aib)
        print(aib.keys2.get2)

