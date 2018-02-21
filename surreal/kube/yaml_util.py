import yaml
import jinja2
import pprint
import uuid
import contextlib
import os
import os.path as path
from io import StringIO
from surreal.utils.ezdict import EzDict
from collections import OrderedDict


def file_content(fpath):
    with open(path.expanduser(fpath), 'r') as fp:
        return fp.read()


class Quoted(str):
    """
    https://stackoverflow.com/questions/8640959/how-can-i-control-what-scalar-form-pyyaml-uses-for-my-data
    """
    pass

class Literal(str):
    """
    Multi-line literals
    """
    pass

def _quoted_presenter(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='"')
yaml.add_representer(Quoted, _quoted_presenter)

def _literal_presenter(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
yaml.add_representer(Literal, _literal_presenter)

def _ordered_dict_presenter(dumper, data):
    return dumper.represent_dict(data.items())
yaml.add_representer(OrderedDict, _ordered_dict_presenter)


class YamlList(object):
    def __init__(self, data_list):
        """
        Args:
            data_list: a list of dictionaries
        """
        if isinstance(data_list, dict):
            data_list = [data_list]
        assert isinstance(data_list, list)
        for d in data_list:
            assert isinstance(d, dict)
        self.data_list = [EzDict(d) for d in data_list]

    def __getitem__(self, index):
        assert isinstance(index, int)
        return self.data_list[index]

    def save(self, fpath):
        """
        Args:
            fpath: yaml path
        """
        with open(path.expanduser(fpath), 'w') as fp:
            # must convert back to normal dict; yaml serializes EzDict object
            yaml.dump_all(
                [d._builtin_to_dict() for d in self.data_list],
                fp,
                default_flow_style=False,
            )

    def to_string(self):
        stream = StringIO()
        yaml.dump_all(
            [d._builtin_to_dict() for d in self.data_list],
            stream,
            default_flow_style=False,
            indent=2
        )
        return stream.getvalue()

    @contextlib.contextmanager
    def temp_file(self, folder='.'):
        """
        Returns:
            the temporarily generated path (uuid4)
        """
        temp_fname = 'temp-{}.yml'.format(uuid.uuid4())
        temp_fpath = path.expanduser(path.join(folder, temp_fname))
        self.save(temp_fpath)
        yield temp_fpath
        os.remove(temp_fpath)

    @classmethod
    def from_file(cls, fpath):
        with open(path.expanduser(fpath), 'r') as fp:
            return cls(list(yaml.load_all(fp)))

    @classmethod
    def from_string(cls, text):
        return cls(list(yaml.load_all(text)))

    @classmethod
    def from_template_string(cls,
                             text,
                             context=None,
                             **context_kwargs):
        """
        Render as Jinja template
        Args:
            text: template text
            context: a dict of variables to be rendered into Jinja2
            **context_kwargs: same as context
        """
        return cls.from_string(JinjaYaml(text).render(
            context=context,
            **context_kwargs
        ))

    @classmethod
    def from_template_file(cls,
                           template_path,
                           context=None,
                           **context_kwargs):
        """
        Render as Jinja template
        Args:
            template_path: yaml file with Jinja2 syntax
            context: a dict of variables to be rendered into Jinja2
            **context_kwargs: same as context
        """
        return cls.from_string(JinjaYaml.from_file(template_path).render(
            context=context,
            **context_kwargs
        ))

    def __str__(self):
        return self.to_string()


class JinjaYaml(object):
    """
    Jinja for rendering yaml
    """
    _env = None

    def __init__(self, text):
        self.text = text
        JinjaYaml._init_jinja_env()

    @staticmethod
    def _init_jinja_env():
        """
        Set custom filters:https://stackoverflow.com/a/47291097/3453033
        """
        if JinjaYaml._env is not None:
            return
        _env = jinja2.Environment(
            trim_blocks=True,
            lstrip_blocks=True
        )
        FILTERS = {
            'to_underscore': lambda s: s.replace('-', '_'),
            'to_hyphen': lambda s: s.replace('_', '-')
        }
        _env.filters.update(FILTERS)

        # https://stackoverflow.com/questions/21778252/how-to-raise-an-exception-in-a-jinja2-macro
        def _raise_jinja(msg):
            raise RuntimeError(msg)
        _env.globals['raise'] = _raise_jinja

        JinjaYaml._env = _env

    def render(self, context=None, **context_kwargs):
        """
        Args:
            template_path: yaml file with Jinja2 syntax
            context: a dict of variables to be rendered into Jinja2
            **context_kwargs: same as context

        Returns:
            rendered text
        """
        if context is not None:
            assert isinstance(context, dict)
            context_kwargs.update(context)
        for key, value in context_kwargs.items():
            if isinstance(value, str) and '\n' in value:
                # correctly render multiline in Yaml
                # remove the first and last single quote, change them to literal double quotes
                context_kwargs[key] = '"{}"'.format(repr(value)[1:-1])
        template = JinjaYaml._env.from_string(self.text)
        return template.render(context_kwargs)

    def render_file(self, out_file, context=None, **context_kwargs):
        """
        Render as Jinja template
        Args:
            template_path: yaml file with Jinja2 syntax
            context: a dict of variables to be rendered into Jinja2
            **context_kwargs: same as context
        """
        with open(path.expanduser(out_file), 'w') as fp:
            fp.write(self.render(context=context, **context_kwargs))

    def _get_temp_filepath(self, folder):
        temp_fname = 'jinja-{}.yml'.format(uuid.uuid4())
        return path.expanduser(path.join(folder, temp_fname))

    def render_temp_file(self, folder='.', context=None, **context_kwargs):
        temp_fpath = self._get_temp_filepath(folder)
        self.render_file(temp_fpath, context=context, **context_kwargs)
        yield temp_fpath
        os.remove(temp_fpath)

    @contextlib.contextmanager
    def render_throwaway_file(self, folder='.', context=None, **context_kwargs):
        """
        Returns:
            the temporarily generated yaml (uuid4)
        """
        temp_fpath = self._get_temp_filepath(folder)
        self.render_file(temp_fpath, context=context, **context_kwargs)
        yield temp_fpath
        os.remove(temp_fpath)

    @classmethod
    def from_file(cls, template_path):
        return cls(file_content(template_path))


if __name__ == '__main__':
    y = YamlList.from_template_string('shit: {{ shit  }} \n'
                                      'myseq: "{% for n in range(10) %} my{{ n }} {% endfor %}"\n'
                                      'mylist: {{lis|join("$")}}', shit='yoyo', n=7, lis=[100,99,98])

    with y.temp_file() as fname:
        os.system('cat ' +fname)
    print(y)
    print(y[0].mylist)
    print('='*40)

    y = JinjaYaml('shit: {{ shit  }} \n'
                                      'myseq: "{% for n in range(10) %} my{{ n }} {% endfor %}"\n'
                                      'mylist: {{lis|join("$")}}')
    print(y.render(shit='yoyo\ndamn\n\nlol', n=7, lis=[100,99,98]))
    y = JinjaYaml('shit: 3')
    print(y.render())
