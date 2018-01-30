import yaml
import jinja2
import pprint
import uuid
import contextlib
import os
import os.path as path
from easydict import EasyDict


class YamlList(object):
    def __init__(self, data_list):
        """
        Args:
            data_list: a list of dictionaries
        """
        assert isinstance(data_list, list)
        for d in data_list:
            assert isinstance(d, dict)
        self.data_list = [EasyDict(d) for d in data_list]

    def __getitem__(self, index):
        assert isinstance(index, int)
        return self.data_list[index]

    def save(self, fpath, pretty=True):
        """

        Args:
            fpath:
            pretty: default_flow_style=True for yaml to be human-friendly
        """
        with open(fpath, 'w') as fp:
            # must convert back to normal dict; yaml serializes EasyDict object
            yaml.dump_all(
                [dict(d) for d in self.data_list],
                fp,
                default_flow_style=not pretty
            )

    @contextlib.contextmanager
    def temp_file(self, folder='.'):
        """
        Returns:
            the temporarily generated path (uuid4)
        """
        temp_fname = 'kube-{}.yml'.format(uuid.uuid4())
        temp_fpath = path.join(path.expanduser(folder), temp_fname)
        self.save(temp_fpath, pretty=False)
        yield temp_fpath
        os.remove(temp_fpath)

    @classmethod
    def from_file(cls, fpath):
        with open(fpath, 'r') as fp:
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
        if context is not None:
            assert isinstance(context, dict)
            context_kwargs.update(context)
        text = jinja2.Template(text).render(context_kwargs)
        return cls.from_string(text)

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
        if context is not None:
            assert isinstance(context, dict)
            context_kwargs.update(context)
        with open(template_path, 'r') as fp:
            text = fp.read()
        return cls.from_template_string(text, context=context, **context_kwargs)

    def __str__(self):
        return pprint.pformat(self.data_list, indent=2)


if __name__ == '__main__':
    y = YamlList.from_template_string('shit: {{ shit  }} \n'
                                      'myseq: "{% for n in range(10) %} my{{ n }} {% endfor %}"\n'
                                      'mylist: {{lis|join("$")}}', shit='yoyo', n=7, lis=[100,99,98])

    with y.temp_file() as fname:
        os.system('cat ' +fname)
    y.save('yo.yml')
    os.system('cat yo.yml')

    print(y[0].mylist)
