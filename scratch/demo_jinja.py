import jinja2


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


text = """
x-{{ MY_VAR }}-x
"""

context = {
    'MY_VAR': [3,2,1]
}

template = _env.from_string(text)
print(template.render(context))
