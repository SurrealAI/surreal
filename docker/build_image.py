# This is used by Surreal developers to build public surreal images

from symphony.addons import DockerBuilder

force_update = True
settings = {
  'temp_directory': '~/symph_temp/',
  'dockerfile': '~/surreal/Surreal/docker/Dockerfile-nvidia',
  'context_directories': [
    {
      'name': 'surreal',
      'path': '~/surreal/Surreal',
      'force_update': force_update,
    },
    # {
    #   'name': 'symphony',
    #   'path': '~/surreal/symphony',
    #   'force_update': force_update,
    # },
    {
      'name': 'build_files',
      'path': '~/surreal/Surreal/docker/build_files',
      'force_update': force_update,
    },
  ]
}

builder = DockerBuilder.from_dict(settings)
builder.build()
repo = "surrealai/surreal-nvidia"
tag = "v0.1"
builder.tag(repo, tag)
builder.push(repo, tag)

# builder.tag('us.gcr.io/jimfan2018-208323/surreal-prepublic-nvidia', 'latest')
# builder.push('us.gcr.io/jimfan2018-208323/surreal-prepublic-nvidia', 'latest')
