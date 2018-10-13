from symphony.addons import DockerBuilder


settings = {
  'temp_directory': '~/symph_temp/',
  'dockerfile': '~/surreal/Surreal/docker/Dockerfile-nvidia',
  'context_directories': [
    {
      'name': 'surreal',
      'path': '~/surreal/Surreal',
      'force_update': True,
    },
    {
      'name': 'mujoco',
      'path': '~/surreal/RoboticsSuite',
      'force_update': True,
    },
    {
      'name': 'caraml',
      'path': '~/surreal/caraml',
      'force_update': True,
    },
    {
      'name': 'symphony',
      'path': '~/surreal/symphony',
      'force_update': True,
    },
    {
      'name': 'build_files',
      'path': '~/surreal/Surreal/docker/build_files',
      'force_update': True,
    },
    # mjkey.txt
    {
      'name': 'mjkey.txt',
      'path': '~/.mujoco/mjkey.txt',
      'force_update': True,
    }
  ]
}

builder = DockerBuilder.from_dict(settings)
builder.build()
builder.tag('us.gcr.io/jimfan2018-208323/surreal-prepublic-nvidia', 'latest')
