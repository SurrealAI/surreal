"""
Create .tar.gz of code repos inside a pod and copy to experiment folder
Use os.environ['repo_<reponame>'] defined in kurreal_template.yml
"""
import os
import surreal.utils as U
from tensorplex import Logger


_logger = Logger.get_logger(
    'loggerplex',
    stream='stdout',
    show_level=True
)


def tar_kurreal_repo(experiment_folder):
    """
    Repo paths are saved in env vars "repo_<reponame>"
    """
    folder = U.f_join(experiment_folder, 'code')
    U.f_mkdir(folder)
    repos = {}
    for name, value in os.environ.items():
        if name.startswith('repo_'):
            repos[name[len('repo_'):]] = value

    for repo_name, repo_source in repos.items():
        repo_tar = U.f_join(folder, repo_name + '.tar.gz')
        U.compress_tar(repo_source, repo_tar)
        _logger.info('compressed {} to {}'.format(repo_source, repo_tar))

