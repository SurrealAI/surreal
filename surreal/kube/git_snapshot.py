from git import Repo
import sys
import time
import uuid
import signal
import surreal.utils as U


def print_err(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr)


def push_snapshot(snapshot_branch, repo_path='.', verbose=True):
    """
    Save a snapshot of the current codebase (with uncommitted changes and untracked
    files) to a temporary branch and then push to github.
    Remote cloud will then pull from the temp branch and run the latest dirty code.

    https://stackoverflow.com/questions/5717026/how-to-git-cherry-pick-only-changes-to-certain-files
    https://stackoverflow.com/questions/48511079/git-commands-to-save-current-files-in-temporary-branch-without-committing-to-mas

    Args:
        snapshot_branch:
        repo_path:
        verbose:
    """
    original_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(
        signal.SIGINT,
        lambda *args: print("Please don't Ctrl-C in the middle of git snapshot")
    )
    _push_snapshot(
        snapshot_branch=snapshot_branch,
        repo_path=repo_path,
        verbose=verbose
    )
    signal.signal(signal.SIGINT, original_sigint)


def _push_snapshot(snapshot_branch, repo_path='.', verbose=True,
                  *, __push_temp_file=None):
    """
    Save a snapshot of the current codebase (with uncommitted changes and untracked
    files) to a temporary branch and then push to github.
    Remote cloud will then pull from the temp branch and run the latest dirty code.

    https://stackoverflow.com/questions/5717026/how-to-git-cherry-pick-only-changes-to-certain-files
    https://stackoverflow.com/questions/48511079/git-commands-to-save-current-files-in-temporary-branch-without-committing-to-mas

    Args:
        snapshot_branch:
        repo_path:
        verbose:
        __push_temp_file: workaround cherry-pick + merge commit, do not touch
    """
    repo = Repo(repo_path)
    git = repo.git
    work_branch = repo.active_branch.name

    ret = git.stash()
    is_stash = not 'No local changes' in ret
    # delete the temp branch if already exist
    try:
        git.branch('-D', snapshot_branch)
    except:  # the branch doesn't exist, fine.
        pass
    git.checkout('-b', snapshot_branch)
    if is_stash:
        git.stash('apply')
    git.add('.')
    try:
        msg = git.commit('-m', 'snapshot save ' + time.strftime('%m/%d/%Y %H:%M:%S'))
        if verbose: print(msg)
    except:
        if verbose:
            print('no temporary changes to push')
    try:
        git.push('-f', 'origin', snapshot_branch)
        if verbose:
            remote_url = list(list(repo.remotes)[0].urls)[0]
            print('successfully pushed to branch {} of {}'.format(
                snapshot_branch, remote_url))
    except Exception as e:
        print_err('push to remote failed', e)
    git.checkout(work_branch)
    try:
        git.cherry_pick('-n', snapshot_branch)
    except Exception as e:
        if __push_temp_file:
            raise  # temp_file doesn't help cherry-pick + merge
        if 'merge' in str(e):
            # known problem: cherry-pick cannot pick a merge commit
            # hacky solution: create an empty temporary file, and then remove it
            temp_file = 'git-snapshot-temp-' + str(uuid.uuid4())
            temp_file = U.f_join(repo_path, temp_file)
            print_err('git cherry-pick cannot handle merge commit. '
                      'Workaround: creating a temporary file', temp_file)
            with open(temp_file, 'w') as f:
                f.write('git cherry-pick + merge workaround')
            push_snapshot(
                snapshot_branch=snapshot_branch,
                repo_path=repo_path,
                verbose=verbose,
                __push_temp_file=temp_file
            )
        else:
            raise
    msg = git.reset('HEAD')
    if verbose: print(msg)
    if __push_temp_file:
        U.f_remove(__push_temp_file)
        if verbose: print('removed', __push_temp_file)


def main():
    import argparse
    parser = argparse.ArgumentParser('Git snapshot tool')
    parser.add_argument('snapshot_branch', help='name of the snapshot branch')
    parser.add_argument('repo_path', help='location of the git repository')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    push_snapshot(**vars(args))


if __name__ == '__main__':
    main()

