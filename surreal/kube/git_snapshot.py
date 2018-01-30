from git import Repo
import time


def push_snapshot(snapshot_branch, repo_path='.'):
    """
    Save a snapshot of the current codebase (with uncommitted changes and untracked
    files) to a temporary branch and then push to github.
    Remote cloud will then pull from the temp branch and run the latest dirty code.

    https://stackoverflow.com/questions/5717026/how-to-git-cherry-pick-only-changes-to-certain-files
    https://stackoverflow.com/questions/48511079/git-commands-to-save-current-files-in-temporary-branch-without-committing-to-mas
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
        git.commit('-m', 'snapshot save ' + time.strftime('%m/%d/%Y %H:%M:%S'))
    except:
        print('no temporary changes to push')
    try:
        git.push('-f', 'origin', snapshot_branch)
    except Exception as e:
        print('push to remote failed', e)
    git.checkout(work_branch)
    git.cherry_pick('-n', snapshot_branch)
    print(git.reset('HEAD'))


def main():
    import argparse
    parser = argparse.ArgumentParser('Git snapshot tool')
    parser.add_argument('snapshot_branch', help='name of the snapshot branch')
    parser.add_argument('repo_path', help='location of the git repository')
    args = parser.parse_args()
    push_snapshot(**vars(args))


if __name__ == '__main__':
    main()

