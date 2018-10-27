"""
File system utils.
"""
import os
import sys
import errno
import shutil
import glob
import pwd
import codecs
import hashlib
import tarfile
from socket import gethostname

f_ext = os.path.splitext

f_expand = os.path.expanduser

f_size = os.path.getsize

is_file = os.path.isfile

is_dir = os.path.isdir

get_dir = os.path.dirname


def owner_name(filepath):
    """
    Returns: file owner name, unix only
    """
    return pwd.getpwuid(os.stat(filepath).st_uid).pw_name


def host_name():
    "Get host name, alias with ``socket.gethostname()``"
    return gethostname()


def host_id():
    """
    Returns: first part of hostname up to '.'
    """
    return host_name().split('.')[0]


def utf_open(fname, mode):
    """
    Wrapper for codecs.open
    """
    return codecs.open(fname, mode=mode, encoding='utf-8')


def is_txt(fpath):
    "Test if file path is a text file"
    _, ext = f_ext(fpath)
    return ext == '.txt'


def f_exists(path):
    return os.path.exists(f_expand(path))


def f_join(*fpaths):
    """
    join file paths and expand special symbols like `~` for home dir
    """
    return f_expand(os.path.join(*fpaths))


def f_mkdir(fpath):
    """
    Recursively creates all the subdirs
    If exist, do nothing.
    """
    os.makedirs(f_expand(fpath), exist_ok=True)


def f_mkdir_in_path(fpath):
    """
    fpath is a file,
    recursively creates all the parent dirs that lead to the file
    If exist, do nothing.
    """
    os.makedirs(get_dir(f_expand(fpath)), exist_ok=True)


def f_last_part_in_path(fpath):
    """
    https://stackoverflow.com/questions/3925096/how-to-get-only-the-last-part-of-a-path-in-python
    """
    return os.path.basename(os.path.normpath(f_expand(fpath)))


def f_time(fpath):
    "File modification time"
    return str(os.path.getctime(fpath))


def f_append_before_ext(fpath, suffix):
    """
    Append a suffix to file name and retain its extension
    """
    name, ext = f_ext(fpath)
    return name + suffix + ext


def f_add_ext(fpath, ext):
    """
    Append an extension if not already there
    Args:
      ext: will add a preceding `.` if doesn't exist
    """
    if not ext.startswith('.'):
        ext = '.' + ext
    if fpath.endswith(ext):
        return fpath
    else:
        return fpath + ext


def f_remove(fpath):
    """
    If exist, remove. Supports both dir and file. Supports glob wildcard.
    """
    fpath = f_expand(fpath)
    for f in glob.glob(fpath):
        try:
            shutil.rmtree(f)
        except OSError as e:
            if e.errno == errno.ENOTDIR:
                try:
                    os.remove(f)
                except: # final resort safeguard
                    pass


def f_copy(fsrc, fdst):
    """
    If exist, remove. Supports both dir and file. Supports glob wildcard.
    """
    fsrc, fdst = f_expand(fsrc), f_expand(fdst)
    for f in glob.glob(fsrc):
        try:
            shutil.copytree(f, fdst)
        except OSError as e:
            if e.errno == errno.ENOTDIR:
                shutil.copy(f, fdst)


def f_move(fsrc, fdst):
    fsrc, fdst = f_expand(fsrc), f_expand(fdst)
    for f in glob.glob(fsrc):
        shutil.move(f, fdst)


def f_split_path(fpath, normpath=True):
    """
    Splits path into a list of its component folders

    Args:
        normpath: call os.path.normpath to remove redundant '/' and
            up-level references like ".."
    """
    if normpath:
        fpath = os.path.normpath(fpath)
    allparts = []
    while 1:
        parts = os.path.split(fpath)
        if parts[0] == fpath:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == fpath:  # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            fpath = parts[0]
            allparts.insert(0, parts[1])
    return allparts


def script_dir():
    """
    Returns: the dir of current script
    """
    return os.path.dirname(os.path.realpath(sys.argv[0]))


def parent_dir(location, abspath=False):
    """
    Args:
      location: current directory or file

    Returns:
        parent directory absolute or relative path
    """
    _path = os.path.abspath if abspath else os.path.relpath
    return _path(f_join(location, os.pardir))


def f_md5(fpath):
    """
    File md5 signature
    """
    hash_md5 = hashlib.md5()
    with open(f_expand(fpath), "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def compress_tar(source_file, output_tarball, compress_mode='gz'):
    """
    Args:
        source_file: source file or folder
        output_tarball: output tar file name
        compress_mode: "gz", "bz2", "xz" or "" (empty for uncompressed write)
    """
    source_file, output_tarball = f_expand(source_file), f_expand(output_tarball)
    assert compress_mode in ['gz', 'bz2', 'xz', '']
    with tarfile.open(output_tarball, 'w:'+compress_mode) as tar:
        tar.add(source_file, arcname=os.path.basename(source_file))


def extract_tar(source_tarball, output_dir='.', members=None):
    """
    Args:
        source_tarball: extract members from archive
        output_dir: default to current working dir
        members: must be a subset of the list returned by getmembers()
    """
    source_tarball, output_dir = f_expand(source_tarball), f_expand(output_dir)
    with tarfile.open(source_tarball, 'r:*')  as tar:
        tar.extractall(output_dir, members=members)


def move_with_backup(path, suffix='.bak'):
    """
    Ensures that a path is not occupied. If there is a file, rename it by
    adding @suffix. Resursively backs up everything.

    Args:
        path: file path to clear
        suffix: Add to backed up files (default: {'.bak'})
    """
    path = str(path)
    if os.path.exists(path):
        move_with_backup(path + suffix)
        shutil.move(path, path + suffix)
