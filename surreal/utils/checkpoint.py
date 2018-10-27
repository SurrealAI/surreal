"""
Checkpoint that supports labeling and performance tracking
"""
import torch
import pickle
import time
import datetime
from benedict import BeneDict
from pkg_resources import parse_version
from collections import OrderedDict
from contextlib import contextmanager
from . import filesys as U


CHEKCPOINT_VERSION = '0.0.1'


class Checkpoint(object):
    """
    Periodically saves attribute of a learner or agent class.
    For PyTorch modules, must extend surreal.utils.pytorch.Module instead of
    the builtin pytorch class.
    One experiment can have multiple Checkpoints with different names. E.g.:
    suppose each agent has different exploration rate annealing. Each agent will
    save to <agent-N>.ckpt and metadata-<agent-N>.yml
    and the central learner will save to learner.ckpt and metadata-learner.yml

    metadata.yml has the following fields:
    - save_counter: how many times "save()" has been called, starts from 1
    - history_ckpt_files: ckpt file names from old to new
    - global_steps: current global_steps counter
    - ckpt: dict of ckpt file names -> info
    """
    # TODO support register custom object serializer method
    def __init__(self, folder,
                 name,
                 *,
                 tracked_obj,
                 tracked_attrs=None,
                 keep_history=1,
                 keep_best=1,
                 mkdir=True):
        """
        Args:
            folder: checkpoint folder path
            name: prefix of the checkpoint
            tracked_obj: arbitrary object whose certain attributes will be tracked.
                typically Learner or Agent class
            tracked_attrs: tracked attribute names of the object, can include
                attrs that point to torch.nn.Module
                set to None if you will restore from existing checkpoint
            keep_history: how many checkpoints to keep, must >= 1
            keep_best: how many best models to keep, sorted by score, must >= 0
            mkdir: True to create intermediate dirs if the path doesn't exist
        """
        self.folder = U.f_expand(folder)
        if mkdir:
            U.f_mkdir(folder)
        self.name = name
        self.tracked_obj = tracked_obj

        if U.f_exists(self.metadata_path()):
            self._load_metadata()
        else:
            # blank-slate checkpoint
            metadata = BeneDict()
            metadata.version = CHEKCPOINT_VERSION
            metadata.save_counter = 0
            metadata.history_ckpt_files = []
            metadata.ckpt = {}
            self._check_tracked_attrs(tracked_attrs)
            metadata.tracked_attrs = tracked_attrs
            assert keep_history >= 1
            metadata.keep_history = keep_history
            assert keep_best >= 0
            metadata.keep_best = keep_best
            metadata.best_ckpt_files = []
            metadata.best_scores = []
            self.metadata = metadata

    def _check_version(self):
        assert 'version' in self.metadata, \
            'version not found in ' + self.metadata_path()
        if parse_version(CHEKCPOINT_VERSION) != parse_version(self.metadata.version):
            raise ValueError('checkpoint version incompatible, please examine '
                             '{} and make sure it is {}'
                             .format(self.metadata_path(), CHEKCPOINT_VERSION))

    def _check_tracked_attrs(self, tracked_attrs):
        ERR_MSG = 'tracked_attrs must be a list of attribute name strings or None'
        if isinstance(tracked_attrs, (list, tuple)):
            for attr_name in tracked_attrs:
                assert isinstance(attr_name, str), ERR_MSG
        else:
            assert tracked_attrs is None, ERR_MSG

    def _load_metadata(self):
        if U.f_exists(self.metadata_path()):
            self.metadata = BeneDict.load_yaml_file(self.metadata_path())
        self._check_version()

    def _restore(self, ckpt_file, check_ckpt_exists):
        """
        helper to be used in self.restore() and self.restore_full_name()
        Returns:
            - ckpt file name if successfully restored
            - None otherwise
        """
        ckpt_path = self._get_path(ckpt_file)
        if not U.f_exists(ckpt_path):
            if check_ckpt_exists:
                raise FileNotFoundError(ckpt_path + ' missing.')
            else:
                return None
        # overwrite attributes on self.tracked_obj
        with open(ckpt_path, 'rb') as fp:
            data = pickle.load(fp)
        for attr_name in self.metadata.tracked_attrs:
            attr_value = getattr(self.tracked_obj, attr_name)
            if isinstance(attr_value, (torch.nn.Module, torch.optim.Optimizer)):
                attr_value.load_state_dict(data[attr_name])
            else:
                setattr(self.tracked_obj, attr_name, data[attr_name])
        return ckpt_path

    @contextmanager
    def _change_folder(self, new_folder):
        """
        Temporary change save folder, useful for loading from other folders
        """
        if new_folder:
            old_folder = self.folder
            new_folder = U.f_expand(new_folder)
            assert U.f_exists(new_folder)
            self.folder = new_folder
            yield
            self.folder = old_folder
        else:
            yield  # noop context

    def restore(self, target,
                mode,
                reload_metadata=True,
                check_ckpt_exists=False,
                restore_folder=None):
        """
        Args:
            target: can be one of the following semantics
              - int: 0 for the last (or best), 1 for the second last (or best), etc.
              - global steps of the ckpt file, the suffix string right before ".ckpt"
            mode: "best" or "history", which group to restore
            reload_metadata: overwrite self.metadata with the metadata.yml file content
            check_ckpt_exists: raise FileNotFoundError if the checkpoint target doesn't exist
            restore_folder: if None, use self.folder, else specify a folder to restore from
                if not None, reload_metadata will be forced to True

        Returns:
            - full ckpt file path if successfully restored
            - None otherwise

        Warnings:
            to restore attrs that point to pytorch Module, you must initialize
            the Module class first, because the checkpoint only saves the
            state_dict, not how to construct your Module
        """
        assert mode in ['best', 'history']
        with self._change_folder(restore_folder):
            if reload_metadata or restore_folder:
                self._load_metadata()
            self._check_version()
            meta = self.metadata
            if isinstance(target, int):
                assert target >= 0, \
                    'target int should start from 0 for the last or best'
                try:
                    if mode == 'best':
                        ckpt_file = meta.best_ckpt_files[target]
                    else:
                        ckpt_file = meta.history_ckpt_files[target]
                except IndexError:
                    if check_ckpt_exists:
                        raise FileNotFoundError(
                            '{} [{}] ckpt file missing'
                            .format(mode.capitalize(), target))
                    else:
                        ckpt_file = '__DOES_NOT_EXIST__'
            else:
                assert '.ckpt' not in target, 'use restore_full_path() instead'
                if mode == 'best':
                    ckpt_file = self.ckpt_name('best-{}'.format(target))
                else:
                    ckpt_file = self.ckpt_name(target)
            return self._restore(ckpt_file, check_ckpt_exists)

    def restore_full_name(self,
                          ckpt_file,
                          check_ckpt_exists=True,
                          restore_folder=None):
        """
        Args:
            ckpt_file: full name of the ckpt_file in self.folder.
            check_ckpt_exists: raise FileNotFoundError if the checkpoint target doesn't exist

        Returns:
            - full ckpt file path if successfully restored
            - None otherwise
        """
        with self._change_folder(restore_folder):
            self._load_metadata()
            return self._restore(ckpt_file, check_ckpt_exists)

    def _get_path(self, fname):
        return U.f_join(self.folder, fname)

    def metadata_name(self):
        return 'metadata.{}.yml'.format(self.name)

    def metadata_path(self):
        return self._get_path(self.metadata_name())

    def ckpt_name(self, suffix):
        return '{}.{}.ckpt'.format(self.name, suffix)

    def _copy_last_to_best(self, suffix):
        U.f_copy(self.ckpt_path(suffix),
                 self.ckpt_path('best-{}'.format(suffix)))

    def ckpt_path(self, suffix):
        return self._get_path(self.ckpt_name(suffix))

    def _save_metadata(self):
        self.metadata.dump_yaml_file(self.metadata_path())

    def _save_ckpt(self, suffix):
        data = OrderedDict()
        assert self.metadata.tracked_attrs is not None, \
            'tracked_attrs must not be None for save(). ' \
            'Did you forget to restore from an existing checkpoint?'
        for attr_name in self.metadata.tracked_attrs:
            attr_value = getattr(self.tracked_obj, attr_name)
            if isinstance(attr_value, (torch.nn.Module, torch.optim.Optimizer)):
                data[attr_name] = attr_value.state_dict()
            else:
                data[attr_name] = attr_value
        with open(self.ckpt_path(suffix), 'wb') as fp:
            pickle.dump(data, fp)

    def save(self, score=None,
             global_steps=None,
             reload_metadata=False,
             **ckpt_info):
        """
        save to `<name>.<global_steps>.ckpt` file

        Args:
            score: float scalar to be compared, for `keep_best`
            global_steps: if None, will use internal save counter
            reload_metadata: True to overwrite self.metadata with disk version
            ckpt_info: additional metadata info for this ckpt
        """
        if reload_metadata:
            self._load_metadata()
        meta = self.metadata
        meta.save_counter += 1
        if global_steps is None:
            global_steps = meta.save_counter
        # save the last step
        suffix = global_steps
        self._save_ckpt(suffix)

        meta.global_steps = global_steps
        # history_ckpt_files count from the newest to the oldest
        meta.history_ckpt_files = [self.ckpt_name(suffix)] + meta.history_ckpt_files
        # delete older history ckpt files.
        ckpt_to_remove = meta.history_ckpt_files[meta.keep_history:]
        for ckpt_name in ckpt_to_remove:
            U.f_remove(self._get_path(ckpt_name))
        del meta.history_ckpt_files[meta.keep_history:]
        # add a ckpt entry to metadata
        ckpt_metadata_entry = {
            'score': score,
            'global_steps': global_steps,
            'save_counter': meta.save_counter,
            'time': time.time(),
            'datetime': str(datetime.datetime.now()),
        }
        ckpt_metadata_entry.update(ckpt_info)
        meta.ckpt[self.ckpt_name(suffix)] = ckpt_metadata_entry
        # save the best-performing weights
        if meta.keep_best > 0:
            assert score is not None, \
                'score cannot be None if keep_best is enabled'
            score_queue = _ScoreQueue(meta.keep_best)
            to_deletes = score_queue.set_queue(
                meta.best_scores, meta.best_ckpt_files)
            best_ckpt_name = self.ckpt_name('best-{}'.format(suffix))
            evict = score_queue.add(score, best_ckpt_name)
            if evict is None or evict[1] != best_ckpt_name:
                # the latest ckpt is not evicted from best queue
                self._copy_last_to_best(suffix)
                meta.ckpt[best_ckpt_name] = ckpt_metadata_entry
            if evict:
                to_deletes.append(evict)
            for _, ckpt_to_delete in to_deletes:
                ckpt_path_to_delete = self._get_path(ckpt_to_delete)
                if U.f_exists(ckpt_path_to_delete):
                    U.f_remove(ckpt_path_to_delete)
                if ckpt_to_delete in meta.ckpt:
                    del meta.ckpt[ckpt_to_delete]
            # print(score_queue.get_scores_filepaths(), best_ckpt_name)
            meta.best_scores, meta.best_ckpt_files = \
                score_queue.get_scores_filepaths()

        self._save_metadata()


class PeriodicCheckpoint(Checkpoint):
    def __init__(self, *args, period, min_interval=0, **kwargs):
        """
        Same signature as Checkpoint
        Args:
            period: Only update when the interval counter % period == 0
            min_interval: Only update when the last update
                          happend min_interval minutes ago
        """
        super().__init__(*args, **kwargs)
        assert period >= 1
        self.period = period
        self._period_counter = 0
        self.min_interval = min_interval
        self.last_update_time = time.time()

    def save(self, *args, **kwargs):
        """
        Same as Checkpoint.save() except that it only runs every
        `period` number of calls.

        Returns:
            True if actually saved
        """
        self._period_counter += 1
        if self._period_counter % self.period == 0:
            if time.time() - self.last_update_time >= self.min_interval:
                super().save(*args, **kwargs)
                self.last_update_time = time.time()
                return True
        return False

    def reset_period(self):
        """
        Set period counter to 0
        """
        self._period_counter = 0


class _ScoreQueue(object):
    """
    Reverse score sorting
    """
    def __init__(self, max_size):
        self._queue = []  # large -> small sorted
        self.max_size = max_size

    def set_queue(self, scores, filepaths):
        "queue of (score, filepath) tuples"
        self._queue = list(zip(scores, filepaths))
        to_delete = self._queue[self.max_size:]
        del self._queue[self.max_size:]
        return to_delete

    def add(self, score, filepath):
        """
        qu = _ScoreQueue(3)
        for i in [1,3,2,4,5,7,3, 8, 2, 2, 1]:
            print(qu.add(i*10, 'path'+str(i)), qu._queue)
        """
        i = len(self._queue)
        while i >= 1:
            sc, _ = self._queue[i - 1]
            if sc > score:
                break
            i -= 1
        self._queue = self._queue[:i] + [(score, filepath)] + self._queue[i:]
        if len(self._queue) > self.max_size:
            to_delete = self._queue[-1]
            del self._queue[-1]
            return to_delete

    def get_scores_filepaths(self):
        "returns score_list, filepath_list"
        return tuple(zip(*self._queue))
