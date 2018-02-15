"""
Checkpoint that supports labeling and performance tracking
"""
import torch
import pickle
import time
import datetime
import surreal.utils as U
from collections import OrderedDict
from surreal.utils.ezdict import EzDict


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
    def __init__(self, folder,
                 name,
                 *,
                 tracked_obj,
                 tracked_attrs=None,
                 keep_history=1,
                 keep_best=True,
                 initial_score=-float('inf')
                 ):
        """
        Args:
            folder: checkpoint folder path
            name: prefix of the checkpoint
            tracked_obj: arbitrary object whose certain attributes will be tracked.
                typically Learner or Agent class
            tracked_attrs: tracked attribute names of the object, can include
                attrs that point to torch.nn.Module
                set to None if you will restore from existing checkpoint
            keep_best: whether to save the best performing parameters
            initial_score: should be specified if keep_best=True
            keep_history: how many checkpoints to keep, must >= 1
        """
        self.folder = U.f_expand(folder)
        self.name = name
        self.tracked_obj = tracked_obj
        self._period_counter = 1

        if U.f_exists(self.metadata_path()):
            self._load_metadata()
        else:
            # blank-slate checkpoint
            metadata = EzDict()
            metadata.save_counter = 1
            metadata.history_ckpt_files = []
            metadata.ckpt = {}
            self._check_tracked_attrs(tracked_attrs)
            metadata.tracked_attrs = tracked_attrs
            metadata.best_score = initial_score
            assert keep_history >= 1
            metadata.keep_history = keep_history
            metadata.keep_best = keep_best
            self.metadata = metadata

    def _check_tracked_attrs(self, tracked_attrs):
        ERR_MSG = 'tracked_attrs must be a list of attribute name strings or None'
        if U.is_sequence(tracked_attrs):
            for attr_name in tracked_attrs:
                assert isinstance(attr_name, str), ERR_MSG
        else:
            assert tracked_attrs is None, ERR_MSG

    def _load_metadata(self):
        if U.f_exists(self.metadata_path()):
            self.metadata = EzDict.load_yaml(self.metadata_path())

    def restore(self, target, reload_metadata=False, check_ckpt_exists=False):
        """
        Args:
            target: can be one of the following semantics
            - negative int: -1, -2 ... counting from the last checkpoint
            - full ckpt file name: "learner.3500.ckpt"
            - string suffix before ".ckpt": "best", "3500"
            reload_metadata: overwrite self.metadata with the metadata.yml file content
            check_ckpt_exists: raise FileNotFoundError if the checkpoint target doesn't exist

        Warnings:
            to restore attrs that point to pytorch Module, you must initialize
            the Module class first, because the checkpoint only saves the
            state_dict, not how to construct your Module
        """
        if reload_metadata:
            self._load_metadata()
        metadata = self.metadata
        if isinstance(target, int):
            assert target < 0, 'target int must be negative, count from last checkpoint'
            assert abs(target) <= metadata.keep_history, \
                'target must < keep_history={}'.format(metadata.keep_history)
            if (check_ckpt_exists and
                abs(target) > len(metadata.history_ckpt_files)):
                raise FileNotFoundError('Last{} ckpt file missing'.format(target))
            ckpt_file = metadata.history_ckpt_files[target]
        elif target.endswith('.ckpt'):
            ckpt_file = target
        else:
            ckpt_file = self.ckpt_name(target)
        ckpt_path = self._get_path(ckpt_file)
        if not U.f_exists(ckpt_path):
            if check_ckpt_exists:
                raise FileNotFoundError(ckpt_path + ' missing.')
            else:
                return
        # overwrite attributes on self.tracked_obj
        with open(ckpt_path, 'rb') as fp:
            data = pickle.load(fp)
        for attr_name in self.metadata.tracked_attrs:
            attr_value = getattr(self.tracked_obj, attr_name)
            if isinstance(attr_value, torch.nn.Module):
                attr_value.load_state_dict(data[attr_name])
            else:
                setattr(self.tracked_obj, attr_name, data[attr_name])

    def _get_path(self, fname):
        return U.f_join(self.folder, fname)

    def metadata_name(self):
        return 'metadata.{}.yml'.format(self.name)

    def metadata_path(self):
        return self._get_path(self.metadata_name())

    def ckpt_name(self, suffix):
        return '{}.{}.ckpt'.format(self.name, suffix)

    def _copy_last_to_best(self, suffix):
        U.f_copy(self.ckpt_path(suffix), self.ckpt_path('best'))

    def ckpt_path(self, suffix):
        return self._get_path(self.ckpt_name(suffix))

    def _save_metadata(self):
        self.metadata.dump_yaml(self.metadata_path())

    def _save_ckpt(self, suffix):
        data = OrderedDict()
        assert self.metadata.tracked_attrs is not None, \
            'tracked_attrs must not be None for save(). ' \
            'Did you forget to restore from an existing checkpoint?'
        for attr_name in self.metadata.tracked_attrs:
            attr_value = getattr(self.tracked_obj, attr_name)
            if isinstance(attr_value, torch.nn.Module):
                data[attr_name] = attr_value.state_dict()
            else:
                data[attr_name] = attr_value
        with open(self.ckpt_path(suffix), 'wb') as fp:
            pickle.dump(data, fp)

    def save(self, score=None, global_steps=None, **ckpt_info):
        """
        save to `<name>.<global_steps>.ckpt` file

        Args:
            score: float scalar to be compared, for `keep_best`
            global_steps: if None, will use internal save counter
            ckpt_info: additional metadata info for this ckpt
        """
        metadata = self.metadata
        if global_steps is None:
            global_steps = metadata.save_counter
        # save the last step
        suffix = global_steps
        self._save_ckpt(suffix)

        # update persistent book-keeping variables in metadata
        metadata.global_steps = global_steps
        metadata.history_ckpt_files.append(self.ckpt_name(suffix))
        # delete older history ckpt files
        delete_point = -metadata.keep_history
        ckpt_to_remove = metadata.history_ckpt_files[:delete_point]
        for ckpt_name in ckpt_to_remove:
            U.f_remove(self._get_path(ckpt_name))
        del metadata.history_ckpt_files[:delete_point]
        # add a ckpt entry to metadata
        ckpt_metadata_entry = {
            'score': score,
            'global_steps': global_steps,
            'save_counter': metadata.save_counter,
            'time': time.time(),
            'datetime': str(datetime.datetime.now()),
        }
        ckpt_metadata_entry.update(ckpt_info)
        metadata.ckpt[self.ckpt_name(suffix)] = ckpt_metadata_entry
        # save the best-performing weights
        if metadata.keep_best:
            assert score is not None, \
                'score cannot be None if keep_best=True'
            if score >= metadata.best_score:
                metadata.best_score = score
                self._copy_last_to_best(suffix)
                metadata.ckpt[self.ckpt_name('best')] = ckpt_metadata_entry

        self._save_metadata()
        metadata.save_counter += 1
