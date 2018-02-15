"""
Checkpoint that supports labeling and performance tracking
"""
import torch
import pickle
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
    - save_counter: how many times "save()" has been called
    - history_ckpt_names: names of ckpt files from old to new
    - global_steps: current global_steps counter
    - ckpt: dict of ckpt file names -> info
    """
    def __init__(self, folder, name, tracked_obj,
                 keep_history=1, keep_best=True):
        """
        Args:
            folder: checkpoint folder path
            name: prefix of the checkpoint
            tracked_obj: arbitrary object whose certain attributes will be tracked.
                typically Learner or Agent class
            keep_history: how many checkpoints in the past to keep
            keep_best: whether to save the best performing parameters
        """
        self.folder = U.f_expand(folder)
        self.name = name
        self._tracked_obj = tracked_obj
        self._tracked_attrs = []
        self._tracked_torch_modules = OrderedDict()
        if U.f_exists(self.metadata_path()):
            self.metadata = EzDict.load_yaml(self.metadata_path())
            self._save_counter = self.metadata.save_counter
            self._history_ckpt_names = self.metadata.history_ckpt_names
        else:
            self.metadata = EzDict()
            self._save_counter = self.metadata.save_counter = 0
            self._history_ckpt_names = self.metadata.history_ckpt_names = []
            self.metadata.ckpt = {}
        self._period_counter = 1
        assert keep_history >= 0
        self._keep_history = keep_history
        self._keep_best = keep_best

    def register_attrs(self, attr_names):
        assert U.is_sequence(attr_names)
        for attr_name in attr_names:
            assert isinstance(attr_name, str), \
                'attr_names must be a list of attribute name strings'
            attr = getattr(self._tracked_obj, attr_name)
            assert not isinstance(attr, torch.nn.Module), \
                'please use `register_torch_module()` method to track torch.nn.Module'
        self._tracked_attrs += list(attr_names)
        return self

    def register_torch_module(self, name, torch_module, *init_args, **init_kwargs):
        """
        Do not include custom classes in init_args unless they are pickleable.

        Args:
            torch_module: instance of torch.nn.Module
            *init_args: args for Module constructor.
            **init_kwargs: kwargs for Module constructor
        """
        assert isinstance(torch_module, torch.nn.Module)
        self._tracked_torch_modules[name] = {
            'module': torch_module,
            'init_args': init_args,
            'init_kwargs': init_kwargs
        }
        return self

    def _get_path(self, fname):
        return U.f_join(self.folder, fname)

    def metadata_name(self):
        return 'metadata.{}.yml'.format(self.name)

    def metadata_path(self):
        return self._get_path(self.metadata_name())

    def ckpt_name(self, suffix):
        return '{}.{}.ckpt'.format(self.name, suffix)

    def ckpt_path(self, suffix):
        return self._get_path(self.ckpt_name(suffix))

    def _save_metadata(self):
        self.metadata.dump_yaml(self.metadata_path())

    def _save_ckpt(self, suffix):
        data = OrderedDict()
        data['attrs'] = OrderedDict()
        for attr_name in self._tracked_attrs:
            data['attrs'][attr_name] = getattr(self._tracked_obj, attr_name)
        data['modules'] = OrderedDict()
        for module_name, module_info in self._tracked_torch_modules.items():
            module_dict = OrderedDict()
            module_dict['state_dict'] = module_info['module'].state_dict()
            module_dict['init_args'] = module_info['init_args']
            module_dict['init_kwargs'] = module_info['init_kwargs']
            data['modules'][module_name] = module_dict
        with open(self.ckpt_path(suffix), 'wb') as fp:
            pickle.dump(data, fp)

    def save(self, global_steps=None):
        """
        Args:
            global_steps: if None, will use internal save counter
        """
        if global_steps is None:
            global_steps = self._save_counter
        metadata = self.metadata
        metadata.global_steps = global_steps
        metadata.save_counter = self._save_counter
        self._history_ckpt_names.append(self.ckpt_name(suffix))
        self._save_ckpt(suffix)

        # delete older history ckpt files
        ckpt_to_remove = self._history_ckpt_names[:-(self._keep_history+1)]
        for ckpt_name in ckpt_to_remove:
            U.f_remove(self._get_path(ckpt_name))
        del self._history_ckpt_names[:-(self._keep_history+1)]
        metadata.history_ckpt_names = self._history_ckpt_names

        self._save_metadata()
        self._save_counter += 1
