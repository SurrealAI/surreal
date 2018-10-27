# Checkpointing in Surreal WIP
This document describes how checkpointing works in Surreal. Checkpoint is currently being refactored. Stay tuned.

## The Checkpoint Class
The class that takes care of checkpointing is defined [here](https://github.com/SurrealAI/Surreal/blob/master/surreal/utils/checkpoint.py) in `utils/checkpoint.py`. It is called `Checkpoint`. There is also a subclass that does checkpointing periodically. It keeps track of declared attributes of a class and tries to save them into the file system properly.   
To check how checkpoint saves and restores classes, refer to its implementation.


## Invocation in learner
* Learner keeps a `self._periodic_checkpoint` attribute which is an instance of `PeriodicCheckpoint`. 
* A learner subclass declare attributes to save in method `checkpoint_attributes`. 
* Below is (abbreviated) how the learner sets up the chekpoint instance. See [this link](https://github.com/SurrealAI/Surreal/blob/424976110571153d549a6807b621edc65cc8e006/surreal/learner/base.py#L258) for original code. 
```
def _setup_checkpoint(self):
    tracked_attrs = self.checkpoint_attributes()
    self._periodic_checkpoint = U.PeriodicCheckpoint(
        U.f_join(self.session_config.folder, 'checkpoint'),
        name='learner',
        period=self.session_config.checkpoint.learner.periodic,
        min_interval=self.session_config.checkpoint.learner.min_interval,
        tracked_obj=self,
        tracked_attrs=tracked_attrs,
        keep_history=self.session_config.checkpoint.learner.keep_history,
        keep_best=self.session_config.checkpoint.learner.keep_best,
    )
```
* Learner checks for restoration [here](https://github.com/SurrealAI/Surreal/blob/424976110571153d549a6807b621edc65cc8e006/surreal/learner/base.py#L293)

## Commandline Argument
The learner interacts with the checkpoint through config. It can also be affected by commandline arguments, e.g. [here](https://github.com/SurrealAI/Surreal/blob/424976110571153d549a6807b621edc65cc8e006/surreal/main_scripts/learner_main.py#L8). 