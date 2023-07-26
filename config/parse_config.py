import os
import logging
from pathlib import Path
from functools import reduce, partial
from operator import getitem
from datetime import datetime
from logger import setup_logging
from utils import write_yaml
from vujade import vujade_omegaconf as omegaconf_
from vujade.vujade_debug import printd


class ConfigParser:
    def __init__(self, config, resume=None, modification=None) -> None:
        """
        class to parse configuration json file. Handles hyperparameters for training, initializations of modules, checkpoint saving
        and logging module.
        :param config: Dict containing configurations, hyperparameters for training. contents of `config.yaml` file for example.
        :param resume: String, path to the checkpoint being loaded.
        :param modification: Dict keychain:value, specifying position values to be replaced from config dict.
        """
        # load config file and apply modification
        self._config = _update_config(config, modification)
        self.resume = resume

        # set ckpt_dir and log_dir where trained model and log will be saved, respectively.
        save_dir = Path(self.config['trainer']['model_dir'])
        log_dir = Path(self.config['visualization']['log_dir'])

        exper_name = self.config['name']
        self._run_id = self.config['run_id']
        self._save_dir = save_dir / exper_name / self._run_id
        self._log_dir = log_dir / exper_name / self._run_id

        # make directory for saving checkpoints and log.
        exist_ok = 'exist_ok' in self._run_id
        self.save_dir.mkdir(parents=True, exist_ok=exist_ok)
        self.log_dir.mkdir(parents=True, exist_ok=exist_ok)

        # save updated config file to the checkpoint dir
        write_yaml(self.config, self.save_dir / 'config.yaml')

        # configure logging module
        self.log_config = setup_logging(self.log_dir)
        self.log_levels = {
            0: logging.INFO,
            1: logging.DEBUG,
            2: logging.WARNING,
            3: logging.ERROR,
            4: logging.CRITICAL
        }

    @classmethod
    def from_args(cls, args, options=''):
        """
        Initialize this class from some cli arguments. Used in train, test.
        """
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)
        if not isinstance(args, tuple):
            args = args.parse_args()

        if args.resume is not None:
            resume = Path(args.resume)
            cfg_fname = resume.parent / 'config.yaml'
        else:
            msg_no_cfg = "Configuration file need to be specified. Add '-c config.yaml', for example."
            assert args.config is not None, msg_no_cfg
            resume = None
            cfg_fname = Path(args.config)

        _config = omegaconf_.OmegaConf.load(_spath_filename=str(cfg_fname), _is_interpolation=False)

        # Update _config from args
        if ('run_id' in args) and (args.run_id is not None):
            _config['run_id'] = args.run_id

        config = dict()
        omegaconf_.OmegaConf.cfg2dict(_res=config, _cfg=_config, _key_recursived=None)

        if args.config and resume:
            # update new config for fine-tuning
            config.update(omegaconf_.OmegaConf.load(_spath_filename=args.config, _is_interpolation=True))

        # parse custom cli options into dictionary
        modification = {opt.target : getattr(args, _get_opt_name(opt.flags)) for opt in options}

        res = cls(config, resume, modification)

        # Update trainer.is_amp to False
        if res['trainer']['is_cuda'] is False:
            res['trainer']['is_amp'] = False

        return res

    def init_obj(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding arguments given.

        `object = config.init_obj('name', module, a, b=1)`
        is equivalent to
        `object = module.name(a, b=1)`
        """
        if isinstance(name, str):
            module_name = self[name]['type']
            if 'args' in self[name].keys():
                module_args = dict(self[name]['args'])
                assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
                module_args.update(kwargs)
                res = getattr(module, module_name)(*args, **module_args)
            else:
                res = getattr(module, module_name)(*args)
        else:
            # elif isinstance(name, tuple):
            kinds = name[0]
            name = name[1]
            module_name = self[kinds][name]['type']
            if 'args' in self[kinds][name].keys():
                module_args = dict(self[kinds][name]['args'])
                assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
                module_args.update(kwargs)
                res = getattr(module, module_name)(*args, **module_args)
            else:
                res = getattr(module, module_name)(*args)

        return res

    def init_ftn(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        function with given arguments fixed with functools.partial.

        `function = config.init_ftn('name', module, a, b=1)`
        is equivalent to
        `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return partial(getattr(module, module_name), *args, **module_args)

    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self.config[name]

    def get_logger(self, name, verbosity=1):
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity, self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    # setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir

    @property
    def run_id(self):
        return self._run_id


# helper functions to update config dict with custom cli options
def _update_config(config, modification):
    if modification is None:
        return config

    for k, v in modification.items():
        if v is not None:
            _set_by_path(config, k, v)
    return config


def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')


def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    keys = keys.split(';')
    _get_by_path(tree, keys[:-1])[keys[-1]] = value


def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)
