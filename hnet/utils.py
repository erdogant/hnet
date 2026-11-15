"""Shared utilities for hnet (logging and helpers)."""
# ------------------------------------
# Name        : utils.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Licence     : See licences
# ------------------------------------

import logging
from datetime import datetime

# Optional TRACE level (also registered in hnet.__init__)
if not hasattr(logging, 'TRACE'):
    logging.TRACE = 5
    logging.addLevelName(logging.TRACE, 'TRACE')

# Always use the package logger so set_logger() controls every message
logger = logging.getLogger('hnet')


# %% Verbosity helpers
# =============================================================================
def convert_verbose_to_old(verbose):
    """Convert new verbosity levels to the legacy 0..4 scale."""
    if verbose is None:
        verbose = 0
    if isinstance(verbose, str) or (isinstance(verbose, int) and verbose >= 10):
        status_map = {
            60: 0, 'silent': 0, 'off': 0, 'no': 0, None: 0,
            40: 1, 'error': 1, 'critical': 1,
            30: 2, 'warning': 2,
            20: 3, 'info': 3,
            10: 4, 'debug': 4,
            5: 4, 'trace': 4,
        }
        return status_map.get(verbose, 0)
    return verbose


def convert_verbose_to_new(verbose):
    """Convert legacy verbosity ints (0..6) to string names."""
    if verbose is None:
        verbose = 0
    if not isinstance(verbose, str) and isinstance(verbose, int) and verbose < 10:
        status_map = {
            0: 'silent',
            6: 'silent',
            1: 'critical',
            2: 'warning',
            3: 'info',
            4: 'debug',
            5: 'debug',
        }
        if verbose >= 2:
            print('[hnet] WARNING use the standardized verbose status. '
                  'The status [1-6] will be deprecated in future versions.')
        return status_map.get(verbose, 'silent')
    return verbose


def get_logger():
    """Return the effective logging level of the hnet package logger.

    Returns
    -------
    int
        Current effective level (e.g. ``logging.INFO`` == 20).
    """
    return logging.getLogger('hnet').getEffectiveLevel()


def set_logger(verbose='info'):
    """Set the hnet logger level.

    Parameters
    ----------
    verbose : str or int or None, default='info'
        * None, 0, 'silent', 'off', 'no' -> silence all messages
        * 'error' / 40
        * 'warning' / 30
        * 'info' / 20  (default)
        * 'debug' / 10
        * 'trace' / 5
        Legacy ints 1..5 (error..debug) are still accepted.

    Returns
    -------
    int
        The numeric logging level that was set.

    Examples
    --------
    >>> import hnet as hn
    >>> hn.set_logger('info')
    >>> hn.set_logger('warning')
    >>> hn.set_logger(None)          # silence
    """
    # Convert legacy 1..5 ints to string names first
    verbose = convert_verbose_to_new(verbose)

    if verbose is None or verbose in (0, 'silent', 'off', 'no', 'None'):
        level = logging.CRITICAL + 10  # above CRITICAL = silent
    elif isinstance(verbose, str):
        name = verbose.strip().lower()
        mapping = {
            'critical': logging.CRITICAL,
            'error': logging.ERROR,
            'warning': logging.WARNING,
            'warn': logging.WARNING,
            'info': logging.INFO,
            'debug': logging.DEBUG,
            'trace': getattr(logging, 'TRACE', 5),
            'silent': logging.CRITICAL + 10,
            'off': logging.CRITICAL + 10,
            'no': logging.CRITICAL + 10,
        }
        if name not in mapping:
            raise ValueError(f'[hnet] Unknown log level: {verbose}')
        level = mapping[name]
    elif isinstance(verbose, int):
        # Standard logging level (10, 20, 30, ...) or TRACE
        if verbose >= 10 or verbose == getattr(logging, 'TRACE', 5):
            level = verbose
        else:
            legacy = {
                0: logging.CRITICAL + 10,
                1: logging.ERROR,
                2: logging.WARNING,
                3: logging.INFO,
                4: logging.DEBUG,
                5: getattr(logging, 'TRACE', 5),
                6: logging.CRITICAL + 10,
            }
            level = legacy.get(verbose, logging.INFO)
    else:
        raise TypeError(f'[hnet] verbose must be str, int, or None; got {type(verbose)}')

    # Apply level on the package logger. Keep handlers at the lowest level so
    # that *only* the logger level controls what is emitted (raising a handler
    # to WARNING would permanently hide DEBUG even after set_logger('debug')).
    _apply_level('hnet', level)

    # Keep dependency loggers in sync. df2onehot's own set_logger() is unreliable
    # (it targets a wrong logger name in some releases), which caused its messages
    # to only respect verbosity on a second run. Setting it here makes the first
    # association_learning() call already honor verbose=.
    _apply_level('df2onehot', level)

    return level


def _apply_level(name, level):
    """Set *level* on logger *name* and all of its existing children."""
    pkg = logging.getLogger(name)
    pkg.setLevel(level)
    for handler in pkg.handlers:
        # Handler stays open; logger level is the filter.
        handler.setLevel(logging.DEBUG)

    prefix = name + '.'
    for child_name in list(logging.Logger.manager.loggerDict):
        if child_name.startswith(prefix):
            logging.getLogger(child_name).setLevel(level)


def disable_tqdm():
    """Return True when tqdm progress bars should be disabled (level >= WARNING)."""
    return logging.getLogger('hnet').getEffectiveLevel() >= logging.WARNING


def check_logger(verbose=None):
    """Emit one message at each level (useful for verifying the logger)."""
    if verbose is not None:
        set_logger(verbose)
    pkg = logging.getLogger('hnet')
    pkg.debug('DEBUG')
    pkg.info('INFO')
    pkg.warning('WARNING')
    pkg.critical('CRITICAL')


def log_prefix(level='INFO'):
    """Return a logger-formatted prefix suitable for tqdm descriptions.

    Examples
    --------
    >>> from hnet.utils import log_prefix
    >>> print(log_prefix())
    [15-11-2026 11:39:05] [hnet] [INFO]
    """
    return (
        f'[{datetime.now().strftime("%d-%m-%Y %H:%M:%S")}] '
        f'[{logger.name}] '
        f'[{level}]'
    )