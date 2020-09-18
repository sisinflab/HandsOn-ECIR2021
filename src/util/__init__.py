import datetime
import functools
import sys
import os
import socket

def cpu_count():
    """
    Returns the number of CPUs in the system
    """
    if sys.platform == 'win32':
        try:
            num = int(os.environ['NUMBER_OF_PROCESSORS'])
        except (ValueError, KeyError):
            num = 0
    elif 'bsd' in sys.platform or sys.platform == 'darwin':
        comm = '/sbin/sysctl -n hw.ncpu'
        if sys.platform == 'darwin':
            comm = '/usr' + comm
        try:
            with os.popen(comm) as p:
                num = int(p.read())
        except ValueError:
            num = 0
    else:
        try:
            num = os.sysconf('SC_NPROCESSORS_ONLN')
        except (ValueError, OSError, AttributeError):
            num = 0

    if num >= 1:
        return num
    else:
        raise NotImplementedError('cannot determine number of cpus')


def get_server_name():
    return socket.gethostname()

def timethis(some_function):
    """
    Wrapper that profiles the time spent in a function
    """

    @functools.wraps(some_function)
    def wrapper(*args, **kwargs):
        started_at = datetime.datetime.now()
        result = some_function(*args, **kwargs)
        print("Function {name} completed in {time}".format(name=some_function.__name__,
                                                           time=datetime.datetime.now() - started_at))
        return result

    return wrapper
