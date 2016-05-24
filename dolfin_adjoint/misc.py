import backend
from contextlib import contextmanager

def uniq(seq):
    '''Remove duplicates from a list, preserving order'''
    seen = set()
    seen_add = seen.add
    return [ x for x in seq if x not in seen and not seen_add(x)]


def pause_annotation():
    flag = backend.parameters["adjoint"]["stop_annotating"]
    backend.parameters["adjoint"]["stop_annotating"] = True
    return flag

def continue_annotation(flag):
    backend.parameters["adjoint"]["stop_annotating"] = flag

@contextmanager
def annotations(flag):

    orig = backend.parameters["adjoint"]["stop_annotating"]
    backend.parameters["adjoint"]["stop_annotating"] = not flag

    yield

    backend.parameters["adjoint"]["stop_annotating"] = orig
