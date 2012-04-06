from ...beagle.context import read_vecs as read_ctxvecs
from ...beagle.order import read_vecs as read_ordvecs

from . import *


def gen_vecs():

    ctxvecs = read_ctxvecs()
    ordvecs = read_ordvecs()
    memvecs = ctxvecs + ordvecs

    write_vecs(memvecs)

    return
