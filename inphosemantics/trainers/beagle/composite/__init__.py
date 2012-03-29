from inphosemantics import *

from ...corpus import *
from ...beagle import *
from inphosemantics.viewers import *
from ...beagle.context import read_vecs as read_ctxvecs
from ...beagle.order import read_vecs as read_ordvecs


read_vecs = mk_read_vecs(__path__[0])
write_vecs = mk_write_vecs(__path__[0])

gen_cosines = mk_gen_cosines(__path__[0])

similar = mk_similar(lexicon, stopwords, __path__[0])
display_similar = mk_display_similar(lexicon, stopwords, __path__[0])

######################################################################


def gen_vecs():

    ctxvecs = read_ctxvecs()
    ordvecs = read_ordvecs()
    memvecs = ctxvecs + ordvecs

    write_vecs(memvecs)

    return
