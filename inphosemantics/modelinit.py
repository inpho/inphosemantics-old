from inphosemantics import *
from ...corpus import *
from ...beagle import *
from inphosemantics.viewers import *

try:
    from specific import *
    print ('Found model-specific training functions in\n    '
           + os.path.join(__path__[0], 'specific.py'))
except ImportError:
    print ('Training functions not found.')


read_vecs = mk_read_vecs(__path__[0])
write_vecs = mk_write_vecs(__path__[0])

gen_cosines = mk_gen_cosines(__path__[0])

similar = mk_similar(lexicon, stopwords, __path__[0])
display_similar = mk_display_similar(lexicon, stopwords, __path__[0])

parse_query = mk_parse_query(lexicon, stopwords)
