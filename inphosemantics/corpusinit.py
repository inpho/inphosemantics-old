from inphosemantics import *
import re
from unidecode import unidecode

try:
    from specific import *
    print ('Found corpus-specific processing functions in\n    '
           + os.path.join(__path__[0], 'specific.py'))
except ImportError:
    print ('Corpus-specific HTML processing functions '
           'not found; using default.')
    from inphosemantics.tools import *


######################################################################
#          Load corpus HTML articles as ElementTree objects 
#               and dump paragraph-tokenized plain text
######################################################################

htmlpath = get_htmlpath(__path__[0])
read_html = mk_read_html(__path__[0])
write_plain = mk_write_plain(__path__[0])


def html2plain(title = ''):

    if title:
        plain = clean(read_html(title), title)
        plain = '\n\n'.join(plain)

        plain = unidecode(plain)
        unknown = re.compile('\\[\\?\\]')
        plain = unknown.sub(' ', plain)

        write_plain(plain, title)
            
    else:
        for title in os.listdir(htmlpath):
            html2plain(title)
            
    return


######################################################################
#                    Load the plain text corpus
#                     and dump tokenized corpus
######################################################################

tok_article = mk_tok_article(__path__[0])
tok_corpus = mk_tok_corpus(__path__[0])


######################################################################
#                   Generate a lexicon for a corpus
######################################################################

tokpath = get_tokpath(__path__[0])
read_sentences = mk_read_sentences(__path__[0])

write_lexicon = mk_write_lexicon(__path__[0])

def gen_lexicon():
    
    lexicon = set()
    for f in os.listdir(tokpath):
        for sent in read_sentences(f):
            lexicon = lexicon.union(set(sent))
    
    write_lexicon(list(lexicon))
    return


######################################################################
#                Additional readers for outside modules
######################################################################


lexicon = mk_read_lexicon(__path__[0])()
stopwords = mk_read_stopwords()()
