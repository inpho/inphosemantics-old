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
    from inphosemantics.tools.generic import *


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

from nltk import TreebankWordTokenizer
import nltk.data

plainpath = get_plainpath(__path__[0])
read_plain = mk_read_plain(__path__[0])
write_tokens = mk_write_tokens(__path__[0])

stok = nltk.data.load('tokenizers/punkt/english.pickle')
wtok = TreebankWordTokenizer()


def tok_corpus():
    
    fs = os.listdir(plainpath)
    for f in fs:
        tok_article(f)
    return


def tok_article(filename):
    
    def st(para):
        para = stok.tokenize(para, realign_boundaries=True)
        para = map(lambda sent: tok_sent(sent), para)
        para = [sent for sent in para if len(sent) > 1]
        return para

    text = read_plain(filename)

    paras = text.split('\n\n')
    paras = map(st, paras)
    paras = [para for para in paras if para]

    title = os.path.splitext(filename)[0]
    write_tokens(paras, title)

    return        


def tok_sent(sent):
    
    sent = rehyph(sent)
    sent = wtok.tokenize(sent)
    sent = [word.lower() for word in sent]
    sent = strip_punc(sent)
    sent = rem_num(sent)

    return sent


def strip_punc(tsent):
    p1 = re.compile(r'^(\W*)')
    p2 = re.compile(r'(\W*)$')
    out = []
    for word in tsent:
        w = re.sub(p2, '', re.sub(p1, '', word))
        if w:
            out.append(w)
    return out


def rem_num(tsent):
    p = re.compile(r'(^\D+$)|(^\D*[0-2]\d\D*$)')
    return [word for word in tsent
            if re.search(p, word)]


def rehyph(sent):
    return re.sub(r'(?P<x1>.)--(?P<x2>.)', '\g<x1> - \g<x2>', sent)



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
