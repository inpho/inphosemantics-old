import os.path
import re
import pickle
from nltk import TreebankWordTokenizer
import nltk.data

from inphosemantics.corpus import CorpusBase


######################################################################
#                    Load the plain text corpus
#                     and dump tokenized corpus
######################################################################

class Tokenizer(CorpusBase):

    def __init__(self, corpus, corpus_param):

        CorpusBase.__init__(self, corpus, corpus_param)
            
        self.stok = nltk.data.load('tokenizers/punkt/english.pickle')
        self.wtok = TreebankWordTokenizer()


    def tok_corpus(self):

        plain_path = os.path.join(self.corpus_path, 'plain')
        
        fs = os.listdir(plain_path)
        for f in fs:
            name = os.path.splitext(f)[0]
            self.tok_article(name)

        return


    def write_tokens(self, sents, name):

        tok_file = os.path.join(self.corpus_path, 
                                'tokenized', name + '.pickle')

        print 'Writing tokenized paragraphs and sentences to', tok_file
        with open(tok_file, 'w') as f:
            pickle.dump(sents, f)


    def tok_article(self, name):
    
        def st(para):
            para = self.stok.tokenize(para, realign_boundaries=True)
            para = map(lambda sent: self.tok_sent(sent), para)
            para = [sent for sent in para if len(sent) > 1]
            return para

        text = self.plain_text(name)
    
        paras = text.split('\n\n')
        paras = map(st, paras)
        paras = [para for para in paras if para]
        
        self.write_tokens(paras, name)
    
        return        


    def tok_sent(self, sent):
    
        sent = rehyph(sent)
        sent = self.wtok.tokenize(sent)
        sent = [word.lower() for word in sent]
        sent = strip_punc(sent)
        sent = rem_num(sent)
        
        return sent


######################################################################
#                              Utilities
######################################################################


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

