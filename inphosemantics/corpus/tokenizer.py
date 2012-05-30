import os
import re
import pickle
import codecs

import nltk



######################################################################
#          Tokenizers for new-style Corpus Class 2012-5-16
######################################################################


def word_tokenize(text):
    """
    Takes a string and returns a list of strings. Intended use: the
    input string is English text and the output consists of the
    lower-case words in this text with numbers and punctuation, except
    for hyphens, removed.

    The core work is done by NLTK's Treebank Word Tokenizer.
    """

    text = rehyph(text)
    text = nltk.TreebankWordTokenizer().tokenize(text)
    tokens = [word.lower() for word in text]
    tokens = strip_punc(tokens)
    tokens = rem_num(tokens)
    
    return tokens



def sentence_tokenize(text):
    """
    Takes a string and returns a list of strings. Intended use: the
    input string is English text and the output consists of the
    sentences in this text.

    This is a wrapper for NLTK's pre-trained Punkt Tokenizer.
    """
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    return tokenizer.tokenize(text)



def paragraph_tokenize(text):
    """
    Takes a string and returns a list of strings. Intended use: the
    input string is English text and the output consists of the
    paragraphs in this text. It's expected that the text marks
    paragraphs with two consecutive line breaks.
    """
    
    return text.split('\n\n')





def textfile_tokenize(path):
    """
    Takes a string and returns a list of strings and a dictionary.
    Intended use: the input string is a directory name containing
    plain text files. The output list is a list of strings, each of
    which is the contents of one of these files. The dictionary is a
    map from indices of this list to the names of the source files.
    """

    out = [],{}
    
    filenames = os.listdir(path)

    for filename in filenames:
        
        filename = os.path.join(path, filename)

        with open(filename, mode='r') as f:
            out[0].append(f.read())
            out[1][len(out[0]) - 1] = filename

        # with codecs.open(filename, encoding='utf-8', mode='r') as f:
        #     out[0].append(f.read())
        #     out[1][len(out[0]) - 1] = filename

    return out

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




######################################################################
#                 Corpus-specific tokenizing classes
######################################################################

class ArticlesTokenizer(object):

    def __init__(self, path):

        self.path = path
        self.word_tokens = []
        self.articles_meta = None
        self.articles, self.paragraphs, self.sentences =\
            self._compute_tokens()


    def _compute_tokens(self):

        articles, metadata = textfile_tokenize(self.path)

        self.articles_meta = metadata

        article_tokens = []
        paragraph_tokens = []
        sentence_spans = []

        #TODO: Write this loop, etc in proper recursive form.

        print 'Computing article and paragraph tokens'
        for i,article in enumerate(articles):

            print 'Processing article in', self.articles_meta[i]

            paragraphs = paragraph_tokenize(article)
            
            for paragraph in paragraphs:
                sentences = sentence_tokenize(paragraph)

                for sentence in sentences:
                    word_tokens = word_tokenize(sentence)

                    self.word_tokens.extend(word_tokens)
                    
                    sentence_spans.append(len(word_tokens))

                paragraph_tokens.append(sum(sentence_spans))
                    
            article_tokens.append(sum(sentence_spans))


        print 'Computing sentence tokens'
        acc = 0
        sentence_tokens = []
        for i in sentence_spans:
            acc += i
            sentence_tokens.append(acc)


        while (article_tokens != []
               and article_tokens[-1] == len(self.word_tokens)):
            article_tokens.pop()

        while (paragraph_tokens != []
               and paragraph_tokens[-1] == len(self.word_tokens)):
            paragraph_tokens.pop()

        while (sentence_tokens != []
               and sentence_tokens[-1] == len(self.word_tokens)):
            sentence_tokens.pop()


        return article_tokens, paragraph_tokens,\
               sentence_tokens


    @property
    def tokens_dict(self):

        d = dict(articles = self.articles,
                 paragraphs = self.paragraphs,
                 sentences = self.sentences)

        return d

    @property
    def tokens_metadata(self):

        d = dict(articles = self.articles_meta,
                 paragraphs = None,
                 sentences = None)

        return d


class IepTokens(ArticlesTokenizer):
    pass

class SepTokens(ArticlesTokenizer):
    pass




######################################################################
#                               Tests
######################################################################


def test_tokenizers():

    path = '/var/inphosemantics/data/iep/complete/corpus/plain'

    iep_articles, iep_meta = textfile_tokenize(path)

    print iep_articles[10], '\n'
    print 'That was the article from', iep_meta[10], '\n'
    
    article_paragraphs = paragraph_tokenize(iep_articles[10])

    print 'Penultimate paragraph:\n\n', article_paragraphs[-2], '\n'

    paragraph_sentences = sentence_tokenize(article_paragraphs[-2])

    print 'Sentences:\n'
    for sent in paragraph_sentences:
        print sent, '\n'

    print 'Words:\n'
    for sent in paragraph_sentences:
        print ', '.join(word_tokenize(sent)), '\n'


def test_IepTokens():

    path = 'test-data/iep/selected/corpus/plain'

    tokens = IepTokens(path)

    print 'Article breaks:\n', tokens.articles
    print '\nParagraph breaks:\n', tokens.paragraphs
    print '\nSentence breaks:\n', tokens.sentences

    return tokens









######################################################################
#                            * Old *
#                    Load the plain text corpus
#                     and dump tokenized corpus
######################################################################

# class Tokenizer(CorpusBase):

#     def __init__(self, corpus, corpus_param):

#         CorpusBase.__init__(self, corpus, corpus_param)
            
#         self.stok = nltk.data.load('tokenizers/punkt/english.pickle')
#         self.wtok = nltk.TreebankWordTokenizer()


#     def tok_corpus(self):
        
#         fs = os.listdir(self.plain_path)
#         for f in fs:
#             name = os.path.splitext(f)[0]
#             self.tok_article(name)

#         return


#     def write_tokens(self, sents, name):

#         tok_file = os.path.join(self.tokenized_path, name + '.pickle')

#         print 'Writing tokenized paragraphs and sentences to', tok_file
#         with open(tok_file, 'w') as f:
#             pickle.dump(sents, f)


#     def tok_article(self, name):
    
#         def st(para):
#             para = self.stok.tokenize(para, realign_boundaries=True)
#             para = map(lambda sent: self.tok_sent(sent), para)
#             para = [sent for sent in para if len(sent) > 1]
#             return para

#         text = self.plain_text(name)
    
#         paras = text.split('\n\n')
#         paras = map(st, paras)
#         paras = [para for para in paras if para]
        
#         self.write_tokens(paras, name)
    
#         return        


#     def tok_sent(self, sent):
    
#         sent = rehyph(sent)
#         sent = self.wtok.tokenize(sent)
#         sent = [word.lower() for word in sent]
#         sent = strip_punc(sent)
#         sent = rem_num(sent)
        
#         return sent
