import os
import re

import numpy as np

import nltk



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
#                         Tokenizing classes
######################################################################


class ArticlesTokenizer(object):
    """
    """
    def __init__(self, path):

        self.path = path
        self.words = []
        self.tok_names = ['articles', 'paragraphs', 'sentences']

        self.tok_data = None
        self._compute_tokens()
    


    def _compute_tokens(self):

        articles, articles_metadata = textfile_tokenize(self.path)

        article_tokens = []
        paragraph_tokens = []
        sentence_spans = []

        print 'Computing article and paragraph tokens'

        for i,article in enumerate(articles):

            print 'Processing article in', articles_metadata[i]

            paragraphs = paragraph_tokenize(article)
            
            for paragraph in paragraphs:
                sentences = sentence_tokenize(paragraph)

                for sentence in sentences:
                    words = word_tokenize(sentence)

                    self.words.extend(words)
                    
                    sentence_spans.append(len(words))

                paragraph_tokens.append(sum(sentence_spans))
                    
            article_tokens.append(sum(sentence_spans))


        print 'Computing sentence tokens'
        
        sentence_tokens = np.cumsum(sentence_spans)

        article_tokens = zip(article_tokens, articles_metadata)

        self.tok_data = [article_tokens, paragraph_tokens, sentence_tokens]






class ArticleTokenizer(object):
    """
    """
    def __init__(self, filename):

        self.filename = filename
        self.words = []
        self.tok_names = ['paragraphs', 'sentences']

        self.tok_data = None
        self._compute_tokens()
    


    def _compute_tokens(self):

        with open(self.filename, mode='r') as f:
            article = f.read()

        paragraph_tokens = []
        sentence_spans = []

        print 'Computing paragraph tokens'

        paragraphs = paragraph_tokenize(article)
            
        for paragraph in paragraphs:
            sentences = sentence_tokenize(paragraph)

            for sentence in sentences:
                words = word_tokenize(sentence)

                self.words.extend(words)
                    
                sentence_spans.append(len(words))

            paragraph_tokens.append(sum(sentence_spans))
                    

        print 'Computing sentence tokens'
        acc = 0
        sentence_tokens = []
        for i in sentence_spans:
            acc += i
            sentence_tokens.append(acc)


        self.tok_data = [paragraph_tokens, sentence_tokens]






######################################################################
#                               Tests
######################################################################


def test_tokenizers():

    path = 'inphosemantics/tests/data/iep/selected/corpus/plain/'

    iep_articles, iep_meta = textfile_tokenize(path)

    print iep_articles[2], '\n'
    print 'That was the article from', iep_meta[2], '\n'
    
    article_paragraphs = paragraph_tokenize(iep_articles[2])

    print 'Penultimate paragraph:\n\n', article_paragraphs[-2], '\n'

    paragraph_sentences = sentence_tokenize(article_paragraphs[-2])

    print 'Sentences:\n'
    for sent in paragraph_sentences:
        print sent, '\n'

    print 'Words:\n'
    for sent in paragraph_sentences:
        print ', '.join(word_tokenize(sent)), '\n'


    
