from inphosemantics.corpus.tokenizer import *


plain_path = 'inphosemantics/tests/data/iep/selected/corpus/plain'


def test_several():

    iep_articles, iep_meta = textfile_tokenize(plain_path)



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
