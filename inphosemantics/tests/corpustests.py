from inphosemantics.corpus import *

################################################################
# TODO: Update this testing code to new version of corpus module
################################################################

# def genBaseCorpus():
#     return BaseCorpus(['cats','chase','dogs','dogs',
#                        'do','not','chase','cats'],
#                       {'sentences' : [3]})


# def test_BaseCorpus():

#     from inphosemantics.corpus.tokenizer import IepTokens

#     path = 'test-data/iep/selected/corpus/plain'

#     tokens = IepTokens(path)

#     c = BaseCorpus(tokens.word_tokens, tokens.tokens)

#     print 'First article:\n', c.view_tokens('articles')[1]
#     print '\nFirst five paragraphs:\n', c.view_tokens('paragraphs')[:5]
#     print '\nFirst ten sentences:\n', c.view_tokens('sentences')[:10]

#     print '\nLast article:\n', c.view_tokens('articles')[-1]
#     print '\nLast five paragraphs:\n', c.view_tokens('paragraphs')[-5:]
#     print '\nLast ten sentences:\n', c.view_tokens('sentences')[-10:]

#     return c


# def test_integer_corpus():

#     from inphosemantics.corpus.tokenizer import IepTokens

#     path = 'test-data/iep/selected/corpus/plain'

#     tokens = IepTokens(path)

#     c = BaseCorpus(tokens.word_tokens, tokens.tokens)

#     int_corpus, encoder = c.encode_corpus()

#     print 'First article:\n',\
#           int_corpus.view_tokens('articles', encoder)[1]
#     print '\nFirst five paragraphs:\n',\
#           int_corpus.view_tokens('paragraphs', encoder)[:5]
#     print '\nFirst ten sentences:\n',\
#           int_corpus.view_tokens('sentences', encoder)[:10]

#     print '\nLast article:\n',\
#           int_corpus.view_tokens('articles', encoder)[-1]
#     print '\nLast five paragraphs:\n',\
#           int_corpus.view_tokens('paragraphs', encoder)[-5:]
#     print '\nLast ten sentences:\n',\
#           int_corpus.view_tokens('sentences', encoder)[-10:]

#     return int_corpus, encoder


# def test_Corpus():

#     from inphosemantics.corpus.tokenizer import IepTokens

#     path = 'test-data/iep/selected/corpus/plain'

#     tokens = IepTokens(path)

#     c = Corpus(tokens.word_tokens, tokens.tokens,
#                tokens.tokens_metadata)

#     print 'First article:\n',\
#           c.view_tokens('articles', True)[1]
#     print '\nFirst five paragraphs:\n',\
#           c.view_tokens('paragraphs', True)[:5]
#     print '\nFirst ten sentences:\n',\
#           c.view_tokens('sentences', True)[:10]

#     print '\nLast article:\n',\
#           c.view_tokens('articles', True)[-1]
#     print '\nLast five paragraphs:\n',\
#           c.view_tokens('paragraphs', True)[-5:]
#     print '\nLast ten sentences:\n',\
#           c.view_tokens('sentences', True)[-10:]

#     print '\nSource of second article:',\
#           c.view_metadata('articles')[2]

#     return c


# def test_Corpus_dumpz():

#     from inphosemantics.corpus.tokenizer import IepTokens

#     path = 'test-data/iep/selected/corpus/plain'
#     filename = 'test-data/iep/selected/corpus/iep-selected.pickle.bz2'

#     tokens = IepTokens(path)

#     c = Corpus(tokens.word_tokens, tokens.tokens,
#                tokens.tokens_metadata)

#     c.dumpz(filename)


# def test_Corpus_dumpz_plato():

#     from inphosemantics.corpus.tokenizer import IepTokens

#     path = 'test-data/iep/plato/corpus/plain'
#     filename = 'test-data/iep/plato/corpus/iep-plato.pickle.bz2'

#     tokens = IepTokens(path)

#     c = Corpus(tokens.word_tokens, tokens.tokens,
#                tokens.tokens_metadata)

#     c.dumpz(filename)



