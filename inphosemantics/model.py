import os.path
import pickle
from nltk.corpus import stopwords as nltk_stopwords


class Model(Corpus):

    data_root = Corpus.data_root

    def __init__(self, corpus, corpus_param, model, model_param):

        Corpus.__init__(self, corpus, corpus_param)

        self.model = model
        self.model_param = model_param

        self.model_path =\
            os.path.join(Model.data_root, self.corpus, self.corpus_param,
                         self.model, self.model_param)


    def sim_vec(self, index):
        
        filename = '-'.join([self.corpus, self.corpus_param,
                             self.model, self.model_param,
                             'cosines-' + str(index) + '.pickle'])
        filename = os.path.join(self.model_path, 'cosines', filename)
        
        print 'Reading similarity vector', filename
        with open(filename, mode='r') as f:
            return pickle.load(f)


    def similar(self, word, n=-1, 
                filter_stopwords = True, filter_degenerate = True):

        # TODO: User friendly error handling
        i = self.lexicon.index(word)
        vec = self.sim_vec(i)
        
        pairs = zip(self.lexicon, vec)
        print 'Sorting results'
        pairs.sort(key=lambda p: p[1], reverse = True)
        
        if filter_degenerate:
            print 'Filtering degenerate vectors'
            pairs = filter(lambda p: p[1] != -2, pairs)

        if n != -1:
            pairs = pairs[:(n + len(self.stopwords))]

        if filter_stopwords:
            print 'Filtering stop words'
            pairs = filter(lambda p: p[0] not in self.stopwords, pairs)

        if n != -1:
            pairs = pairs[:n]

        return pairs

    def display_similar(word, n=20):
        
        pairs = mk_similar(lexicon, stopwords, cospath)(word, n=n)

        # TODO: Make pretty printer robust
        print ''.join(['-' for i in xrange(38)])
        print '{0:^25}{1:^12}'.format('Word','Similarity')
        for w,v in pairs:
            print '{0:<25}{1:^12.3f}'.format(w,float(v))
        print ''.join(['-' for i in xrange(38)])
            
        return


    def print_similar(self, word, n=20):
    
        pairs = self.similar(word, n=n)

        # TODO: Make pretty printer robust
        print ''.join(['-' for i in xrange(38)])
        print '{0:^25}{1:^12}'.format('Word','Similarity')
        for w,v in pairs:
            print '{0:<25}{1:^12.3f}'.format(w,float(v))
        print ''.join(['-' for i in xrange(38)])
            
        return
