from inphosemantics import * 

extstop = ['especially', 'many', 'several', 'perhaps', 
           'various', 'key', 'found', 'particularly', 'later', 'could',
           'might', 'must', 'would', 'may', 'actually', 'either',
           'without', 'one', 'also', 'neither']

def mk_similar(lexicon, stopwords, cospath):

    def similar(word):
        # TODO: User friendly error handling
        i = lexicon.index(word)
        simvec = mk_read_simvec(cospath)(i)
        
        pairs = zip(lexicon, simvec)
        print 'Sorting results'
        pairs.sort(key=lambda p: p[1], reverse = True)

        return pairs

    return similar


def mk_display_similar(lexicon, stopwords, cospath):

    stopwords = stopwords + extstop

    def display_similar(word, n=20):
        
        pairs = mk_similar(lexicon, stopwords, cospath)(word)
        pairs = pairs[:(n + len(stopwords))]

        print 'Filtering stop words'
        pairs = filter(lambda p: p[0] not in stopwords, pairs)

        pairs = pairs[:n]

        # TODO: Make pretty printer robust
        print ''.join(['-' for i in xrange(38)])
        print '{0:^25}{1:^12}'.format('Word','Similarity')
        for w,v in pairs:
            print '{0:<25}{1:^12.3f}'.format(w,float(v))
        print ''.join(['-' for i in xrange(38)])
            
        return

    return display_similar
