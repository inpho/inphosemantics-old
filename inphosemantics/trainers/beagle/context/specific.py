from ...beagle.environment import read_vecs as read_envecs


def init_proc_doc():

    proc_doc.lexdict = dict(zip(lexicon, xrange(len(lexicon))))
    # Encode stop words
    proc_doc.stoplist = [proc_doc.lexdict[word] for word in stopwords]
    proc_doc.envecs = read_envecs()
    return


def proc_doc(file):
    
    sents = read_sentences(file)
    def encode(sent):
        return [proc_doc.lexdict[word] for word in sent]
    sents = [encode(sent) for sent in sents]
    doclex = set([word for sent in sents for word in sent])
    zerovecs = [np.zeros(dim) for i in xrange(len(doclex))]
    memvecs = dict(zip(doclex, zerovecs))
    
    for sent in sents:
        for i,word in enumerate(sent):
            context = sent[:i] + sent[i+1:]
            for ctxword in context:
                if ctxword not in proc_doc.stoplist:
                    memvecs[word] += proc_doc.envecs[ctxword]

    return memvecs


def gen_vecs():

    init_proc_doc()

    docs = os.listdir(tokpath)
    memvecs = np.zeros((len(lexicon), dim))

    p = Pool()
    results = p.map(proc_doc, docs, 2)
    p.close()

    # Reduce
    for result in results:
        for word,vec in result.iteritems():
            memvecs[word] = memvecs[word] + vec
        
    write_vecs(memvecs)
    return
