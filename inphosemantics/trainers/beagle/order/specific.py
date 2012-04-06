from ...beagle.environment import read_vecs as read_envecs
from ...beagle.environment import RandomWordGen

from numpy.dual import fft, ifft


def init_proc_doc():
    proc_doc.lexdict = dict(zip(lexicon, xrange(len(lexicon))))
    envecs = read_envecs()

    print 'Computing DFTs of environment vectors'
    # p = Pool()
    # fftenvecs = p.map(fft, envecs, 20)
    # p.close()

    fftenvecs = map(fft, envecs)

    proc_doc.fftenvecs = fftenvecs
    
    proc_doc.placeholder = fft(RandomWordGen(d=dim).make_rep(''))

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
        for k in xrange(1, lmda):
            for i,word in enumerate(sent):
                
                a = i - k
                left = 0 if a < 0 else a
                b = i + k
                right = len(sent) if b > len(sent) else b
                
                fftvecseq = ([proc_doc.fftenvecs[w] for w in sent[left:i]]
                             + [proc_doc.placeholder]
                             + [proc_doc.fftenvecs[w] for w in sent[i+1:right]])

                ordvec = ifft(reduce(lambda v1,v2: v1*v2, fftvecseq))
                
                memvecs[word] += ordvec

    return memvecs


def gen_vecs():

    init_proc_doc()

    docs = os.listdir(tokpath)
    memvecs = np.zeros((len(lexicon), dim))

    q = Pool()
    results = q.map(proc_doc, docs, 2)
    q.close()

    # Reduce
    for result in results:
        for word,vec in result.iteritems():
            memvecs[word] = memvecs[word] + vec

    write_vecs(memvecs)
    return
