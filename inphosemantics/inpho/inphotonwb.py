from inphosemantics import Model

queries = ['metaphysics', 'aristotle', 'ethics', 'causation', 'epistemology']

n = 10

def simtonwb(out='test-simtonwb.txt'):

    iep = Model('iep','complete','beagle','composite')
    sep = Model('sep','complete','beagle','composite')

    data_iep = [iep.similar(query, n) for query in queries]
    data_sep = [sep.similar(query, n) for query in queries]

    iep_edges = [(queries[i], term, value, 'iep') 
                 for i,results in enumerate(data_iep) 
                 for (term, value) in results]

    sep_edges = [(queries[i], term, value, 'sep') 
                 for i,results in enumerate(data_sep) 
                 for (term, value) in results]
    
    def get_edge(source, target, edges):
        for (src, tgt, v, s) in edges:
            if src == source and tgt == target:
                return (src, tgt, v, s)
        return False

    edges = iep_edges + sep_edges

    terms = zip(*edges)[1]
    terms = set(terms)
    terms = list(terms)    

    term_index = dict(zip(terms,xrange(len(terms))))

    f = open(out, 'w')

    print >>f, '*Nodes'
    print >>f, 'id*int label*string'
    for i,term in enumerate(terms):
        print >>f, i, '"' + term + '"' 
    print >>f, '*UndirectedEdges'
    print >>f, 'source*int target*int corpus*string weight*float'

    for term1 in terms:
        for term2 in terms:
            sep_edge = get_edge(term1, term2, sep_edges)
            iep_edge = get_edge(term1, term2, iep_edges)
            if sep_edge and iep_edge:
                print >>f, term_index[term1], term_index[term2], 'sep+iep', (sep_edge[2] + iep_edge[2])/2
            elif sep_edge:
                print >>f, term_index[term1], term_index[term2], 'sep', sep_edge[2]
            elif iep_edge:
                print >>f, term_index[term1], term_index[term2], 'iep', iep_edge[2]
