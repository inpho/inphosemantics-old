import numpy as np

def gen_nwb(matrix, labels, corpus_name=None,
            model_name=None, filename=None):


    matrix = matrix.astype(np.float32)


    node_header = ['id*int', 'label*string']
    edge_header = ['source*int', 'target*int', 'weight*float']

    if corpus_name:
        node_header.append('corpus*string')
        edge_header.append('corpus*string')

    if model_name:
        node_header.append('model*string')
        edge_header.append('model*string')

    node_header = '*Nodes\n'\
                  + ' '.join(node_header) + '\n'

    edge_header = '*UndirectedEdges\n'\
                  + ' '.join(edge_header) + '\n'



    out = node_header
    
    for i in xrange(1, len(labels)+1):
        out += '{0} "{1}" '.format(i, labels[i-1])

        if corpus_name:
            out += corpus_name + ' '

        if model_name:
            out += model_name + ' '

        out += '\n'



    out += edge_header

    for i in xrange(matrix.shape[0]):
        for j in xrange(i+1, matrix.shape[1]):

            out += '{0} {1} {2} '.format(i+1, j+1, matrix[i,j])
            
            if corpus_name:
                out += corpus_name + ' '

            if model_name:
                out += model_name + ' '

            out += '\n'


    if filename:
        with open(filename, 'w') as f:
            f.write(out)

    return out





def gen_word2word(matrix, labels, filename=None, comment=None):

    out = ''

    if comment:
        out += comment

    out += '\n' # First line of CSV represents title
    
    labels = ['"' + label + '"' for label in labels]

    out += ' , ' + ', '.join(labels) + '\n'

    for i in xrange(matrix.shape[0]):
        
        values = [str(value) for value in matrix[i,:]]
        
        out += labels[i] + ', ' + ', '.join(values) + '\n'


    if filename:
        with open(filename, 'w') as f:
            f.write(out)

    return out
