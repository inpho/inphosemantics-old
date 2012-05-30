
def gen_nwb(matrix, labels, filename=None):

    out = ''

    out += '*Nodes\n'
    out += 'id*int label*string\n'
    
    for i in xrange(1, len(labels)+1):
        out += '{0} "{1}"\n'.format(i, labels[i-1])

    out += '*UndirectedEdges\n'
    out += 'source*int target*int weight*float\n'

    for i in xrange(matrix.shape[0]):
        for j in xrange(i+1, matrix.shape[1]):

            out += '{0} {1} {2}\n'.format(i+1, j+1, matrix[i,j])


    if filename:
        with open(filename, 'w') as f:
            f.write(out)

    return out





def gen_word2word(matrix, labels, filename=None):

    out = ''

    labels = ['"' + label + '"' for label in labels]

    out += ' , ' + ', '.join(labels) + '\n'

    for i in xrange(matrix.shape[0]):
        
        values = [str(value) for value in matrix[i,:]]
        
        out += labels[i] + ', ' + ', '.join(values) + '\n'


    if filename:
        with open(filename, 'w') as f:
            f.write(out)

    return out
