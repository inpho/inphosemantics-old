from inphosemantics.tools import *


def clean(tree, title):
    """
    Takes an ElementTree object and a string (title of the article)
    and returns the textual content of the article as a list of
    strings.
    """

    root = tree.getroot()
    article = sep_get_body(root)
    
    if article:
        clr_toc(article)
        sep_clr_pubinfo(article)
        sep_clr_bib(article)
        clr_sectnum(article)
        
        proc_imgs(article)
        clr_inline(article)
        fill_par(article)
        return flatten(article)
    else:
        return ''
    

def sep_get_body(tree):
    """
    Takes an Element object and returns a subtree containing only the body
    of an SEP article.
    """
    for el in filter_by_tag(tree.getiterator(), 'div'):
        if el.attrib['id'] == 'aueditable':
            return el

    print '** Article body not found **'
    return


#TODO: Rewrite in a functional style
def sep_clr_pubinfo(elem):
    """
    Takes an Element object and removes any node with the id
    attribute 'pubinfo'. (For SEP)
    """
    cp = cp_map(elem)
    for el in filter_by_tag(elem.getiterator(), 'div'):
        if (el.attrib.has_key('id') and
            el.attrib['id'] == 'pubinfo'):
            cp[el].remove(el)
            return
        
    print '** Pub info not found **'


#TODO: Rewrite in a functional style        
def sep_clr_bib(elem):
    """
    Takes an Element object and removes nodes which are likely
    candidates for the bibliography in the SEP.
    """
    cp = cp_map(elem)
    
    hs = (filter_by_tag(elem.getiterator(), 'h2') +
          filter_by_tag(elem.getiterator(), 'h3'))
    
    for h in hs:
        for el in h.getiterator() + [h]:
            if ((el.text and
                 re.search(r'Bibliography', el.text)) or
                (el.tail and
                 re.search(r'Bibliography', el.tail))):
                p = cp[h]
                i = list(p).index(h)
                for node in p[i:]:
                    p.remove(node)
                return

    print '** Bibliography not found. **'
    return
