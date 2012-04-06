from inphosemantics.tools import *


def clean(tree, title):
    """
    Takes an ElementTree object and a string (title of the article)
    and returns the textual content of the article as a list of
    strings.
    """

    root = tree.getroot()
    article = iep_get_body(root)

    if article:
        clr_toc(article)
        iep_clr_pubinfo(article)
        iep_clr_bib(article)
        clr_sectnum(article)

        proc_imgs(article)
        clr_inline(article)
        fill_par(article)
        return flatten(article)
    else:
        return ''


def iep_get_body(root):
    """
    Takes an Element object and returns a subtree containing only the body
    of an SEP article.
    """
    for el in filter_by_tag(root.getiterator(), 'div'):
        if el.get('class') == 'entry':
            return el
    
    print '** Article body not found **'
    return

#TODO: Rewrite in a functional style
def iep_pubinfo(elem):
    """
    Takes an Element object and removes nodes which are likely
    candidates for the author info sections in the IEP.
    """
    cp = cp_map(elem)

    hs = (filter_by_tag(elem.getiterator(), 'h1') +
          filter_by_tag(elem.getiterator(), 'h2') +
          filter_by_tag(elem.getiterator(), 'h3') +
          filter_by_tag(elem.getiterator(), 'h4') +
          filter_by_tag(elem.getiterator(), 'h5') +
          filter_by_tag(elem.getiterator(), 'h6'))
    
    for h in hs:
        for el in h.getiterator() + [h]:
            if ((el.text and re.search(r'Author', el.text)) 
                or 
                (el.tail and re.search(r'Author', el.tail))):
                p = cp[h]
                i = list(p).index(h)
                for node in p[i:]:
                    p.remove(node)
                return
    print '** Author Information not found. **'
    return


#TODO: Rewrite in a functional style        
def iep_clr_bib(elem):
    """
    Takes an Element object and removes nodes which are likely
    candidates for the bibliographies in the IEP.
    """
    cp = cp_map(elem)

    hs = (filter_by_tag(elem.getiterator(), 'h1') +
          filter_by_tag(elem.getiterator(), 'h2') +
          filter_by_tag(elem.getiterator(), 'h3') +
          filter_by_tag(elem.getiterator(), 'h4') +
          filter_by_tag(elem.getiterator(), 'h5') +
          filter_by_tag(elem.getiterator(), 'h6'))
    
    for h in hs:
        for el in h.getiterator() + [h]:
            if ((el.text and 
                 (re.search(r'Reference', el.text) 
                  or re.search(r'Bibliograph', el.text))) 
                or (el.tail and 
                    (re.search(r'Reference', el.tail) 
                     or re.search(r'Bibliograph', el.tail)))):
                p = cp[h]
                i = list(p).index(h)
                for node in p[i:]:
                    p.remove(node)
                return
    print '** Bibliography not found. **'
    return
