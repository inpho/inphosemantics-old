import re
from xml.etree.ElementTree import *
from unidecode import unidecode


######################################################################
#                 HTML to paragraph-tokenized plain text
######################################################################

def clean(tree, title):
    """
    Takes an ElementTree object and a string (title of the article)
    and returns the textual content of the article as a list of
    strings.
    """

    root = tree.getroot()
    article = get_body(root)

    # clr_toc(article)
    # clr_sectnum(article)

    proc_imgs(article)
    clr_inline(article)
    fill_par(article)
    return flatten(article)


def get_body(root):
    """
    Takes an Element object and returns a subtree containing only the body
    of an SEP article.
    """
    body = filter_by_tag(root.getiterator(), 'body')
    if len(body) != 1:
        raise Exception('Unique article body not found.')
    else:
        return body[0]


#TODO: Rewrite in a functional style
def clr_toc(elem):
    """
    Takes an Element object and removes any subtrees which are
    unordered or ordered lists of anchors. Such things are usually tables of
    contents.
    """
    cp = cp_map(elem)
    uls = filter_by_tag(elem.getiterator(), 'ul')
    ols = filter_by_tag(elem.getiterator(), 'ol')
    for l in ols[:] + uls[:]:
        if reduce(lambda v1, v2: v1 and v2,
                  [filter_by_tag(li.getiterator(), 'a') is not []
                   for li in filter_by_tag(l.getiterator(), 'li')]):
            cp[l].remove(l)
            return
    print '** TOC not found **'
    return


#TODO: Rewrite in a functional style    
def clr_sectnum(elem):
    """
    Takes an Element object and removes text identifying section
    numbers.
    """
    hs = (filter_by_tag(elem.getiterator(), 'h1') +
          filter_by_tag(elem.getiterator(), 'h2') +
          filter_by_tag(elem.getiterator(), 'h3') +
          filter_by_tag(elem.getiterator(), 'h4') +
          filter_by_tag(elem.getiterator(), 'h5') +
          filter_by_tag(elem.getiterator(), 'h6'))
    
    n = re.compile('^[a-zA-Z ]*[0-9 \.]+ *')
    
    for h in hs:
        for el in h.getiterator() + [h]:
            if el.text:
                el.text = re.sub(n, '', el.text)
            elif el.tail:
                el.tail = re.sub(n, '', el.tail)
                
    return


#TODO: Rewrite in a functional style
def proc_imgs(elem):
    """
    Takes an Element object and removes img nodes or replaces them
    with div nodes containing the alt text.
    """
    imgs = filter_by_tag(elem.getiterator(), 'img')
    
    for img in imgs:
        alt = img.attrib['alt']
        if alt:
            img.tag = get_prefix(img) + 'div'
            img.text = alt


#TODO: Rewrite in a functional style
def clr_inline(elem):
    """
    Takes an Element object, looks for nodes whose tags are xhmtl
    inline tags, and removes these nodes while appending the contents
    of their text and tail attributes in the appropriate places.
    """
    inline = ['b', 'em', 'i', 'tt', 'big', 'small', 'bdo',
              'strong', 'dfn', 'code', 'samp', 'kbd', 'var',
              'cite', 'span', 'font', 'sub', 'sup', 's',
              'strike', 'center', 'a', 'abbr', 'acronym',
              'u', 'br', 'del', 'ins', 'q']
    
    # Recall that text in xhtml documents will show up in two places
    # in an ElementTree Element Object: either in the text or in the tail
    # attribute. Suppose you have this chunk of html
    # '<p>real<em>foo</em>bar</p>'. The text attribute for the node
    # corresponding to the p tag has value 'real'. The text attribute for
    # the node corresponding to the em tag has value 'foo'. Where should
    # 'bar' go? In fact, the *tail* attribute of em stores 'bar'.

    def clr(el, cp):
        for node in el[:]:
            clr(node, cp_map(cp[el]))
        if [inl for inl in inline if match_qname(inl, el.tag)]:
            i = list(cp[el]).index(el)
            if i == 0:
                # no left sibling
                if cp[el].text is None:
                    cp[el].text = ''
                if el.text:
                    cp[el].text = cp[el].text + el.text
                if el.tail:
                    cp[el].text = cp[el].text + el.tail
                cp[el].remove(el)
            else:
                # left sibling
                if cp[el][i-1].tail is None:
                    cp[el][i-1].tail = ''
                if el.text:
                    cp[el][i-1].tail = cp[el][i-1].tail + el.text
                if el.tail:
                    cp[el][i-1].tail = cp[el][i-1].tail + el.tail
                cp[el].remove(el)
        return
    
    for el in elem[:]:
        clr(el, cp_map(elem))
    return
    
        
#TODO: Rewrite in a functional style
def fill_par(elem):
    """
    Takes an Element object and removes extraneous spaces and line
    breaks from text and tail attributes.
    """
    els = elem.getiterator()
    
    sp = re.compile(' +')
    nl = re.compile('\n+')
    le = re.compile('^ +')
    tr = re.compile(' +$')
    
    for el in els[:]:
        if el.text:
            el.text = re.sub(nl, ' ', el.text)
            el.text = re.sub(sp, ' ', el.text)
            el.text = re.sub(le, '', el.text)
        if el.tail:
            el.tail = re.sub(nl, ' ', el.tail)
            el.tail = re.sub(sp, ' ', el.tail)
            el.tail = re.sub(tr, '', el.tail)
            
    return


######################################################################
#                        ElementTree utilities
######################################################################

def flatten(t):
    """
    Takes an Element and returns a list of strings, extracted from the
    text and tail attributes of the nodes in the tree, in a sensible
    order.
    """
    pre = pre_iter(t)
    post = post_iter(t)

    out = [tag.text for tag in pre]

    i = 0
    for k,n in enumerate(post):
        j = pre.index(n) + k
        i = j if j > i else i + 1
        out.insert(i+1, n.tail)

    return [text for text in out if text]

def pre_iter(t, tag=None):
    """
    Takes an Element, and optionally a tag name, and performs a
    preorder traversal and returns a list of nodes visited ordered
    accordingly.
    """
    return t.getiterator(tag)

def post_iter(t, tag=None):
    """
    Takes an Element object, and optionally a tag name, and performs a
    postorder traversal and returns a list of nodes visited ordered
    accordingly.
    """
    nodes = []
    for node in t._children:
        nodes.extend(post_iter(node, tag))
    if tag == "*":
        tag = None
    if tag is None or t.tag == tag:
        nodes.append(t)
    return nodes

def cp_map(tree):
    """
    Takes an Element object and returns a child:parent dictionary.
    """
    return dict((c, p) for p in tree.getiterator() for c in p)

def match_qname(local, qname):
    """
    Expects a tag name given as the local portion of a QName (e.g.,
    'h1') and matches it against a full QName.
    """
    return re.search('^\{.*\}' + local, qname)

def filter_by_tag(elems, tag):
    """
    Takes a list of Element objects and filters it by a local tag
    name (e.g., 'h1').
    """
    return [el for el in elems if match_qname(tag, el.tag)]

def get_prefix(t):
    """
    Takes an Element object and returns the prefix portion of the
    QName (its tag). For example, if t is XHTML, the QName may be
    'http://www.w3.org/1999/xhtml'. (A typical tag in t would be
    '{http://www.w3.org/1999/xhtml}div').
    """
    p = re.compile('^\{.*\}')
    m = re.search(p, t.tag)
    if m is None:
        return ''
    else:
        return m.group(0)
