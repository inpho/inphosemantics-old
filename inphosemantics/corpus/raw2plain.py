import os
import codecs
import unidecode
import re

root = '/var/inphosemantics/data/'

def malaria2plain():

    def clean(raw):

        plain = unidecode.unidecode(raw)
        unknown = re.compile('\\[\\?\\]')
        plain = unknown.sub(' ', plain)

        #Double line breaks to make tokenizer recognize paragraph
        #tokens
        plain = re.sub('\n', '\n\n', raw)

        #Replace hyphens with spaces. Many sentence breaks are
        #indicated by hyphens and not spaces. This is not a standard
        #hyphen.
        plain = re.sub(ur'\xe2', ' ', plain)

        return plain


    corpus_path = 'malaria/complete/corpus/'
    raw_path = os.path.join(root, corpus_path, 'raw')
    plain_path = os.path.join(root, corpus_path, 'plain')
    
    for filename in os.listdir(raw_path):

        raw_filename = os.path.join(raw_path, filename)

        with open(raw_filename, mode='r') as f:

            raw = f.read()

            plain = clean(raw)


        plain_filename = os.path.join(plain_path, filename)
        plain_filename = plain_filename[:-3] + 'inpho.txt'

        with open(plain_filename, mode='w') as f:

            f.write(plain)
