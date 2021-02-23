import os
corpusroot = './presidential_debates'
for filename in os.listdir(corpusroot):
    file = open(os.path.join(corpusroot, filename), "r", encoding='UTF-8')
    doc = file.read()
    file.close()
    doc = doc.lower()   # Converts text to lowercase.
#print(doc)

############################

from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
tokens = tokenizer.tokenize(doc)
print(tokens)

############################
# S T O P  W O R D S - words to ignore

#import nltk
#nltk.download()

from nltk.corpus import stopwords
print(stopwords.words('english'))
print(sorted(stopwords.words('english')))

############################
# S T E M M E R
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
print(stemmer.stem('studying'))
print(stemmer.stem('vector'))
print(stemmer.stem('entropy'))
print(stemmer.stem('hispanic'))
print(stemmer.stem('ambassador'))

############################
# S O U R C E S
#https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
