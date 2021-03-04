import os
from nltk.tokenize import RegexpTokenizer
from collections import Counter
import nltk
#nltk.download()
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import math
stemmer = PorterStemmer()
corpusroot = './presidential_debates'
df = Counter()                  # document frequency
tf_doc = {}                     # term frequency document
tf_idf_vec = {}                 # TF-IDF vector weights, W_td
d_len = Counter()               # document length

for filename in os.listdir(corpusroot):
    file = open(os.path.join(corpusroot, filename), "r", encoding='UTF-8')
    doc = file.read()
    file.close()
    doc = doc.lower()                   # Converts text to lowercase.
    ############################
    # T O K E N I Z E R
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    tokens = tokenizer.tokenize(doc)
    ############################
    # S T O P  W O R D S - words to ignore
    tokens_without_sw = [word for word in tokens if word not in stopwords.words('english')]     # Deletes stopwords
    ############################
    # S T E M M E R
    stemed_tokens = [stemmer.stem(tokens_without_sw) for tokens_without_sw in tokens_without_sw]# Stems word endings
    ############################
    # C O U N T E R S
    tf = Counter(stemed_tokens)                 # counts each word's frequency from 1 document
    df += Counter(list(set(stemed_tokens)))
    tf_doc[filename] = tf.copy()
    tf.clear()

########################################################
def getidf(term):                                  # idf = log10(N/df)
    if df[term] == 0:                              # df = num of docs term occurs in
        return -1                                  # N = total num of documents
    return math.log10(len(tf_doc)/df[term])

def getweight(filename, term):
    return tf_idf_vec[filename][term]

def getlen(filename):
    return d_len[filename]

def query(input):
    input = input.lower()                   # (6) "Remember to convert it to lower case"
    cosine_similarity = Counter()
    postings_list = {}                      # form of (document d, TF-IDF weight w)
    # 7.1 Here ###############
    ##########################
    #########################
    for token in input:
        if token not in postings_list:
            continue                        # (7.2) If the token doesn't exist in the corpus, ignore it.
                                            # (7.2) If the token doesn't exist in the corpus, ignore it.
    return "None", 0

########################################################
# Tf-Idf vector calulation - not normalized
for filename in tf_doc:
    tf_idf_vec[filename] = Counter()
    length = 0
    for term in tf_doc[filename]:
        tf_idf_vec[filename][term] = (1 + math.log10(tf_doc[filename][term]))*getidf(term)  # Weight formula = (1 + log10(tf_td))(idf)
        length += tf_idf_vec[filename][term]**2       # tf_td = number occurences of term, t, in document, d
    d_len[filename] = math.sqrt(length)

# Td-Idf vectors - normalized
for filename in tf_idf_vec:
    for term in tf_idf_vec[filename]:
        tf_idf_vec[filename][term] = tf_idf_vec[filename][term] / d_len[filename]
        # d_len[filename] = document length
        # tf_idf_vec[filename][term] = weight

########################################################
# T E S T E R S
# g e t l e n
#print("Length of d_len for file: %d" % getlen("2012-10-03.txt"))
#print("Length of d_len for file: %d" % getlen("1960-10-21.txt"))
#print("Length of d_len for file: %d" % getlen("1976-10-22.txt"))
#print("Length of d_len for file: %d" % getlen("2012-10-16.txt"))
# g e t i d f
print("%.12f" % getidf("health"))               # 0.079181246048
print("%.12f" % getidf("agenda"))               # 0.363177902413
print("%.12f" % getidf("vector"))               # -1.000000000000
print("%.12f" % getidf("reason"))               # 0.000000000000
print("%.12f" % getidf("hispan"))               # 0.632023214705
print("%.12f" % getidf("hispanic"))             # -1.000000000000
# g e t w e i g h t
print("\n%.12f" % getweight("2012-10-03.txt","health"))       # 0.008528366190
print("%.12f" % getweight("1960-10-21.txt","reason"))         # 0.000000000000
print("%.12f" % getweight("1976-10-22.txt","agenda"))         # 0.012683891289
print("%.12f" % getweight("2012-10-16.txt","hispan"))         # 0.023489163449
print("%.12f" % getweight("2012-10-16.txt","hispanic"))       # 0.000000000000
# q u e r y
print("\n(%s, %.12f)" % query("health insurance wall street"))        # (2012-10-03.txt, 0.033877975254)
print("(%s, %.12f)" % query("particular constitutional amendment")) # (fetch more, 0.000000000000)
print("(%s, %.12f)" % query("terror attack"))                       # (2004-09-30.txt, 0.026893338131)
print("(%s, %.12f)" % query("vector entropy"))                      # (None, 0.000000000000)
########################################################
# S O U R C E S
# https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
# https://www.w3schools.com/python/python_functions.asp
# https://www.hackerrank.com/challenges/collections-counter/problem
