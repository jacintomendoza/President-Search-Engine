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
df = Counter()
tf_doc = {}

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
    tokens_without_sw = [word for word in tokens if word not in stopwords.words('english')]
    #print(tokens_without_sw)
    ############################
    # S T E M M E R
    stemed_tokens = [stemmer.stem(tokens_without_sw) for tokens_without_sw in tokens_without_sw]
    #print(stemed_tokens)
    ############################
    # C O U N T E R S
    tf = Counter(stemed_tokens)
    df += Counter(list(set(stemed_tokens)))
    tf_doc[filename] = tf.copy()
    tf.clear()

def cal_tf_idf_weight():
    print(stemmer.stem('ambassador'))

def getidf(term):
    if df[term] == 0:
        return -1
    return math.log10(len(tf_doc)/df[term])        # idf = log10(N/df)

print("%.12f" % getidf("health"))               # 0.079181246048
print("%.12f" % getidf("agenda"))               # 0.363177902413
print("%.12f" % getidf("vector"))               # -1.000000000000
print("%.12f" % getidf("reason"))               # 0.000000000000
print("%.12f" % getidf("hispan"))               # 0.632023214705
print("%.12f" % getidf("hispanic"))             # -1.000000000000


############################
# S O U R C E S
# https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
# https://www.w3schools.com/python/python_functions.asp
# https://www.hackerrank.com/challenges/collections-counter/problem
