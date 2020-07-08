import numpy as np
import pandas as pd
import nltk
import spacy
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation

def freq_words(x, terms =30):
    text = ' '.join([text for text in x])
    all_words = text.split()
    fdist = nltk.FreqDist(all_words)
    words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})
    d = words_df.nlargest(columns = 'count', n = terms)
    # plt.figure(figsize = (20,5))
    # ax = sns.barplot(data = d, x= 'word',y='count')
    # ax.set(ylabel = 'Count')
    # plt.show()

pd.set_option("display.max_colwidth", 200)
#print(text)
filenames = os.listdir(r"D:\NLP\\AnalyticsVidhya-NLP\\Handouts_v4\\Project - Categorization of Sports Articles\\sports_notebook_and_data\\sports_articles")

articles =[]
for f in filenames:
    file = open(r'D:\NLP\\AnalyticsVidhya-NLP\\Handouts_v4\\Project - Categorization of Sports Articles\\sports_notebook_and_data\\sports_articles\\'+f, mode = 'rt', encoding='utf-8')
    text = file.read()
    file.close()
    articles.append(text)
#print(len(articles))

#cleaning text
clean_articles =[]
for i in articles:
    clean_articles.append(i.replace("\n"," ").replace("\'", " "))

clean_articles = [re.sub("[^a-zA-Z]", " ",x) for x in clean_articles]
clean_articles =[' '.join([w for w in x.split() if len(w) > 1])for x in clean_articles]
clean_articles = [x.lower() for x in clean_articles]
nlp = spacy.load('en_core_web_sm')
clean_articles = [' '.join([token.lemma_ for token in nlp(x)]) for x in clean_articles]
clean_articles = [' '.join([w for w in x.split() if nlp.vocab[w].is_stop==False]) for x in clean_articles]
clean_articles = [re.sub('-PRON-',  '', i) for i in clean_articles]
freq_words(clean_articles)  

#document- term matrix where rows - no of articles and col - unique words using SVD
# vectorizer = TfidfVectorizer()
# vectorizer = TfidfVectorizer(max_features = 1000,
                            # min_df = 5,
                            # max_df = 0.9)
    
# x = vectorizer.fit_transform(clean_articles)
# svd_model = TruncatedSVD(n_components=4, random_state=12, n_iter=100)
# svd_model.fit(x)

# # get column names of document term matrix
# terms = vectorizer.get_feature_names()

# for i, comp in enumerate(svd_model.components_):
    # terms_comp = zip(terms, comp)
    # sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:50]
    
    # print("Topic "+str(i)+": ")
    # topics = []
    # for t in sorted_terms:
        # topics.append(t[0])
    
    # print(topics)
    # print('\n')

# lsa_topic_matrix = svd_model.transform(x)

# print(articles[9])

# print(np.argmax(lsa_topic_matrix[9]))

#document- term matrix where rows - no of articles and col - unique words using LDA
vectorizer = TfidfVectorizer()
vectorizer = TfidfVectorizer(max_features = 1000,
                            min_df = 5,
                            max_df = 0.9)
    
x = vectorizer.fit_transform(clean_articles)
lda_model = LatentDirichletAllocation(n_components=4, max_iter=100, random_state=12)
lda_model.fit(x)

# get column names of document term matrix
terms = vectorizer.get_feature_names()

for i, comp in enumerate(lda_model.components_):
    terms_comp = zip(terms, comp)
    sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:50]
    
    print("Topic "+str(i)+": ")
    topics = []
    for t in sorted_terms:
        topics.append(t[0])
    
    print(topics)
    print('\n')

lda_topic_matrix = lda_model.transform(x)

print(articles[9])

print(np.argmax(lda_topic_matrix[9]))


