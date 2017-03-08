
# coding: utf-8

# # Predicting Similar Questions

# The objective of this analysis is to use different Natural Language Processing methods to predict if pairs of questions have the same meaning. The data is from Quora and hosted on Kaggle: https://www.kaggle.com/quora/question-pairs-dataset. The sections of this analysis include:
# - Transforming the text
# - Method 1: TfidfVectorizer 
# - Method 2: Doc2Vec

# In[2]:

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


# In[33]:

df = pd.read_csv("questions.csv")


# In[34]:

df.head(10)


# In[35]:

df.isnull().sum()


# In[36]:

# Drop nulls since they can't be used by either method.
df = df.dropna()


# In[37]:

df.isnull().sum()


# In[38]:

df.is_duplicate.value_counts()


# In[39]:

255043/len(df)


# Although accuracy won't be as good of a performance metric as F1, it's still good to establish some sort of a baseline. In this case, 63.1% will be our baseline for accuracy. 
# 
# Note: 50% could also be used as the baseline. This would represent a random guess, rather than always picking the most common answer.

# In[40]:

# Take a look at some of the question pairs.
print("Not duplicate:")
print(df.question1[0])
print(df.question2[0])
print()
print("Not duplicate:")
print(df.question1[1])
print(df.question2[1])
print()
print("Is duplicate:")
print(df.question1[5])
print(df.question2[5])


# This task looks like it will be a little difficult since the first pair of questions have very similar wordings but different meanings, and the third pair have less similar wordings but the same meaning.

# In[41]:

def review_to_wordlist(review, remove_stopwords=True):
    # Clean the text, with the option to remove stopwords.
    
    # Convert words to lower case and split them
    words = review.lower().split()

    # Optionally remove stop words (true by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    
    review_text = " ".join(words)

    # Clean the text
    review_text = re.sub(r"[^A-Za-z0-9(),!.?\'\`]", " ", review_text)
    review_text = re.sub(r"\'s", " 's ", review_text)
    review_text = re.sub(r"\'ve", " 've ", review_text)
    review_text = re.sub(r"n\'t", " 't ", review_text)
    review_text = re.sub(r"\'re", " 're ", review_text)
    review_text = re.sub(r"\'d", " 'd ", review_text)
    review_text = re.sub(r"\'ll", " 'll ", review_text)
    review_text = re.sub(r",", " ", review_text)
    review_text = re.sub(r"\.", " ", review_text)
    review_text = re.sub(r"!", " ", review_text)
    review_text = re.sub(r"\(", " ( ", review_text)
    review_text = re.sub(r"\)", " ) ", review_text)
    review_text = re.sub(r"\?", " ", review_text)
    review_text = re.sub(r"\s{2,}", " ", review_text)
    
    words = review_text.split()
    
    # Shorten words to their stems
    stemmer = SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in words]
    
    review_text = " ".join(stemmed_words)
    
    # Return a list of words
    return(review_text)


# In[42]:

def process_questions(question_list, questions, question_list_name):
# function to transform questions and display progress
    for question in questions:
        question_list.append(review_to_wordlist(question))
        if len(question_list) % 100000 == 0:
            progress = len(question_list)/len(df) * 100
            print("{} is {}% complete.".format(question_list_name, round(progress, 1)))


# In[43]:

questions1 = []     
process_questions(questions1, df.question1, "questions1")
print()
questions2 = []     
process_questions(questions2, df.question2, "questions2")


# In[44]:

# Take a look at some of the processed questions.
for i in range(5):
    print(questions1[i])
    print(questions2[i])
    print()


# In[45]:

# Stores the indices of unusable questions
invalid_questions = []
for i in range(len(questions1)):
    # questions need to contain a vowel (which should be part of a full word) to be valid
    if not re.search('[aeiouyAEIOUY]', questions1[i]) or not re.search('[aeiouyAEIOUY]', questions2[i]):
    # Need to subtract 'len(invalid_questions)' to adjust for the changing index values as questions are removed.
        invalid_questions.append(i-len(invalid_questions))
print(len(invalid_questions))


# In[46]:

# list of invalid questions
invalid_questions


# These questions look pretty unusable, so it should be okay to remove them. Plus, we are only removing less than 0.09% of all of the questions.

# In[47]:

# 1 = duplciate, 0 = not duplicate
print(df.is_duplicate[invalid_questions[0]])
print(questions1[invalid_questions[0]])
print(questions2[invalid_questions[0]])
print()
print(df.is_duplicate[invalid_questions[1]+1])
print(questions1[invalid_questions[1]+1])
print(questions2[invalid_questions[1]+1])
print()
print(df.is_duplicate[invalid_questions[2]+2])
print(questions1[invalid_questions[2]+2])
print(questions2[invalid_questions[2]+2])


# In[48]:

# Remove the invalid questions
for index in invalid_questions:
    df = df[df.id != index]
    questions1.pop(index)
    questions2.pop(index)

# These questions are also unusable, but were not detected initially.
# They were found when the function 'cosine_sim' stopped due to an error.
unexpected_invalid_questions = [36460,42273,65937,304867,306828,353918] 
for index in unexpected_invalid_questions:
    df = df[df.id != index]
    questions1.pop(index)
    questions2.pop(index)


# In[49]:

# Use TfidfVectorizer() to transform the questions into vectors,
# then compute their cosine similarity.
vectorizer = TfidfVectorizer()
def cosine_sim(text1, text2):
    tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0,1]


# In[69]:

Tfidf_scores = []
for i in range(len(questions1)):
    score = cosine_sim(questions1[i], questions2[i])
    Tfidf_scores.append(score)
    if i % 40000 == 0:
        progress = i/len(questions1) * 100
        print("Similarity Scores is {}% complete.".format(round(progress,2)))


# In[70]:

# Plot the scores
plt.figure(figsize=(12,4))
plt.hist(Tfidf_scores, bins = 200)
plt.xlim(0,1)
plt.show()


# In[96]:

# Function to report the quality of the model
def performance_report(value, score_list):
    # the value (0-1) is the cosine similarity score to determine if a pair of questions
    # have the same meaning or not.
    scores = []
    for score in score_list:
        if score >= value:
            scores.append(1)
        else:
            scores.append(0)

    accuracy = accuracy_score(df.is_duplicate, scores) * 100
    print("Accuracy score is {}%.".format(round(accuracy),1))
    print()
    print("Confusion Matrix:")
    print(confusion_matrix(df.is_duplicate, scores))
    print()
    print("Classification Report:")
    print(classification_report(df.is_duplicate, scores))


# In[97]:

performance_report(0.51, Tfidf_scores)


# Using a threshold of 0.51 for the cosine similarity maximizes both the f1-score and accuracy. It's good to see that we are scoring better than the baseline value of 63.1% accuracy. I'm not too surprised that we didn't score much above the baseline accuracy, given the difficulty of this task.

# ## Method 2: Doc2Vec

# In[50]:

# Reset index to match the index values of questions1 and questions2
df = df.reset_index()


# In[51]:

# Contains the processed questions for Doc2Vec
questions_labeled = []

for i in range(len(questions1)):
    # Question strings need to be separated into words
    # Each question needs a unique label
    questions_labeled.append(LabeledSentence(questions1[i].split(), df[df.index == i].qid1))
    questions_labeled.append(LabeledSentence(questions2[i].split(), df[df.index == i].qid2))
    if i % 40000 == 0:
        progress = i/len(questions1) * 100
        print("{}% complete".format(round(progress, 2)))


# In[52]:

# Split questions for computing similarity and determining the lengths of the questions.
questions1_split = []
for question in questions1:
    questions1_split.append(question.split())
    
questions2_split = []
for question in questions2:
    questions2_split.append(question.split())


# In[53]:

# Determine the length of questions to select more optimal parameters.
lengths = []
for i in range(len(questions1_split)):
    lengths.append(len(questions1_split[i]))
    lengths.append(len(questions2_split[i]))
lengths = pd.DataFrame(lengths, columns=["count"])


# In[54]:

lengths['count'].describe()


# In[55]:

# 99% of the questions include 18 or fewer words.
np.percentile(lengths['count'], 99)


# In[56]:

# Build the model
model = Doc2Vec(dm = 1, min_count=1, window=10, size=150, sample=1e-4, negative=10)
model.build_vocab(questions_labeled)


# In[57]:

# Train the model
for epoch in range(20):
    model.train(questions_labeled)
    print("Epoch #{} is complete.".format(epoch+1))


# In[58]:

# Check a few terms to ensure the model was trained properly.
model.most_similar('good')


# In[59]:

model.most_similar('peopl')


# In[60]:

model.most_similar('book')


# These words have appropriate similar words, so I am pleased with the training.

# In[61]:

doc2vec_scores = []
for i in range(len(questions1_split)):
    # n_similarity computes the cosine similarity in Doc2Vec
    score = model.n_similarity(questions1_split[i],questions2_split[i])
    doc2vec_scores.append(score)
    if i % 100000 == 0:
        progress = i/len(questions1_split) * 100
        print("{}% complete.".format(round(progress,2)))


# In[62]:

# Plot the scores
plt.figure(figsize=(12,4))
plt.hist(doc2vec_scores, bins = 200)
plt.xlim(0,1)
plt.show()


# It's interesting to see how Doc2Vec computes the pairs of questions to be more similar than TfidfVectorizer.

# In[68]:

performance_report(0.945, doc2vec_scores)


# Using 0.945 as our threshold, we are able to score slightly higher with the Doc2Vec method. Accuracy is 2 percentage points higher and the f1-score increased by 0.01. Much like with TfidfVectorizer, it would have been nice to score higher, but this is by no means an easy challenge. Nonetheless, I hope that you have learned something from reading this and enjoyed this project as much as I did.

# In[ ]:



