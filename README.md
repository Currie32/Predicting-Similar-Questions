# Predicting-Similar-Questions

There are two slightly different projects included in this folder. 

The "Predicting_Similar_Questions" files relate to a dataset that was hosted by Quora on Kaggle: https://www.kaggle.com/quora/question-pairs-dataset. For this project, I used TfidfVectorizer and Doc2Vec to predict if pairs of questions have the same meaning. 

The "Predicting_Duplicate_Questions_Competition" files relate to a Kaggle competition that is being hosted by Quora on Kaggle: https://www.kaggle.com/c/quora-question-pairs. A few weeks after I completed my dataset project, Quora started this competition. I used a much better strategy for this competition, which has placed me in the top 14% of entrants. I am using 1D-Convolutional Neural Networks and Time Distributed Dense layers to build my model. I'm also using GloVe to pretrain my word vectors. This method, compared to the first, increased the accuracy of my work from 67% to 82%.

To view either of the projects most easily, click on the .ipynb file.

The questions.csv.zip file contains the questions for the first project, i.e. not the competition.

Here are some examples of the pairs of questions asked:

Not duplicates:
- What is the step by step guide to invest in share market in india?
- What is the step by step guide to invest in share market?

Not duplicates:
- What is the story of Kohinoor (Koh-i-Noor) Diamond?
- What would happen if the Indian government stole the Kohinoor (Koh-i-Noor) diamond back?

Are duplicates:
- Astrology: I am a Capricorn Sun Cap moon and cap rising...what does that say about me?
- I'm a triple Capricorn (Sun, Moon and ascendant in Capricorn) What does this say about me?
