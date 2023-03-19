# Car Price Prediction

## About

This is the mini project for NTU-SC1015 (Introduction to Data Science and Artificial Intelligence).

(maybe write short intro sentence why we chose to do this problem)

## Problem definition

- What are the features differences do cars or different prices have?
- Are we able to predict car price through the car models alone?
- Are we able to predict through comparing the features that the car has? (Engine Size, Fuel type, Year etc) If so, what are the most significant features? 

## Dataset used
The dataset used for this project is retrieved from [here](TODO)

For the cleaned dataset, please download it from [here](TODO)

## Presentation
The presentation video can be found [here](TODO)

## Brief process walkthrough (In order)

1. [Data Preparation & Cleaning](<Data Preparation & Cleaning.ipynb>)
    - Data cleaning
      - Salvage empty rows
      - Removal of numbers & symbols (Excluding punctuation)
      - Removal of stopwords
      - Word stemming
      - Drop empty rows after all cleaning steps (Dirty data)
    - Data generation
      - Word count & char count
      - Stopwords count
      - Sentiment
      - Emotions
      - Parts-of-speech (POS)

2. [Exploratory Data Analysis](<Exploratory Data Analysis & Visualization.ipynb>)
   - Class analysis
   - Wordcount & charcount analysis
   - Author analysis
   - Corpus analysis
   - N-gram analysis
   - Sentiment & emotion analysis
   - Parts-of-speech (POS) analysis
   - Correlation analysis

3. [Model Training Attempt 1 & 2](<Model Training Attempt 1 & 2.ipynb>)
   - Attempt 1 (Decision Tree)
     - Train with top 5 predictors
     - Average accuracy: 0.73
     - Model evaluation
       - Plotting decision tree
       - Confusion matrix
   - Attempt 2 (Random Forest)
     - Train with top 5 predictors
     - Average accuracy: 0.75
     - Model evaluation
       - Confusion matrix
       - Grid search hyper-parameter tuning

5. [Model Training Attempt 3](<Model Training Attempt 3.ipynb>)
   - TF-IDF analysis 
   - Attempt 3 (Logistic Regression)
     - Train with only title
     - Average accuracy: 0.93
     - Model evaluation
       - Confusion Matrix
       - Recall, precision, F1 score
       - Receiver Operating Characteristic (ROC) Curve & Area Under Curve (AUC)
       - Model weights

## Conclusion
- Surprisingly, polarity & emotions does not have a strong relation to fake news, therefore, not a good indicator of fake news.
- Instead, indicators such as title wordcount, title adjective count, and text stopwords count are the best indicators to fake news. 
- Based on attempt 3, detection of fake news using title is sufficient. However, for the best results, author & title are required.
- Out of all the 3 models we implemented, decision tree performed the worst while logistic regression performed the best.
- Based on findings, we can suggest that from a reader's perspective in identifying fake news, author is a quick & credible identifier, and the title could further support a reader's attempt in identifying fake news.

## Key learning points
- NLP & Text processing techniques
  - Removal of stopwords
  - Removal of noisy data (Numbers & symbols)
  - Word stemming
  - Sentiment & emotion analysis
  - Parts-of-speech (POS)
  - N-gram analysis (Bi-grams)
- Logistic regression model training & evaluation
- Converting unstructured text into text vectors using TF-IDF scoring metric
- Using Python libraries with pre-trained models to predict and generate emotions & sentiment
- Plotting correlation matrix with categorical data


## Contributors

1. @TODO (TODO) - Data Preparation & Cleaning, Exploratory Data Analysis, Model Training Attempt 3
2. @TODO (TODO) - Model Training Attempt 1 & 2, Presentation Slides, Presenter
3. @TODO (TODO) - Model Training Attempt 1 & 2, Presentation Slides, Presenter

## References

- https://data-flair.training/blogs/advanced-python-project-detecting-fake-news/comment-page-2/
