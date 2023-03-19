# Car Price Prediction

## About

This is the mini project for NTU-SC1015 (Introduction to Data Science and Artificial Intelligence).

(maybe write short intro sentence why we chose to do this problem)

## Problem definition

- What are the features differences do cars or different prices have?
- Are we able to predict car price through a subset of the car features alone? (Engine Size, Fuel type, Year etc) If so, what are the most significant features? 

## Dataset used
The dataset used for this project is retrieved from [here](https://github.com/Owen-Choh/ntu-sc1015-mini-project/blob/main/ford.csv)

For the cleaned dataset, please download it from [here](https://github.com/Owen-Choh/ntu-sc1015-mini-project/blob/main/cleaned_cardata.csv)

## Presentation
The presentation video can be found [here](TODO)

## Brief process walkthrough (In order)

1. [Data Preparation & Cleaning](<Data Preparation & Cleaning.ipynb>)
    - Feature Selection
      - Selected columns: 'model', 'year', 'price', 'transmission', 'mileage', 'fuelType', 'engineSize'
    - Data cleaning
      - Checking for missing values (Dirty data)
      - Converting "year" from int64 to object type
      - Splitting dataset into numerical and categorical variables

2. [Exploratory Data Analysis](<Exploratory Data Analysis & Visualization.ipynb>)
    - Price distribution analysis
    - Numerical Variable analysis
    - Correlation analysis
    - Categorical Variable analysis
    - Price with Numerical data analysis
    - Price with Categorical data analysis

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
