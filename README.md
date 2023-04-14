# SC1015 Mini-Project
For our mini project in the Introduction to Data Science and Artificial Intelligence module (SC1015), we performed analysis on the Ford Car Price [dataset](https://www.kaggle.com/datasets/adhurimquku/ford-car-price-prediction) from Kaggle. 

# Car Price Prediction

## About

This is the mini project for NTU-SC1015 (Introduction to Data Science and Artificial Intelligence).

The goal of this data science project is to develop a predictive model that can accurately forecast Ford car sales based on historical
sales data and relevant features. The model will be used to help Ford Motor Company optimize their sales strategies and make data-driven decision
to improve sales performance 

## Problem definition

- What are the features differences do cars or different prices have?
- Are we able to predict car price through a subset of the car features alone? (Engine Size, Fuel type, Year etc) If so, what are the most significant features? 

## Dataset used
The dataset used for this project is retrieved from [here](https://github.com/Owen-Choh/ntu-sc1015-mini-project/blob/main/ford.csv)

For the cleaned dataset, please download it from [here](https://github.com/Owen-Choh/ntu-sc1015-mini-project/blob/main/cleaned_cardata.csv)

## Presentation
The presentation video can be found [here](TODO)

### Members (Z136_Team 4)
1. Chong Choy Jun 
2. Chew Di Heng
3. Choh Lit Han Owen

### Files Included
1. ford.csv - original dataset
2. cleaned_cardata.csv - cleaned dataset for analysis
3. 
4.
5.
6. SC1015 Project Slides.pdf - presentation slides for our project
7. SC1015 Mini Project.ipynb 
    - Cleaning and preparation
    - Basic visualization
    - Exploratory data analysis
    - Machine learning: Random Forest, Logistic Regression, Neural Network, kFold  

### Notebook Descriptions
#### Data Cleaning And Preparation
   a. Removing insignificant features: 'Tax', 'mpg'
   
   b. Check for missing values within the dataset

   c. Converting 'year' column to a categorical variable
   
   d. Save cleaned dataset to new csv file

#### Exploratory Data Analysis
##### Part 1: Analysis of Numeric Variables
   a. Used box plots, histograms, and violin plots in visualizing numeric variables
   
   b. Checked for number of outliers in numeric variables: 'mileage' 878, 'engineSize' 190 
   
   c. Used Heatmap and pairplot to visualise the correlation of all numeric variables and identify relationships and patterns within the data

##### Part 2: Analysis of Categorical data
   a. Used describe function to get an idea of how the categorical data looks like
   
   b. Used catplot to visualise the distribution of each categorical variable: 'model', 'year', 'transmission', 'fuelType'

##### Part 3: Relationship with price for numeric data
   a. Used jointplot to visualise each numeric variable against the price and identify patterns within the data
   
   b. Used boxplot to visualise each categorical variable against the price and identify relationships between the data


#### Machine Learning
*1. Random Forest*

    a. Extract numeric variables that have high correlation with attrition as predictors and build a model using Random Forest Classifier
       - A accuracy of 84.5% is achieved

    b. Do hyperparemeter tuning using Random Search and plot one of the trees from Random Forest
       - The accuracy is improved to 84.7%

*2. Logistic Regression*

    a. Get the 6 categorical variables and convert them into indicator variables using pandas.get_dummies
    
    b. Build a Logistic Regression model based on the predictors and print out the rank of importance of predictors using Recursive Feature Elimination (RFE)
       - The accuracy on test set is 0.84
       - 'OverTime' and 'Gender' contribute more to the predicton of attrition than the rest of the predictors
    
    c. Examine the classification report and analyze the model based on 'precision' and 'recall'
       - Get the conclusion that the model is not effective even with high accuracy
       - The highly imbalanced distribution of employees when categorizing by attrition is the major factor for the high accuracy
    
    d. Get the confusion matrix to back up our conclusion

*3. Neural Network*

    a. Import PyTorch library and select major numeric attributes
        - The attributes include 'MonthlyIncome', 'DistanceFromHome' and 'YearsInCurrentRole'

    b. Build a multilayer perceptron model for multi-label classification
    
    c. Train the model with CrossEntropyLoss as loss function and SGD as optimizer.
        - The loss of the model reduced significantly after training through 3 epochs
        - The accuracy on test cases is 85.7%


### Conclusion

*Machine Learning Comparisons*
- Random Forest suggests that numeric variables with relatively high correlation with attrition are useful in predicting attrition
- Logistic Regression is not recommended as a technique as most categorical variables are irrelevant in determining attrition
- Neural Network is highly useful in effectively predicting attrition

*Data Driven Insights*
- Common profile of employees who quit: low salary, lives far away from office, low chance of career progression/lack of opportunities
- Actions for IBM: increase salary incentives, enhance effective employee assessments, change up roles in senior management

### What we have learnt from this project?
- Using pandas.get_dummies to convert catrgorical variables into indicator variables
- Logistic Regression model 
- Justify the suitability of a model based on readings from classification report
- Neural Network model

### References
1. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
2. https://scikit-learn.org/stable/modules/ensemble.html#forest
3. https://scikit-learn.org/stable/modules/model_evaluation.html
4. https://stephenallwright.com/cross_val_score-sklearn/#:~:text=Can%20I%20train%20my%20model%20using%20cross_val_score%3F
5. https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
6. https://imbalanced-learn.org/dev/references/generated/imblearn.over_sampling.SMOTENC.html
7. https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html


## Brief process walkthrough (In order)

1. [Data Preparation & Cleaning](<Data_Cleaning_And_Preparation.ipynb>)
    - Feature Selection
      - Selected columns: 'model', 'year', 'price', 'transmission', 'mileage', 'fuelType', 'engineSize'
    - Data cleaning
      - Checking for missing values (Dirty data)
      - Converting "year" from int64 to object type
      - Splitting dataset into numerical and categorical variables

2. [Exploratory Data Analysis](<Exploratory_Data_Analysis.ipynb>)
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

- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- https://scikit-learn.org/stable/modules/ensemble.html#forest
- https://scikit-learn.org/stable/modules/model_evaluation.html
- https://stephenallwright.com/cross_val_score-sklearn/#:~:text=Can%20I%20train%20my%20model%20using%20cross_val_score%3F
- https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
- https://imbalanced-learn.org/dev/references/generated/imblearn.over_sampling.SMOTENC.html
