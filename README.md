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
1. [ford.csv](https://github.com/Owen-Choh/ntu-sc1015-mini-project/blob/main/ford.csv) - original dataset
2. [cleaned_cardata.csv](https://github.com/Owen-Choh/ntu-sc1015-mini-project/blob/main/cleaned_cardata.csv) - cleaned dataset for analysis
3. [Data_Cleaning_And_Preparation.ipynb](https://github.com/Owen-Choh/ntu-sc1015-mini-project/blob/main/Data_Cleaning_And_Preparation.ipynb)
4. [Exploratory_Data_Analysis.ipynb](https://github.com/Owen-Choh/ntu-sc1015-mini-project/blob/main/Exploratory_Data_Analysis.ipynb)
5. [Random Forest.ipynb](https://github.com/Owen-Choh/ntu-sc1015-mini-project/blob/main/Random%20Forest.ipynb)
6. [Decision Tree Classifier with K Fold.ipynb](https://github.com/Owen-Choh/ntu-sc1015-mini-project/blob/main/Decision%20Tree%20Classifier%20with%20K%20Fold.ipynb)
7. [Gradient Boosting Regressor.ipynb](https://github.com/Owen-Choh/ntu-sc1015-mini-project/blob/main/Gradient%20Boosting%20Regressor.ipynb)
8. SC1015 Project Slides.pdf - presentation slides for our project

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

    a. One-Hot-Encoded categorical variables and build a model using Random Forest Classifier
       - A R^2 Score of 0.719 is achieved

    b. Used the feature importance of the variables used by the model to retrain the model to see if the accuracy when using only a few variables
       - A similar R^2 Score of 0.681 is achieved
    
    c. Did Synthetic Minority Oversampling Technique (SMOTE) to rebalance the car model distribution
       - The R^2 Score decreased to 0.485

*2. Decision Tree Classifier with K Fold*

    a. Get the 6 categorical variables and convert them into indicator variables using pandas.get_dummies
    
    b. Build a Logistic Regression model based on the predictors and print out the rank of importance of predictors using Recursive Feature Elimination (RFE)
       - The accuracy on test set is 0.84
       - 'OverTime' and 'Gender' contribute more to the predicton of attrition than the rest of the predictors
    
    c. Examine the classification report and analyze the model based on 'precision' and 'recall'
       - Get the conclusion that the model is not effective even with high accuracy
       - The highly imbalanced distribution of employees when categorizing by attrition is the major factor for the high accuracy
    
    d. Get the confusion matrix to back up our conclusion

*3. Gradient Boosting Regressor*

    a. Import PyTorch library and select major numeric attributes
        - The attributes include 'MonthlyIncome', 'DistanceFromHome' and 'YearsInCurrentRole'

    b. Build a multilayer perceptron model for multi-label classification
    
    c. Train the model with CrossEntropyLoss as loss function and SGD as optimizer.
        - The loss of the model reduced significantly after training through 3 epochs
        - The accuracy on test cases is 85.7%


### Conclusion

*Machine Learning Comparisons*
- Random Forest suggests that the Mileage, Year and Model variables are most useful in predicting car price
- Decision Tree Classifier with K Fold is not recommended as a technique as most categorical variables are irrelevant in determining attrition
- Gradient Boosting Regressor is highly useful in effectively predicting attrition
- The random forest achieved an R-squared value of 0.719. The KFold achieved a lower R-squared value of 0.636. The gradient boosting regressor achieved the highest R-squared value of 0.918 which indicates the best performer among the three.

*Data Driven Insights*
- Models can provide information on the impact of the car features on car prices. 
- For example, the gradient boosting regressor model can provide a feature importance chart, displaying how each feature contributes to the prediction of car prices.
- This will allow buyers, car manufacturers and dealerships to understand how each feature impacts the price of the car and make informed decision.
- For recommendations on how to improve, it is important to note that these models alone does not accurately solve the original problem of predicting car prices. 
- To ensure that the model is consistent, we need to test the model on new set of data which is also known as "out of sample testing"

### What we have learnt from this project?
- Using scikit learn OneHotEncoder to convert catrgorical variables into a more useable form
- Random Forrest Model
- Decision Tree Classifier Model
- K-Fold technique
- Gradient Boosting Regressor Model
- Justify which variable is best to predict car price

### References
1. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
2. https://scikit-learn.org/stable/modules/ensemble.html#forest
3. https://scikit-learn.org/stable/modules/model_evaluation.html
4. https://stephenallwright.com/cross_val_score-sklearn/#:~:text=Can%20I%20train%20my%20model%20using%20cross_val_score%3F
5. https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
6. https://imbalanced-learn.org/dev/references/generated/imblearn.over_sampling.SMOTENC.html
7. https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
8. https://www.ibm.com/topics/random-forest

