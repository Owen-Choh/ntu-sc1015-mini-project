# SC1015: Data Science Mini Project 

School of Computer Science and Engineering \
Nanyang Technological University \
Lab: Z136 \
Group : 4

Members: 
1. Chong Choy Jun ([@TODO](https://github.com/TODO))
2. Chew Di Heng ([@TODO](https://github.com/TODO))
3. Choh Lit Han Owen ([@Owen-Choh](https://github.com/Owen-Choh))

---
### Description:
This repository contains all the Jupyter Notebooks, datasets, images, video presentations, and the source materials/references we have used and created as part of the Mini Project for SC1015: Introduction to Data Science and AI. 

This README briefly highlights what we have accomplished in this project. If you want a more detailed explanation of things, please refer to the Jupyter Notebooks in this repository. They contain more in-depth descriptions and smaller details which are not mentioned here in the README. For convenience, we have divided the notebooks into 5 parts which broadly relate to the 5 main sections of this project.

---
### Table of Contents:
1. [Problem Formulation](#1-Problem-Formlation)
2. [Data Preparation and Cleaning](#2-Data-Preparation-and-Cleaning)
3. [Exploratory Data Analysis](#3-Exploratory-Data-Analysis)
4. [Dimensionality Reduction](#4-Dimensionality-Reduction)
5. [Clustering](#5-Clustering)
6. [Data Driven Insights and Conclusion](#6-Data-Driven-Insights-and-Conclusion)
7. [References](#7-References)
---
### 1. [Problem Formulation](Part_1_Data_Prep_Cleaning.ipynb)

**Our Dataset:** [Car Prices](TODO) \
**Our Question:** How Much Does Features Affect Car Prices? 


### 2. [Data Preparation and Cleaning](Part_1_Data_Prep_Cleaning.ipynb)
In this section of the project, we prepped and cleaned the dataset to help us analyze our data better and also to help us use our data for the purposes of machine learning in the later sections. 

We performed the following:
1. **Preliminary Feature Selection:** `8` relevant variables out of `61` were selected.
2. **Dropping `NaN`s**: All the `NaN` values in these `8` variables were dropped. 
3. **Splitting Dataset in Two:** The `8` variables were then split in 2 DataFrames. One with `6` variables relating to conventionality and the other with `2` relating to success. 
4. **Encoding Categorical Variables:** The categorical variables in both the DataFrames were encoded appropriately. 


### 3. [Exploratory Data Analysis](Part_2_EDA.ipynb)
Then, we explored each of our two DataFrames further using Exploratory Data Analysis to answer questions like are there any patterns we are noticing? What do our success variables look like? What about the conventionality variables? Are there any underlying relationships between them? Can we make any inferences for our question at this stage? 

To achieve this we did the following:
1. **Explored `ConvertedComp`**: This variable is the annual compensation in USD (a.k.a Salary). Median of around $54k was seen. A lot of outliers with high salaries were present.
2. **Explored `JobSat`:** This variable is the job satisfaction (`0-4` scale). Most frequent ratings were `2` and `4`. The mean rating was at `2.3`.
3. **Explored Relationships Between `JobSat` and `ConvertedComp`:** Weak correlation was seen between `JobSat` and `ConvertedComp`.
4. **Explored Variables Related to Conventionality:** Studied which options in the `6` variables were more frequently selected by respondents. 

For further findings and explanations, please refer to the Jupyter Notebook on EDA.


### 4. [Dimensionality Reduction](Part_3_Dimension_Reduction.ipynb)
Our DataFrame with `6` variables after encoding was converted to a DataFrame with `94` which is a very high dimensional data. 

This meant a few problems (curse of dimensionality):
1. It would probably not result in nicely formulated clusters.
2. High dimensional data is difficult to work with because of space and time increases when running algorithms on them.
3. High dimensional data is difficult to visualize.

So, **Multiple Correspondence Analysis (MCA)** was used to reduce these dimensions. The reason we chose MCA was that the general convention with dimensionality reduction is Principal Component Analysis (PCA), however it does not work well with categorical data which is what we have. MCA works well with multiple columns of categorical data. 

Using MCA, the dimensions were reduced from `94` columns to just `42`!


### 5. [Clustering](Part_4_Clustering.ipynb)

With these `42` columns, we then performed clustering. We chose the **Hierarchical Density-Based Spatial Clustering of Applications with Noise (HDBCAN)**. 

The reasons for this are:
1. 

More details on HDBSCAN and its parameters are presented in the Jupyter Notebook on Clustering.

In this section, we performed the following:
1. 

Our final clustering resulted in a total of `3` clusters and `6206` outliers (out of `19362` total points).


### 6. [Data Driven Insights and Conclusion](Part_5_Data_Driven_Insights.ipynb)
Here, we re-combined our variables related to success and the clustered variables related to conventionality to see if there are any differences between outliers and non-outliers. We performed a comparative Exploratory Data Analysis on the outliers vs. non-outliers to see if we can infer anything from the similarities and differences. 

In this section, we also looked at the characteristics of the individuals in our `3` clusters using the variables related to conventionality. The findings have been presented in the Jupyter Notebook on Data Driven Insights. 

Most notably, however, we found that there were no difference in the distribution of the Salary or the Job Satisfaction among Outliers and Non-outliers (Conventional individuals and non-conventional individuals). So, we concluded that unconventionality might NOT be an indicator of success. 

### 7. References
1. [REFERENCE 1](TODO) 
