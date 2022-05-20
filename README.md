# House Price Prediction

The objective of this project is to predict the houses sale. This project leverages the supervised/ unsupervised ML techniques, utilizes different exploratory data analysis and processing techniques, uses different features engineer methods, and then select the appropriate machine learning models. 

The general approach used for this project follows the standard machine learning life cycle that includes project planning phase, data preparation, exploratory data analysis, features engineering, modeling, and predictions as shown in Figure 1. Each step in the diagram shown in Figure 1 will be described in the next subsections.
The next step is to collect and consolidate the data, wrangle it, and conduct exploratory data analysis. Once the data is ready to go, the next steps are to select ML models, split the dataset, train the selected models with training data, fine-tune the models, and evaluate the models using the testing data based on the pre-determined success metrics. The last step is to select the best model and productionize it. 

![image](https://user-images.githubusercontent.com/89004966/167239297-c72af1df-e1b8-4fa3-8166-10c90a9d540c.png)

Figure 1: The machine learning project life cycle.


## Dataset

The Ames housing dataset  was used to predict house prices. This dataset contains 79 explanatory variables describing (almost) every aspect of residential homes in Ames. The variable number 80 is the Sale Price which is the target variable to be predicted using other variables. It is more complex to handle, containing too many features and variables, missing data, outliers, and both numerical and categorical features.

## Exploratory Data Analysis (EDA)

Exploratory Data Analysis is a very important process to perform initial investigations on dataset, understand the data, discover the patterns in the data to identify anomalies, test hypothesis and to check assumptions with the help of summary statistics and graphical representations. It is all about making sense of data in hand and getting important insights from it.
The dataset has two parts train dataset and test dataset. The size of train dataset is (118,260) with shape of (1460, 81) whereas the size of test set is (116,720) with shape of (1459, 80). Also, the type of data is heterogeneous containing both numerical (quantitative) and categorical (qualitative) data with number of 38 and 43 variables respectively.
After exploring the data, the next step is to explore the target variable which is the “SalePrice”. Univariate analysis will be used to get more knowledge about the target variable and other important dependant variables using distribution plots (histogram/displot) and probability plot. Multivariate analysis will be used to understand the relationship between the target variable (SalesPrice) and other variables (predictors). For numerical variables, scatter plot, correlation matrix, and pair plot will be used to study the relationship between target variable and numerical variables. For categorical variables, bar plot and boxplot will be used to study the relationship between target variable and categorical variables. Figure 2 summarizes the workflow of exploratory data analysis.

![image](https://user-images.githubusercontent.com/89004966/167241146-fcf3e406-701b-48a6-bfe2-8513c690f55e.png)

Figure 2: Exploratory Data Analysis Workflow.

## Data Processing 

Data preprocessing is an extremely important step in machine learning to enhance the quality of data to promote the extraction of meaningful insights from the data. Data preprocessing in machine learning refers to the technique of preparing (cleaning and organizing) the raw data to make it suitable for a building and training machine learning models. The steps in data preprocessing pipeline are shown in Figure 3, which includes:
- Outlier detection using visual method and interquartile range rule (IQR) technique.
- Identifying and handling the missing values
- Variable transformation using:
  - Log for target variable
  - Scaling (StandardScaler) for other numerical variables.
- Handling categorical variables by using dummy variables and encoding.

![image](https://user-images.githubusercontent.com/89004966/167241219-0ea0018b-c223-4a5b-937c-e4cabe936d18.png)

Figure 3: Data Processing Pipeline.

## Features Engineering
One of the most important steps in machine learning life cycle is the features engineering where the most important features will be selected, new features may be created, or/and existing features may be simplified or eliminated. The goal of features engineering is to simplify and speed up data transformations while also enhancing the performance of machine learning models.
In this project, many features engineering techniques have been used to create features pipeline as shown in Figure 4.

![image](https://user-images.githubusercontent.com/89004966/167241243-49dea2e4-0e34-4699-b499-ad1f1f1abd61.png)

Figure 4: Features Engineering Pipeline.

The first step in features engineering is to leverage mathematical transforms and data manipulation using panadas to create new features and simplify existing features. Many techniques are used to create new features as follows:
-	Combination of existing features.
-	Ratios, creating new features based on the ratio between two variables can often lead to some easy performance gains, for example, creating two new features expressing important ratios using mathematical transformation as follows:
  - LivLotRatio: the ratio of GrLivArea to LotArea.
  - paciousness: the sum of FirstFlrSF and SecondFlrSF divided by TotRmsAbvGrd
- Counts, creating new features based on counting some important features, for example: Creating a new feature called PorchTypes that describes how many kinds of outdoor areas a dwelling has. We will count how many of WoodDeckSF, OpenPorchSF, EnclosedPorch, Threeseasonporch, and ScreenPorch are greater than 0.0. Creating another new feature TotalHalfBath that contains the sum of half-bathrooms within the property.
  - Creating new feature called TotalRoom that sums up the total number of rooms (including full and half bathrooms) in each property.
- Grouped transform, the value of a home often depends on how it compares to typical homes in its neighborhood. Therefore, let's create a new feature called MedNhbdArea that describes the median of GrLivArea grouped on Neighborhood.
- Simplification of the Existing Features
- Filter Feature Selection, the feature selection is the process of selecting a subset of relevant features for use in machine learning model construction. The filter feature selection method is one of features selection methods that ranks each feature based on some uni-variate metric and then selects the highest-ranking features. The criteria used in this project for filter feature selectors is based on the following: 
  - Removing features with small variance, removing the columns with very little variance. Small variance equals small predictive power because all houses have very similar values. (VarianceThreshold)
  - Removing correlated features, the goal of this part is to remove one feature from each highly correlated pair. This can be done in 3 steps:
    - Calculate a correlation matrix
    - Get pairs of highly correlated features
    - Remove correlated columns
- Forward Regression
We have removed the features with no information and correlated features so far. The last thing we will do before modeling is to select the k-best features in terms of the relationship with the target variable. We will use the forward wrapper method for that.
- Mutual information scores, it is one of filter-based feature selection methods. We have a number of features that are highly informative and several that don't seem to be informative at all (at least by themselves). Therefore, we will focus our efforts on the top scoring features. Training on uninformative features can lead to overfitting as well, so features with 0.0 MI scores are going to be dropped entirely.
- Interaction effects, they occur when the effect of an independent variable depends on the value of another independent variable, for example, checking the interaction effect between GrLivArea and BldgType.
- Unsupervised machine learning:
The feature selection can leverage unsupervised algorithm to create new features. For this purpose, the clustering approach (i.e., k-mean clustering) and dimensionality-reduction methods (i.e., principal component analysis) can be used to create new features. The **clustering** approach can be used for feature selection. The formation of clusters reduces the dimensionality and helps in selection of the relevant features for the target class by using cluster labels as features. **Principal Component Analysis (PCA)** is another unsupervised learning method to create more new features. It is a dimensionality reduction method that is often used to reduce the dimensionality of large data sets, by identifying important relationships in dataset, transforms the existing data based on these relationships, and then quantifies the importance of these relationships, keeping the most important relationships and drop the others. When reducing dimensionality through PCA, the most important information will be kept by selecting the principal components that explain most of the relationships among the features. More information will be presented in results section.

## Machine Learning Algorithms

The Ames Housing dataset was chosen due to its richness and huge features that allow us to utilize many machine learning techniques at each stage of the ML project's life cycle.
The main goal behind using Ames Housing dataset is to predict the houses prices based on other housing features, therefore the appropriate ML approach is the regression algorithm which is one of supervised ML algorithms. However, unsupervised machine learning approaches can be applied to the dataset as well to create new features during features engineering stage.
The supervise ML algorithms used in this project for regression are Regression Tree Model, Extreme Gradient Boosting (XGBoost) Model, and Linear Regression. For unsupervised ML, Kmeans clustering, hierarchical clustering, PCA will be used to create new features. Figure 5 summarizes the ML algorithms used in this project.

![image](https://user-images.githubusercontent.com/89004966/169474293-82fa2820-b883-494f-a55e-b687d7ddac51.png)

Figure 5: Machine learning algorithms used in the project.

## Evaluation of Machine Learning Algorithms 
To evaluate the performance of regression models, the root mean squared error (RMSE) will be used to measure and compare the performance of our models. For clustering models, silhouette score will be used to measure the goodness of a clustering models.

![image](https://user-images.githubusercontent.com/89004966/169476427-cbee7c29-5d04-4c28-b40f-55cd7afdc8a2.png)

Figure 6: The RMSE for regression models.
RT:  Regression Tree Model
XGB: Extreme Gradient Boosting
LR: Linear Regression










