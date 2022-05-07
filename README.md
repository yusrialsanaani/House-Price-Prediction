# House-Price-Prediction
#

1. # **Design**
the objective of this project is not only to predict the houses sale. This project leverages the supervised/ unsupervised ML techniques, utilizes different exploratory data analysis and processing techniques, uses different features engineer methods, and then select the appropriate machine learning models. 

The general approach that will be used for the core project design follows the standard machine learning life cycle that includes project planning phase, data preparation, exploratory data analysis, features engineering, modeling, and predictions as shown in Figure ‎11. Each step in the diagram shown in Figure ‎11 will be described in the next subsections.

The next step is to collect and consolidate the data, wrangle it, and conduct exploratory data analysis. Once the data is ready to go, the next steps are to select ML models, split the dataset, train the selected models with training data, fine-tune the models, and evaluate the models using the testing data based on the pre-determined success metrics. The last step is to select the best model and productionize it. 


||
| :-: |
|Figure ‎11: The machine learning project life cycle.|
1. ## **Dataset**
The Ames housing dataset  will be used to predict house prices [9]. This dataset contains 79 explanatory variables describing (almost) every aspect of residential homes in Ames. The variable number 80 is the Sale Price which is the target variable to be predicted using other variables. It is more complex to handle, containing too many features and variables, missing data, outliers, and both numerical and categorical features.
1. ## **Exploratory Data Analysis (EDA)**
Exploratory Data Analysis is a very important process to perform initial investigations on dataset, understand the data, discover the patterns in the data to identify anomalies, test hypothesis and to check assumptions with the help of summary statistics and graphical representations. It is all about making sense of data in hand and getting important insights from it.

The dataset has two parts train dataset and test dataset. The size of train dataset is (118,260) with shape of (1460, 81) whereas the size of test set is (116,720) with shape of (1459, 80). Also, the type of data is heterogeneous containing both numerical (quantitative) and categorical (qualitative) data with number of 38 and 43 variables respectively.

After exploring the data, the next step is to explore the target variable which is the “SalePrice”. Univariate analysis will be used to get more knowledge about the target variable and other important dependant variables using distribution plots (histogram/displot) and probability plot. Multivariate analysis will be used to understand the relationship between the target variable (SalesPrice) and other variables (predictors). For numerical variables, scatter plot, correlation matrix, and pair plot will be used to study the relationship between target variable and numerical variables. For categorical variables, bar plot and boxplot will be used to study the relationship between target variable and categorical variables. Figure ‎12 summarizes the workflow of exploratory data analysis.

||
| :-: |
|Figure ‎12: Exploratory Data Analysis Workflow.|

1. ## **Data Processing** 
Data preprocessing is an extremely important step in machine learning to enhance the quality of data to promote the extraction of meaningful insights from the data. Data preprocessing in machine learning refers to the technique of preparing (cleaning and organizing) the raw data to make it suitable for a building and training machine learning models. The steps in data preprocessing pipeline are shown in Figure ‎13, which includes:

- Outlier detection using visual method and interquartile range rule (IQR) technique.
- Identifying and handling the missing values
- Variable transformation using:
  - Log for target variable
  - Scaling (StandardScaler) for other numerical variables.
- Handling categorical variables by using dummy variables and encoding.

||
| :-: |
|Figure ‎13: Data Processing Pipeline.|
1. ## **Features Engineering**
One of the most important steps in machine learning life cycle is the features engineering where the most important features will be selected, new features may be created, or/and existing features may be simplified or eliminated. The goal of features engineering is to simplify and speed up data transformations while also enhancing the performance of machine learning models.

In this project, many features engineering techniques have been used to create features pipeline as shown in Figure ‎14.

||
| :-: |
|Figure ‎14: Features Engineering Pipeline.|

The first step in features engineering is to leverage mathematical transforms and data manipulation using panadas to create new features and simplify existing features. Many techniques are used to create new features as follows:

- Combination of existing features.
- Ratios, creating new features based on the ratio between two variables can often lead to some easy performance gains, for example, creating two new features expressing important ratios using mathematical transformation as follows:
  - LivLotRatio: the ratio of GrLivArea to LotArea.
  - Spaciousness: the sum of FirstFlrSF and SecondFlrSF divided by TotRmsAbvGrd
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

The feature selection can leverage unsupervised algorithm to create new features. For this purpose, the clustering approach (i.e., k-mean clustering) and dimensionality-reduction methods (i.e., principal component analysis) can be used to create new features. The **clustering approach** can be used for feature selection. The formation of clusters reduces the dimensionality and helps in selection of the relevant features for the target class by using cluster labels as features. More information will be presented in results section [10]. **Principal Component Analysis** (PCA) is another unsupervised learning method to create more new features. It is a dimensionality reduction method that is often used to reduce the dimensionality of large data sets, by identifying important relationships in dataset, transforms the existing data based on these relationships, and then quantifies the importance of these relationships, keeping the most important relationships and drop the others. When reducing dimensionality through PCA, the most important information will be kept by selecting the principal components that explain most of the relationships among the features. More information will be presented in results section.
1. ## **Machine Learning Algorithms**
The Ames Housing dataset was chosen due to its richness and huge features that allow us to utilize many machine learning techniques at each stage of the ML project's life cycle.

The main goal behind using Ames Housing dataset is to predict the houses prices based on other housing features, therefore the appropriate ML approach is the regression algorithm which is one of supervised ML algorithms. However, unsupervised machine learning approaches can be applied to the dataset as well to create new features during features engineering stage.

The supervise ML algorithms used in this project for regression are Regression Tree Model, Extreme Gradient Boosting (XGBoost) Model, and Linear Regression. For unsupervised ML, Kmeans clustering, hierarchical clustering, PCA will be used to create new features. Figure ‎15 summarizes the ML algorithms used in this project.

||
| :-: |
|Figure ‎15: Machine learning algorithms used in the project.|

1. ## **Evaluation of Machine Learning Algorithms** 
To evaluate the performance of regression models, the root mean squared error (RMSE) will be used to measure and compare the performance of our models. Besides that, the bias and variance will be calculated as well to analyze the bias and variance errors for each model. For clustering models, silhouette score will be used to measure the goodness of a clustering models.

Figure ‎16 summarizes the evaluation metrics used for evaluating the performance of regression and clustering models.

||
| :-: |
|Figure ‎16: Evaluation metrics for regression and clustering models.|

1. ## **Best Model Selection and Hyperparameters Tuning**
The best predictive regression model will be selected based on RSME score compared with other regression models. The model with lowest RSME score is the best model, then the hyperparameters tuning will be done on it for further enhancing its performance. After hyperparameters tuning and checking its performance again, the model will be ready for predictions.


63

1. # **Results**
   1. ## **Exploratory Data Analysis (EDA)**
      1. ## **Data Discovery**
The dataset used in this project is Ames housing dataset that contains train set and test set. The size of trainset is (118,260) with shape of (1460, 81) whereas the size of test set is (116,720) with shape of (1459, 80). 

The first step is to check the types of variables the train dataset has. By using Python, the type of variables has been identified. The dataset has 38 numerical (quantitative) variables and 43 categorical (qualitative) variables as well. Also, there are no duplicates in the dataset.
1. ## **Exploring the Target Variable**
The second step, we need to explore the target variable using descriptive statistics and the histogram. The goal of this project is to predict the price of a house. These values are stored in the SalePrice variable. Let's check the descriptive statistics and the histogram of SalePrice variable as shown in Table ‎51.

|Table ‎21. Generating descriptive statistics for SalePrice variable.|
| :-: |
||
Now, lets check the histogram distribution and probability plot for SalePrice variable as shown in Figure ‎53 and Figure ‎54.

||
| :-: |
|Figure ‎53: SalePrice (Target Variable) distribution.|


||
| :-: |
|Figure ‎54: The probability plot for SalePrice variable.|

Since the histogram is a good chart to check the distribution of variables, we need to calculate the skewness and kurtosis for SalePrice variable as well. The skewness represents the bias of the distribution, and kurtosis represents the sharpness of the distribution. Their calculated values for SalePrice variable are as follows: Skewness = 1.883 whereas Kurtosis = 6.536. The target variable (SalePrice in our case) should follow normality to ensure that the model performs well. The absolute value of skewness is 0-3 and kurtosis is 1-8.

From Figure ‎53 and Figure ‎54 above, we can observe that the distribution has a long tail and most of the density of sale's price lies between 100k and 200k. It means that most of the house are normally distributed but a couple of houses have a higher than normal value, resulting in slightly deviation from a normal distribution. Also, it’s skewed and has some outliers. So. It is critical to take this peculiarity into account when designing predictive models.
1. ## **Numerical and Categorical Variables**
For simplicity, we will analyze the numerical and categorical variables separately. So, the third step is to generate descriptive statistics for both numerical and categorical variables using python. These descriptive statistics give us a general idea about the central tendency, dispersion and shape of a dataset’s distribution, excluding NaN values. 

Table ‎52 shows the summary of these descriptive statistics for numerical variables. Also, we can obtain a concise summary of them including the index data type and columns, non-null values and memory usage as shown in Table ‎53. 

After looking at the Table ‎52 and Table ‎53, we can see that 36 numerical features are available, and that the dataset contains 1460 samples. However, some features contain missing values. When looking at min, max, and mean, we can see that some variables contain outliers, and the features are not normally distributed. We will examine the distribution, outliers, and missing values later in this section and in data preprocessing section. 

Similarly, Table ‎54 shows the summary of descriptive statistics for categorical variables in term of count, top, unique values, and frequency. Also, we can obtain a concise summary of their information including the index data type and columns, non-null values, and memory usage as shown in Table ‎55.

After looking at the Table ‎54 and Table ‎55, we can see that 43 categorical features are available, and that the dataset contains 1460 samples. However, some features contain missing values. We will examine the distribution, outliers, and missing values later in this section and in data preprocessing section.






|*Table ‎52. Generating descriptive statistics for numerical variables.*|
| :-: |
||



|*Table ‎53. Generating concise information summary for numerical variables.*|
| :-: |
||



|Table ‎54. Generating descriptive statistics for categorical variables.|
| :-: |
||



|*Table ‎55. Generating concise information summary for categorical variables.*|
| :-: |
||




1. ## **Univariate Analysis for Numerical Variables**
To check the distribution of all numerical variables, we plotted them together as shown in Figure ‎55 and Figure ‎56. They show that some variables seem to be able to follow a normal distribution through log transformation or box cox transformation. 

Some variables were extremely biased toward one value (0) since they have high picks for 0. It could be linked that this value was assigned when the criterion did not apply, for instance the area of the swimming pool when no swimming pools are available. So, they don't seem to be very important variables. 

We also have some feature encoding some date (for instance year). This information is useful and should also be considered when designing a predictive model. Since these plots are packed together, we can take random variables and plot them separately to look at their distribution more clearly as shown in Figure ‎57 and Figure ‎58.

||
| - |
|Figure ‎55: The distribution plots for numerical variables using distplot.|
||
|Figure ‎56: The distribution plots for numerical variables using histplot.|


|||
| :-: | :-: |
|a) distplot|b) hist plot|
|Figure ‎57: The distribution plots for LoftFrontage variable.|
|||
|a) distplot|b) histplot|
|Figure ‎58: The distribution plots for BsmtFinSF2 variable.|

1. ## **Univariate Analysis for Categorical Variables**
For categorical variables, we created bar plots to see categories count for all categorical variables. Plotting this information allows us to find how many categories a given feature has and discover if there are rare categories for some features since knowing that would help at designing the predictive pipeline.

Since there are too many categorical variables, we provided plots for selected categorical variables that we think they have good disparity with respect to SalePrice. We will verify this hypothesis later when analysing the relationship between categorical variables to SalePrice in multivariate analysis. Figure ‎59 (a-m) shows multiple plots for categories count for selected categorical variables including MSZoning, Neighborhood, Condition1, Condition2, RoofMatl, MasVnrType, ExterQual, BsmtQual, BsmtCond, CentralAir, KitchenQual, SaleType, and SaleCondition respectively. 


|||
| :-: | :-: |
|(a)|(b)|
|||
|(c)|(d)|
|||
|(e)|(f)|
|||
|(g)|(h)|
|||
|(i)|(j)|
|||
|(k)|(l)|
||
|(m)|
|Figure ‎59: The categories count for multiple selected categorical variables.|

1. ## **Multivariate Analysis for Numerical Variables**
Now, we can explore the relationship between numerical variables and target variable using scatter plots, for example, Figure ‎510 and Figure ‎511 show the relationship of GrLivArea and TotalBsmtSF with regard to SalesPrice respectively. We can see a linear relationship in the Figure ‎510 and a quadratic relationship in Figure ‎511.

||
| :-: |
|Figure ‎510: The relationship plot between SalePrice and GrLivArea.|
||
|Figure ‎511: The relationship plot between SalePrice and TotalBsmtSF.|
||

We need to check the correlation of numerical variables with SalePrice variable as the target variable. Figure ‎512 shows the correlation of all numerical variables with SalePrice whereas Figure ‎513 shows the correlation matrix between numeric attributes.

|||
| -: | :- |
|||
|Figure ‎512: The correlation of all numerical variables with SalePrice.|


||
| :-: |
|Figure ‎513: The correlation matrix between numeric attributes.|
To make the correlation matrix more readable, we can plot only those variables that correlate with the target variable more than 0.5 to obtain the most correlated features with SalePrice. Figure ‎514 shows the features which the correlation is higher than 0.50 and Figure ‎515 shows the correlation matrix for most top correlated features.

Based on correltion values and correlation matrix, we can see that OverallQual, GrLivArea, TotalBsmtSF, GarageCars, GarageArea, YearBuilt, and FullBath are strongly correlated with SalePrice. GarageCars and GarageArea are like twin brothers. You'll never be able to distinguish them. Therefore, we just need one of these variables in our analysis (we can keep GarageCars since its correlation with SalePrice is higher). TotalBsmtSF and 1stFloor also seem to be twin brothers. TotRmsAbvGrd and GrLivArea are twin brothers again. However, 1stFlrSF, GarageCars, TotRmsAbvGrd are excluded because they are related to one of those most correlated ones. The relationship between these top correlated variables and target variable (SalePrice) is shown in Figure ‎516 (a-f).

||
| :-: |
|Figure ‎514: The correlation of numerical variables with SalePrice when correlation > 0.5|
||
|Figure ‎515: The correlation matrix for most top correlated features.|



|||
| :-: | :-: |
|**(a)**|**(b)**|
|||
|**(c)**|**(d)**|
|||
|**(e)**|**(f)**|
|Figure ‎516: The relationship between SlaPrice and top correlated variables.|

From those charts, highly correlated variables mentioned above have sort of linear relationship with 'SalePrice'. Those charts also imply some exponential relationships, so we can log transform some features to get a better model. Candidates for log transformation: TotalBsmtSF, GrLivArea.





1. ## **Multivariate Analysis for Categorical Variables**
Also, we can explore the relationship between categorical variables and target variable using boxplot. We performed analysis for boxplot for all categorical variables and found that variables having good disparity with respect to SalePrice are: MSZoning, Neighborhood, Condition1, Condition2, RoofMatl, MatVnrType, ExterQual, BsmtQual, BsmtCond, CentralAir, KitchenQual, SaleType, SaleCondition. For simplicity, we will present only these plots in this report.

Figure ‎517 shows the relationship between Neighborhood and SalePrice. We can see that there are some outliers need to be eliminated. Similarly, Figure ‎518 (a-l) shows the relationship between selected categorical variables mentioned above and SalePrice. We can see that there are some outliers need to be eliminated as well.

||
| :-: |
|Figure ‎517: The relationship plot between SalePrice and Neighborhood.|

|||
| :-: | :-: |
|**(a)**|**(b)**|
|||
|**(c)**|**(d)**|
|||
|**(e)**|**(f)**|
|||
|**(g)**|**(h)**|
|||
|**(i)**|**(j)**|
|||
|**(k)**|**(l)**|
|Figure ‎518: The relationship plot between SalePrice and top selected categorical variables.|

1. ## **Data Preprocessing**
   1. ## **Outlier Treatment**
Let's check the values of our target variable (SalePrice) together with the other numeric attributes. Figure ‎519 shows the scatter plot between SalePrice and TotalBsmtSF. We can see that there is one house with a huge basement and a cheap price, this is an outlier. This house would cause problems in modeling because, except for this one house, we can see a clear linear relationship between the size of a basement and the price of a house. So, we need to remove this outlier from the dataset.

We used two techniques to remove outliers, the first one is based on visual representation of variables of which we need to remove the outliers from and the target variable. Then, we can identify the outlier location and use Panadas to remove it. The second way is to use Interquartile range (IQR) method, which is stricter that the visual way.

For outliers shown in Figure ‎519, we will remove all observations that have more than 5,000 square feet of basement and a price lower than $300,000, Figure ‎520 shows the scatter plot between SalePrice and TotalBsmtSF after removing the outliers using visual methods. We can see that the relationship in the picture is much nicer now.

Figure ‎521 shows the scatter plot between SalePrice and TotalBsmtSF after outliers removal using IQR method. It is clear that IQR method is stricter that visual one because it removed low and high outliers.

||
| :-: |
|Figure ‎519: The scatter plot between SalePrice and TotalBsmtSF.|
||
|Figure ‎520: The scatter plot between SalePrice and TotalBsmtSF after outliers removal using visual method.|
||
|Figure ‎521: The scatter plot between SalePrice and TotalBsmtSF after outliers removal using IQR method.|
Now let’s take another variable and apply the two methods to remove its outliers. Figure ‎522 shows the scatter plot between SalePrice and GrLivArea. If we look only at GrLivArea there are no outliers because the largest area is quite close to the second and the third largest. However, if we take a look at SalePrice together with GrLivArea, we can see that the price of the largest house is really small, which will again cause problems in the modeling step. Therefore, we should remove this observation as well.

Figure ‎523 shows the scatter plot between SalePrice and GrLivArea after outliers removal using Visual method whereas Figure ‎524 shows the removing outliers using IQR method. We followed the same way to check the outliers and remove them if available in other variables as a step in data preprocessing. So, we provide only two examples here to show how we treated the outliers in the dataset and make the report short instead of filling it with repeated examples.

||
| :-: |
|Figure ‎522: The scatter plot between SalePrice and GrLivArea.|
||
|Figure ‎523: The scatter plot between SalePrice and GrLivArea after outliers removal using Visual method.|
||
|Figure ‎524: The scatter plot between SalePrice and GrLivArea after outliers removal using IQR method.|
1. ## **Null Values Replacement**
As a first step, we will check whether we have columns with missing values. *Table ‎56* shows number of missing values and their percentages for each column has missing values. Figure ‎525 shows the percentages of missing values.

The first strategy is to drop the five columns with the biggest percentage of null values and check the missing values in dataset to make sure that the highest percentages have been dropped. Figure ‎526 shows the remaining percentages of missing vales after dropping the highest percentages.

Now we can work on the remaining Null values to replace them with appropriate values. We replaced the missing values of LotFrontage with the mean. For GarageYrBlt variable, if the house has a garage and the year is missing, we assume it's the minimum value available. If the veneer area (MasVnrArea) is missing, we assume it's 0.

With the help of the data documentation we have, we can figure out that the missing values in Garage and Basement variables mean no garage and no basement respectively. Therefore, we will replace the missing values with "None"

The information about Electrical and MasVnrType is missing in the documentation. Since we are dealing with categorical variables, we will create a new category for a missing value called ‘Empty’ and replace the missing values of Electrical and MasVnrType with it. After replacing the missing values, we checked the dataset again to ensure that there are zero missing values.




|*Table ‎56. Missing values and their percentages.*|
| :-: |
||
||
|Figure ‎525: The percentages of missing vales|
||
|Figure ‎526: The percentages of missing vales after dropping the highest percentages.|

1. ## **Variable Transformation**
Log transformation has been used to normalize the data that does not follow the normal distribution. Before performing the log transform, let’s plot the distribution of our target (SalePrice) first and compare it to normalized one. Figure ‎527 and  Figure ‎528 show the distribution and probability plots for SalePrice variable before using log transformation. We can observe that the distribution has a long tail and most of the density of sale's price lies between 100k and 200k. It means that most of the house are normally distributed but a couple of houses have a higher than normal value, resulting in slightly deviation from a normal distribution with skewness of 1.88. 

Figure ‎529 and Figure ‎530 show the distribution and probability plots for SalePrice variable after applying log transformation. It is clear that the logarithmic transformation of the SalePrice is more normal and skewness has been reduced to 0.12. We applied log transformation for other numerical variables that do not follow the normal distribution.

Furthermore, the StandardScaler has been used from sklearn to scale variables that do not highly deviate from normal distribution. Scaling is important for some algorithms that require to have values with the same scale, for example between 0 and 1.

||
| :-: |
|Figure ‎527: SalePrice (Target Variable) before log transformation.|
||
|Figure ‎528: The probability plot for SalePrice variable before log transformation.|
||
|Figure ‎529: SalePrice (Target Variable) after log transformation.|
||
|Figure ‎530: The probability plot for SalePrice variable after log transformation.|

1. ## **Categorical Variables Encoding**
In our dataset, we have two types of categorical variables including nominal and ordinal variables. The difference is that with an ordinal variable, we can order the categories by importance/value/score, whereas a nominal variable has no intrinsic ordering to its categories. Therefore, we can transform the ordinal variables into numbers and create numeric variables out of them. We will use the help of our documentation to understand the sorting. For nominal variables, we will use dummy variables to encode them. These dummy variables will be created with one-hot encoding and each attribute will have a value of either 0 or 1, representing the presence or absence of that attribute.
1. ## **Features Engineering**
   1. ## **Combination of the Existing Features**
An example of combining existing variables is to sum two variables. We have two variables 1stFlrSF and 2ndFlrSF but we don't have the total square footage. So, we can create a new feature which represents the sum of these two called 1stFlr\_2ndFlrSF as shown in Figure ‎531. We can see that there is a significant relationship between the new variable and our target (SalePrice).

||
| :-: |
|Figure ‎531: Creating new feature by summing two variables (1stFlrSF and 2ndFlrSF).|

Another example is to create OverallGrade feature by multiplying two variables (OverallQual \* OverallCond). The result of the new feature is shown in Figure ‎532.

In this project, we created new features based on ratios between two variables for example, creating two new features expressing important ratios using mathematical transformation as follows: LivLotRatio: the ratio of GrLivArea to LotArea. Spaciousness: the sum of FirstFlrSF and SecondFlrSF divided by TotRmsAbvGrd. 



Also, we created new features based on counting some important features, for example: 

- Creating a new feature called PorchTypes that describes how many kinds of outdoor areas a dwelling has. We will count how many of WoodDeckSF, OpenPorchSF, EnclosedPorch, Threeseasonporch, and ScreenPorch are greater than 0.0.
- Creating another new feature TotalHalfBath that contains the sum of half-bathrooms within the property.
- Creating new feature called TotalRoom that sums up the total number of rooms (including full and half bathrooms) in each property.
- Grouped Transform, the value of a home often depends on how it compares to typical homes in its neighborhood. Therefore, we created a new feature called MedNhbdArea that describes the median of GrLivArea grouped on Neighborhood.

||
| :-: |
|Figure ‎532: Creating new feature by multiplying two variables (OverallQual \* OverallCond).|

1. ## **Simplification of the Existing Features**
An example of simplification features is to simplify GarageQual which is an ordinal feature. Let's check the distribution per category of GarageQual as shown in Figure ‎533. We can see that there are categories for which the SalePrice is similar. If we move from the category 0 to 1 or 2, there is no change in SalePrice. Therefore, we can merge these categories into one as shown in Figure ‎534. We can see that the simplified feature has a "nicer" relationship. If we go from 0 to 1 or 2, the average SalePrice increases.

||
| :-: |
|Figure ‎533: The distribution per category of GarageQual variable before simplification.|


||
| :-: |
|Figure ‎534: The distribution per category of GarageQual variable after simplification.|

1. ## **Filter Feature Selection**
The criteria used for filter feature selectors is based on the following: 

- Removing features with small variance, removing the columns with very little variance. Small variance equals small predictive power because all houses have very similar values.
- Removing correlated features, the goal of this part is to remove one feature from each highly correlated pair. This can be done in 3 steps:
  - Calculate a correlation matrix
  - Get pairs of highly correlated features
  - Remove correlated columns
- Forward Regression

We have removed the features with no information and correlated features so far. The last thing we will do before modeling is to select the k-best features in terms of the relationship with the target variable. We will use the forward wrapper method for that. After performing filter feature selection, we came up with10 features which should be pretty good predictors of our target variable, SalePrice as shown in *Table ‎57*.

|*Table ‎57. The 10 top features obtained from filter feature selection.*|
| :-: |
||

1. ## **Mutual Information (MI) Scores**
Mutual information score is one of filter-based feature selection methods. We have several features that are highly informative and several that don't seem to be informative at all. Therefore, we will focus our efforts on the top scoring features. Figure ‎535 shows the mutual information scores of our dataset. Training on uninformative features can lead to overfitting as well, so features with 0.0 MI scores have been dropped from the dataset.
1. ## **Interaction Effects**
Interaction effects occur when the effect of an independent variable depends on the value of another independent variable, for example, checking the interaction effect between GrLivArea and BldgType as shown in Figure ‎536. The trend lines being significantly different from one category to the next indicates an interaction effect between GrLivArea and BldgType that relates to a home's SalePrice.  Several other detected interaction effects between categorical and numerical variables have been performed too.



||
| :-: |
|Figure ‎535: The mutual information scores.|
||
|Figure ‎536: The interaction effect between GrLivArea and BldgType.|


1. ## **Creating New Feature Using Clustering**
The clustering approach can be used for feature selection. The formation of clusters reduces the dimensionality and helps in selection of the relevant features for the target class by using cluster labels as features. We used k-means clustering and hierarchical clustering. 

The first step in k-means clustering is to select the number of clusters first. We used the elbow method to find the appropriate number of clusters as shown in Figure ‎537. The elbow method runs k-means clustering on the dataset for a range of values for k (1-30) and then for each value of k computes an average score for all clusters. By default, the distortion score is computed, the sum of square distances from each point to its assigned center. We selected k = 5 at the corresponding small inflection since the distortion does not have high change after this point.

The k-means clustering model has been trained with k = 5, the clusters have been visualized in Figure ‎538. K-means model resulted with good clusters with average silhouette score of 0.474 for 5 clusters. Figure ‎539 shows the silhouette score visualization for k-means clustering model. The distance of the observations to each cluster has been calculated to be used as another new feature.

||
| :-: |
|Figure ‎537: The elbow method to find the appropriate number of clusters.|
||
|Figure ‎538: The clusters visualization of k-means model. |
||
|Figure ‎539: The silhouette score visualization for k-means with k = 5.|

Moreover, we used hierarchical clustering (Agglomerative algorithm) as well. We plotted a dendrogram first to find out the number of clusters as shown Figure ‎540. To find the number of clusters, we added a horizontal line across the dendrogram that cuts the long vertical lines and not the small vertical lines (clusters). By counting the total vertical lines that this horizontal line cuts, we can find the number of clusters to pass to the hierarchical clustering model. From the dendrogram, we can find that the number of clusters = 6. After running the hierarchical clustering (Agglomerative algorithm) with n = 5, the resulted clusters are plotted as shown in Figure ‎541. The silhouette score is 0.431 which is almost close to silhouette score of k-means clustering.

||
| :-: |
|Figure ‎540: The dendrogram plot for hierarchical clustering.|
||
|Figure ‎541: The clusters visualization of hierarchical clustering model.|

1. ## **Principal Component Analysis (PCA)**
PCA is another unsupervised learning method to create more new features. It is a dimensionality reduction method that is often used to reduce the dimensionality of large data sets, by identifying important relationships in dataset, transforms the existing data based on these relationships, and then quantifies the importance of these relationships, keeping the most important relationships and drop the others. When reducing dimensionality through PCA, the most important information will be kept by selecting the principal components that explain most of the relationships among the features. More information will be presented in results section.

First, we plotted a matrix dataset as a hierarchically clustered heatmap to order dataset by similarity as shown in Figure ‎542. This reorganizes the data for the rows and columns and displays similar content next to one another for even more depth of understanding the dataset.

||
| - |
|Figure ‎542: The hierarchically clustered heatmap|
The PCA algorithm gives us *loadings* which describe each *component* of variation, and also the components which were the transformed datapoints. The loadings can suggest features to create. Additionally, we can use the components as features directly. After performing the PCA, the explained variance and cumulative variance based on components from PCA have been plotted as shown in Figure ‎543. The goal is to use the results of PCA to discover one or more new features that could improve the performance of our models, we can use PCA loadings to create features.

||
| :-: |
|Figure ‎543: The explained variance and cumulative variance.|

1. ## **Regression Algorithms**
As mentioned previously, the main goal behind using Ames Housing dataset is to predict the houses prices based on other housing features, therefore the appropriate ML approaches are the regression algorithms which are supervised ML algorithms. These algorithms include regression tree model, extreme gradient boosting (XGBoost) model, and linear regression. To compare the performance of these algorithms, we used Root Mean Square Error (RMSE).
1. ## **Baseline Models**
At first, these models were trained with dataset before applying the features engineering results. The goal here is to create baseline models to compare the performance of regression models with after applying the features engineering to see how much features engineering results improve the performance of regression models.

Figure ‎544 shows the RMSE for baseline models, it is clear that extreme gradient boosting (XGB) has the lowest RMSE value followed by regression tree (RT) model, whereas liner regression (LR) model has the highest value. 

||
| :-: |
|Figure ‎544: The RMSE for baseline models.|
1. ## **Regression Models with Engineered Features**
After training the models with engineered features, the performance of extreme gradient boosting (XGB) and liner regression (LR) model have been improved as shown in Figure ‎545. The performance of regression tree (RT) model went to the worse. Figure ‎546 compares the performance of regression models in term of RMSE. Also, *Table ‎58* shows the exact RMSE values before and after features engineering. We can conclude that XGB and LR have been improved after applying features engineering, but XGB is still the best. Therefore, we selected XGB as our champion model and went further to tune its parameters by performing GridSearchCV to optimize XGB’s hyperparameters. Figure ‎547 shows the RMSE plot before and after hyperparameter tuning for XGB model. The tuned XGB has lower RMSE, so the tuning improved XGB’s performance.

||
| :-: |
|Figure ‎545: The RMSE for regression models after applying features engineering.|
||
|Figure ‎546: The RMSE of regression models before and after applying features engineering.|


|*Table ‎58. RMSE values for regression models before and after applying features engineering.*|
| :-: |
||
||


||
| :-: |
|Figure ‎547: The RMSE of XGB model before and after parameters tuning.|

The XGB model has been saved as pickle format to be used for prediction and avoid retraining. Also, the model has been scored with test dataset and provide predictions for houses prices as shown in *Table ‎59*.

|*Table ‎59. The sale price predictions for houses prices using XGB model.*|
| :-: |
||
||
1. ## **Feature Importance** 
We can get information about what features in the dataset the model think they are most important. Figure ‎548 shows the feature importance based on SHAP values, so the values have been ordered based on their importance to the model. We can check, for any single prediction from a model, how did each feature in the data affect that particular prediction as shown in Figure ‎549. 

Finally, Figure ‎550 shows the feature impact on overall model prediction. So, we can get idea about how does each feature affect the model's predictions in a big-picture sense, i.e., what is its typical effect when considered over a large number of possible predictions.





||
| :-: |
|Figure ‎548: Feature importance based on SHAP values.|
||
||
||
|Figure ‎549: Feature impact on the model prediction.|


||
| :-: |
||
|Figure ‎550: Feature impact on overall model prediction.|



1. ## **Automated Solution Module**
The final delivery of this project to CENGN is to provide a notebook that contains the Python code, step by step instructions, notes, etc. 

Since the project is big and its code is too long, the code notebook has been divided mainly into four sub-modules including exploratory data analysis (EDA) module, preprocessing module, features engineering module and modelling module. The CENGN team will deploy the content of this project into CENGN Machine Learning course platform and its lab platform. 

Since the project will be used as a guided project in the end of CENG Machine Learning course, CENGN will integrate two versions of this project. The first one contains only the steps, hints and instructions and this version will be integrated with the course platform first, so the learners can go over it first and try it themselves. Then, the learners will be forwarded to the lab platform where the second complete version deployed there, so the learners can follow the instructions and check their solutions. 

It is possible to create a special solution module that can be run on Jupyter Notebook to provide hints and solutions by creating python classes and methods to provide solutions and hints for each step when calling the hint or solution methods, this is a kind of creating automated code checker providing interactive notebook for the learners. So, the learners can start following the instructions and try coding themselves, Figure ‎551 shows an example about how the instructions provided to the learners and how to provide specific hints and solutions for each step, the solutions and hint should be imported first from the learning tool module as shown before step 2. 

If the learners stuck somewhere, they could call hint() method that provide a specific hint for that step as shown in Figure ‎552. If the learner is still sticking there, he\she can call the solution() method to provide a specific solution for that part as shown in Figure ‎553.  Figure ‎554 shows another example of providing automated hints and solutions.

||
| :-: |
|Figure ‎551: An example about how the instructions, hints, and solutions provided to the learners.|
||
|Figure ‎552: An example about how to provide automated hints.|


||
| :-: |
|Figure ‎553: An example about how to provide automated solutions.|


||
| :-: |
||
|Figure ‎554: Another example about how to provide automated hints and solutions.|


1. # **Evaluation**
Since our project is about designing a guided ML project that will be used as learning module in CENGN machine learning course, the following criteria was followed as testing, evaluation, and validation plan to ensure that the project workflow meets the requirements. These criteria include:

- Models performance
- Code Reliability & Reproducibility 
- Code Readability 
- Instructions, Tips, Details
- Coverage, the project should provide a good coverage for CENGN ML course such as:
  - Supervise/ Unsupervised ML
  - Features Engineering
  - EDA & Visualization

The performance of machine learning models has been evaluated on a testing set using statistical metrics to assess testing results including root mean squared error (RMSE), bias/variance errors to evaluate the performance of regression models whereas silhouette score was used evaluate clustering models. The models performance was controlled by manual testing for a random couple of data points. Then, the accuracy of the ML models was evaluated to achieve acceptable loss.

The software testing was conducted by performing unit test (break down the program into blocks, and test each block separately), regression test (covers already tested software to see if it doesn’t suddenly break) and integration test (observes how multiple components of the program work together). The code reliability, reproducibility, readability, instructions, tips, details, and coverage have been tested and validated by team members and mentor.

When it comes to testing the guided project, it must meet the previously mentioned requirements of being relatable to students, containing all subjects taught in the course and promoting critical thinking. Since we as a group were new towards machine learning as students and had to take the course, two of the requirements being containing all subject matter and promoting critical thinking can be tested by us as students. Although we were also developing the project, the idea behind the project was simple being the projection of sales price but the results from it was testing our ability to critically think which can be assumed to be similar to other students from CENGN due to the fact that we share the same background according to CENGN.

The reason that we as students will be the main test subjects is because the project overall takes a bit less than a semester to complete. Although to diversify testing, team members were tested in completing aspects of the guided project that they were not influenced in designing. 

Abdalla Osman tested the supervised machine learning chapter, while Yusri Al-Sanaani tested the unsupervised machine learning chapter, while Yacine Diagne tested the supervised and unsupervised machine learning chapters. It was concluded that it met the requirement of promoting critical thinking due to the fact that the team members were able to complete the other sections although the aspect where the most critical thinking is required cannot be tested currently. That aspect being connecting between the data that requires to be learned via unsupervised learning with the results that can only be extrapolated from supervised learning. This can be done via using unsupervised learning to explore features and association within the data, but it could only be tested by Yacine who already knew the generality of the solution beforehand. So, the testing will be done in the future with CENGN on future students by giving the general solution as a hint to some students and not to other students while witnessing the consequences of such actions. Based on how students respond to such stimuli, will the final decision be made. For the third requirement, that of being relatable to students, a test was done checking the ease of access for information pertaining to it and it was concluded that machine learning and real estate price prediction is a very well-known topic that can be easily accessed and learned about [1].

Through this project, we built a strong background and advanced our skills in python programming, data science, machine learning through the following stages:

- Recognizing the key concepts, best practices, and applications of machine learning
- Identifying the most widely used machine learning algorithms and their strengths and weaknesses.
- Reviewing the basics of probability, statistics, linear algebra, and calculus.
- Gaining wide handy experience working on the following topics:
- Basic machine learning models such as classification, regression, clustering, association learning, and dimensionality reduction.
- Building, training, and evaluating the performance of machine learning models using Python and its associated libraries.
- Selecting the appropriate machine learning model for a given problem.
- Performing different techniques for exploratory data analysis on a dataset to detect anomalies and to summarize its main characteristics.
- Performing different data processing techniques.
- Through the CENGN machine learning course and guided project, we also learned the machine learning trends in industry and how ML drives the industry and businesses.

In this project, we used different tools such as Jupyter Notebook, google colab, GitHub, Lucid chart, Python, PowerPoint, word, OneDrive, and google drive. We also gained project management and professional skills such as project plans, communication skills, team building, user needs analysis, etc. 

1. # **Conclusion and Future Work**
In this project, we designed a guided project for the CENGN Machine Learning course to be used in the end of their course to help learners consolidate and extend their CENGN learning. This project was meant to provide a complex problem to enhance the learning cycle through providing an end-to-end real-world based project. It was designed in a way that covers the main topics explained in the CENGN Machine Learning course, providing clear instructions for learners to follow, and create reliable and readable modular codes, thus enabling participants to check their code against the project requirements and allowing learners to consolidate and extend their learning in a new and innovative way that imprints their new gained knowledge.

To design a project that meets the mentioned specifications, the Ames housing dataset was used to predict house prices. This dataset contains 79 explanatory variables describing (almost) every aspect of residential homes in Ames. It is quite complex to handle, containing an uncountable number of features and variables, along with missing data, outliers, and both numerical and categorical features. These characteristics make this dataset a great candidate for the proposed project since we can apply different techniques to process the data, explore it and apply different supervised/unsupervised techniques and statistical analysis methods to select and create features, analyze and visualize the data, train and optimize ML algorithms and so on.

The standard machine learning life cycle was followed in this project which includes a project planning phase, data preparation, exploratory data analysis, features engineering, modeling, and predictions. The main goal behind using Ames Housing dataset is to predict the houses prices based on other housing features, therefore the appropriate ML approach was the regression algorithm which is one of the supervised ML algorithms learned from CENGN. However, unsupervised machine learning approaches were applied to the dataset as well to create necessary new features during the features engineering stage. The supervised ML algorithms used in this project for regression are the regression tree model, extreme gradient boosting (XGBoost) model, and linear regression. For unsupervised ML, K-means clustering, hierarchical clustering, and principal component analysis (PCA) were used to create new features.

To conclude, the project leverages many techniques to process, analyze, and visualize the data. It also leverages many techniques to create creative features and provide great predictions. Therefore, the project covers most machine learning concepts explained in the CENGN machine learning course. Most of our results in terms of figures and tables are shown in this report. However, other computational results that are based on python coding had their ideas and concepts explained.

Throughout this project, a strong background was built and our skills in python programming, data science, and machine learning was advanced. We also learned of trends in the industry when it comes to machine learning and which industries will it be applicable in. We also improved our communication and teamwork skills throughout our project design and implementation.

In possible future work, we suggest using pipeline of transformations with a final model to assemble several steps that can be cross validated together while setting different parameters. Also, we suggest an appropriate deployment to integrate the final trained and scored regression model into an existing product environment to make practical business decisions based on the trained data.


1. # **References** 
[1]	I. H. Sarker, “Machine Learning: Algorithms, Real-World Applications and Research Directions,” SN Comput. Sci., vol. 2, no. 3, pp. 1–21, Mar. 2021, doi: 10.1007/s42979-021-00592-x.

[2]	Q. Liu and Y. Wu, “Supervised Learning,” Encycl. Sci. Learn., pp. 3243–3245, 2012, doi: 10.1007/978-1-4419-1428-6_451.

[3]	M. Alloghani, D. Al-Jumeily, J. Mustafina, A. Hussain, and A. J. Aljaaf, “A Systematic Review on Supervised and Unsupervised Machine Learning Algorithms for Data Science,” pp. 3–21, 2020, doi: 10.1007/978-3-030-22475-2_1.

[4]	Brownlee J., “Supervised and Unsupervised Machine Learning Algorithms,” Machine Learning Mastery Pty. Ltd. pp. 1–9, 2019, Accessed: Dec. 15, 2021. [Online]. Available: https://machinelearningmastery.com/supervised-and-unsupervised-machine-learning-algorithms/.

[5]	J. VanderPlas, Python Data Science Handbook | Python Data Science Handbook. 2016.

[6]	E. Zouganeli, V. Tyssø, B. Feng, K. Arnesen, and N. Kapetanovic, “Project-based learning in programming classes - The effect of open project scope on student motivation and learning outcome,” in IFAC Proceedings Volumes (IFAC-PapersOnline), Jan. 2014, vol. 19, no. 3, pp. 12232–12236, doi: 10.3182/20140824-6-za-1003.02412.

[7]	M. A. Almulla, “The Effectiveness of the Project-Based Learning (PBL) Approach as a Way to Engage Students in Learning,” SAGE Open, vol. 10, no. 3, Jul. 2020, doi: 10.1177/2158244020938702.

[8]	E. C. Miller and J. S. Krajcik, “Promoting deep learning through project-based learning: a design problem,” Discip. Interdiscip. Sci. Educ. Res., vol. 1, no. 1, pp. 1–10, Nov. 2019, doi: 10.1186/s43031-019-0009-6.

[9]	D. De Cock, “Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project,” J. Stat. Educ., vol. 19, no. 3, 2011, Accessed: Dec. 15, 2021. [Online]. Available: www.amstat.org/publications/jse/v19n3/decock.pdf.

[10]	H. D. Gangurde, “Feature Selection using Clustering approach for Big Data,” Int. J. Comput. Appl., pp. 975–8887.



