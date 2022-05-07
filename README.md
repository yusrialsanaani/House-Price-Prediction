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

Data preprocessing is an extremely important step in machine learning to enhance the quality of data to promote the extraction of meaningful insights from the data. Data preprocessing in machine learning refers to the technique of preparing (cleaning and organizing) the raw data to make it suitable for a building and training machine learning models. The steps in data preprocessing pipeline are shown in Figure ‎1 3, which includes:
- Outlier detection using visual method and interquartile range rule (IQR) technique.
- Identifying and handling the missing values
- Variable transformation using:
  - Log for target variable
  - Scaling (StandardScaler) for other numerical variables.
- Handling categorical variables by using dummy variables and encoding.



