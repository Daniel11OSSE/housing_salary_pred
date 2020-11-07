# Machine Learning median house pricing prediction : Project training

**This project is an application of Machine Learning training from the book Hands-on Machine Learning with Scikit-Learn, Keras and TensorFlow of Aurélien Géron which aims to predict the median housing price of a district in California. The solution obtained could be fed to another machine learning system, along with other signals to determine if it is worth investing in a given area or not. Supervised learning algorithms like Linear Regression, DecisionTree and RandomForest will be used for realizing the project whose performance will be measured by RMSE & MAE.**

* Created a tool that estimates the median price of a house in California (RMSE ~ $ 47,730.2).
* Optimized Linear Regression, Decision Tree Regressor, Random Forest Regressors, Support Vector Regressor using GridsearchCV and RandomizedSearchCV to reach the best model.  

## Code and Resources Used 
**Python Version:** 3.7  
**Packages:** pandas, numpy, sklearn, matplotlib, seaborn
**For Web Framework Requirements:**  ```pip install -r requirements.txt```    
**handson_ml book by Aurélien Géron code :** https://github.com/ageron/handson-ml/blob/master/02_end_to_end_machine_learning_project.ipynb

## EDA
Looking at the distribution of homes in the State of California, we found that most homes are located around the Bay, in Los Angeles, in San Diego, plus a long line of fairly high density in the Central Valley, particularly around Sacramento and Fresno. 
Also, the housing prices are very much related to the location (e.g., close to the ocean) and to the population density
Below are a few highlights. 

![alt text](https://github.com/Daniel11OSSE/housing_salary_pred/blob/master/California.png "Population by District")
![alt text](https://github.com/Daniel11OSSE/housing_salary_pred/blob/master/correlation.png "Correlation between median housing price and median income")
![alt text](https://github.com/Daniel11OSSE/housing_salary_pred/blob/master/Correlation_many.PNG "Correlations with median housing price")

## Model Building 
### Data Cleaning
*	impute missing values with the median 
* get dummy varibles for the categorical variable "Ocean proximity"
* create 3 others variables :
  *rooms_per_household
  *bedrooms_per_room
  *population_per_household

After combining everything in a transformer for the different models,

I tried three different models and evaluated them using Root Mean Squared Error.   

I first tried three different models before adding at the end of the notebook a fitting with support vector Regressor:
*	**Multiple Linear Regression**
*	**Decision Tree Regressor**
*	**Random Forest**

## Model performance
The Random Forest model far outperformed the other approaches on the test and validation sets. 
*	**Decision Tree Regressor** : RMSE = 70992
*	**Linear Regression**: RMSE = 68628
*	**Random Forest**: **RMSE = 47,730.2**
*	**Support Vector Regressor**: RMSE = 70445
