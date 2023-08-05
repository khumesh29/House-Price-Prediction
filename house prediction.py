#!/usr/bin/env python
# coding: utf-8

# # House Price Prediction ##

# ## Context 

# The price of the house depends on various factors like locality, connectivity, number of rooms, etc. Change in the mindset of the millennial generation also contributes to ups and down in house prices as the young generation is much into renting than to owe a house. Predicting the right price of the house is important for investors in the real estate business. This makes it very important to come up with proper and smart technique to estimate the true price of the house.

# ## Problem Statement

# You are willing to sell your house. You are not sure about the price of your house and want to estimate its price. You are provided with the dataset and need to make a prediction model which will help you to get a good estimate of your house for selling it.

# ## Data Description

# The housing dataset contains the prices and other attributes. There are 
#  rows and 
#  attributes (features) with a target column (price).
# 
# Following are the features:
# 
# 
Column	     :Description
Price	     :Price in INR
area	     :Area in square ft.
bedrooms	 :Number of bedrooms in the house
bathrooms	 :Number of bathrooms in the house
stories	     :Number of stores in the house
mainroad	 :Whether house is on main road or not(binary)
guestroom	 :Whether house have guestroom or not(binary)
basement	 :Whether house have basement or not(binary)
airconditioning	:Whether house have airconditioning or not(binary)
hotwaterheating	:Whether house have hotwaterheating or not(binary)
parking	      :Number of parking area
prefarea	  :Whether house have prefarea or not(binary)
furnishingstatus :Furnish status of the house
# ## things to do

# 1. Convert categorical attributes into numerical attributes using feature encoding.
# 2. Explore the Housing dataset by creating the following plots:
# 
#   * histogram of each feature.
#   * heat map of correlation between each and every features.
#   * making normal distribution curve of price
# 
# 3. Build a linear regression model and Random forest model by selecting the most relevant features to predict the price of houses.
# 
# 4. Evaluate the model by calculating the parameters such as coefficient of determination, MAE, MSE, RMSE, mean of residuals and then compare the best model accoring to r^2 score

# ## 1. Import Modules and Load Dataset

# In[2]:


# Import the required modules and load the dataset.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the dataset
df = pd.read_csv("house-prices (1).csv")
df.head(5)


# In[3]:


# Get the information on DataFrame.
df.info()


# In[4]:


# Check if there are any NULL values.
df.isnull().sum()


# In[5]:


# Check categorical attributes
df_cat=df.select_dtypes(['object'])
df_cat.head()


# In[ ]:





# ## 2. Feature encoding

# Perform feature encoding using map() function and one-hot encoding.

# In[6]:


# Replace yes with 1 and no with 0 for all the values in features 'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea' using map() function.

a=['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
def binary_map(x):
    return x.map({'yes':1,'no':0})
df[a]=df[a].apply(binary_map)


# In[7]:


df['furnishingstatus'].unique()


# In[8]:


# Replace furnishingstatus with 0, 1 and 2
furnished = {'furnished':0, 'semi-furnished':1,'unfurnished':2 }
df['furnishingstatus'] = df['furnishingstatus'].map(furnished)


# In[9]:


# Print dataframe
df.head()


# ## 3. EDA

# In[11]:


# Create a normal distribution curve for the 'price'.

# Create a probablity density function for plotting the normal distribution
def prob_density_func(series):
  CONST = 1 / (series.std() * np.sqrt(2 * np.pi))
  power_of_e = - (series - series.mean()) ** 2 / (2 * series.var()) # 'pd.Series.var()' function returns the variance of the series.
  new_array = CONST * np.exp(power_of_e)
  return new_array


# Plot the normal distribution curve using plt.scatter()
print(df['price'].mean())
plt.figure(figsize=[15,5])
plt.scatter(df['price'],prob_density_func(df['price']))
plt.title('Normal Distribution Curve for Price')
plt.legend()
plt.show() 


# In[12]:


# finding the correlation between price and features
df.corr()['price']


# In[13]:


#Construct heat map for clean data

plt.figure(figsize=(12,10))
p=sns.heatmap(df.corr(), annot=True,cmap ='RdYlGn')


# In[14]:


# removing the 'hotwaterheating' feature as it do not effect the independent variable as much.

df = df.drop('hotwaterheating', axis=1)


# ## 4. Model Building and Evaluation

# Build a multiple linear regression model using the statsmodels.api module.

# In[15]:


# creating variables of feature and target variable

y = df["price"]
X = df.loc[:, df.columns !="price"]


# In[16]:


# Split the 'df' Dataframe into the train and test sets.
from sklearn.model_selection import train_test_split


# Split the DataFrame into the train and test sets such that test set has 25% of the values.

Xtrain, Xtest, ytrain, ytest = train_test_split(X,y, test_size = 0.25, random_state = 42)


# ## Linear Regression

# In[17]:


# Importing library for linear regression model
from sklearn.linear_model import LinearRegression


# In[18]:


lr= LinearRegression()
lr.fit(Xtrain, ytrain)


# In[19]:


# Value of y intercept
lr.intercept_


# In[20]:


#Converting the coefficient values to a dataframe
coeffcients = pd.DataFrame([Xtrain.columns,lr.coef_]).T
coeffcients = coeffcients.rename(columns={0: 'Attribute', 1: 'Coefficients'})
coeffcients


# In[26]:


# Model prediction on train data
y_pred = lr.predict(Xtrain)


# In[21]:


# Evaluate the linear regression model using the 'r2_score', 'mean_squared_error' & 'mean_absolute_error' functions of the 'sklearn' module.
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
y_test_pred = lr.predict(Xtest)
y_train_pred = lr.predict(Xtrain)

print(f"\n\nTrain Set\n{'-' * 50}")
print(f"R-squared: {r2_score(ytrain, y_train_pred):.6f}")
print(f"Mean Squared Error: {mean_squared_error(ytrain, y_train_pred):.3f}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(ytrain, y_train_pred)):.3f}")
print(f"Mean Absolute Error: {mean_absolute_error(ytrain, y_train_pred):.3f}")


# 𝑅^2 : It is a measure of the linear relationship between X and Y. It is interpreted as the proportion of the variance in the dependent variable that is predictable from the independent variable.
# 
# MAE : It is the mean of the absolute value of the errors. It measures the difference between two continuous variables, here actual and predicted values of y. 
# 
# MSE: The mean square error (MSE) is just like the MAE, but squares the difference before summing them all instead of using the absolute value. 
# 
# RMSE: The mean square error (MSE) is just like the MAE, but squares the difference before summing them all instead of using the absolute value. 

# In[22]:


# Visualizing the differences between actual prices and predicted values
plt.scatter(ytrain, y_pred)
plt.xlabel("prices")
plt.ylabel("Predicted prices")
plt.title("Prices vs Predicted prices")
plt.show()


# In[23]:


# Checking residuals
plt.scatter(y_pred,ytrain-y_pred)
plt.title("Predicted vs residuals")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.show()


# There is no pattern visible in this plot and values are distributed equally around zero. So Linearity assumption is satisfied

# In[24]:


# Checking Normality of errors
sns.distplot(ytrain-y_pred)
plt.title("Histogram of Residuals")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()


# Here the residuals are normally distributed. So normality assumption is satisfied

# In[25]:


# Model Evaluation

print(f"\n\nTest Set\n{'-' * 50}")
lr_r2 = (f"R-squared: {r2_score(ytest, y_test_pred):.3f}")
print(lr_r2)
print(f"Mean Squared Error: {mean_squared_error(ytest, y_test_pred):.3f}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(ytest, y_test_pred)):.3f}")
print(f"Mean Absolute Error: {mean_absolute_error(ytest, y_test_pred):.3f}")


# ## Random Forest Regressor

# In[26]:


# Import Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor

# Create a Random Forest Regressor
reg = RandomForestRegressor()

# Train the model using the training sets 
reg.fit(Xtrain, ytrain)


# In[27]:


# Model prediction on train data
y_pred = reg.predict(Xtrain)


# In[28]:


y_test_pred = reg.predict(Xtest)
y_train_pred = reg.predict(Xtrain)

print(f"\n\nTrain Set\n{'-' * 50}")
print(f"R-squared: {r2_score(ytrain, y_train_pred):.6f}")
print(f"Mean Squared Error: {mean_squared_error(ytrain, y_train_pred):.3f}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(ytrain, y_train_pred)):.3f}")
print(f"Mean Absolute Error: {mean_absolute_error(ytrain, y_train_pred):.3f}")


# In[29]:


# Visualizing the differences between actual prices and predicted values
plt.scatter(ytrain, y_pred)
plt.xlabel("prices")
plt.ylabel("Predicted prices")
plt.title("Prices vs Predicted prices")
plt.show()


# In[30]:


# Checking residuals
plt.scatter(y_pred,ytrain-y_pred)
plt.title("Predicted vs residuals")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.show()


# #### for test data

# In[31]:


# Model Evaluation

print(f"\n\nTest Set\n{'-' * 50}")
reg_r2 = (f"R-squared: {r2_score(ytest, y_test_pred):.3f}")
print(reg_r2)
print(f"Mean Squared Error: {mean_squared_error(ytest, y_test_pred):.3f}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(ytest, y_test_pred)):.3f}")
print(f"Mean Absolute Error: {mean_absolute_error(ytest, y_test_pred):.3f}")


# ## Evaluation and comparision of all the models

# In[32]:


models = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest'],
    'R-squared Score': [lr_r2*100, reg_r2*100]})
models.sort_values(by='R-squared Score', ascending=False)


# ### Hence Linear Regression works the best for this dataset.** 

# In[ ]:




