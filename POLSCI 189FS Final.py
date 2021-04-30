#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score


# In[4]:



df1 = pd.read_csv(r'/Users/jimmygao/Desktop/Duke Freshman Year/Fall Semester/POLSCI189FS/final_data2.csv')

print (df1.iloc[:3])
print (df1.describe())


# In[5]:


df2 = df1.dropna()
print (df2.describe())


# In[6]:


vars = ['minwage', 'welfare', 'uniform', 'gaym', 'income', 'gender', 'fp', 'edu']
x = df2.loc[:, vars].values
y = df2.loc[:, 'obama'].values


# In[7]:


x_norm = StandardScaler().fit_transform(x)


# In[8]:


dim = 8
pca1 = PCA(n_components=dim)
latent_vars = pca1.fit_transform(x_norm)


# In[9]:


print ("Variance explained by each latent variable in PCA: ", pca1.explained_variance_ratio_)
print ("\n")

for i in range(0,8):
    print ("x",i,": ", end='')
    for j in range(0,dim):
        print (round(pca1.components_[j][i],4), ", ", end='')
    print ("\n")


# In[10]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
vif_data = pd.DataFrame()
vif_data["feature"] = vars
vif_data["VIF"] = [variance_inflation_factor(x, i)
for i in range(len(vars))]
print (vif_data)


# In[11]:


np.corrcoef(df2['edu'], df2['income'])


# In[12]:


vars2 = ['minwage', 'welfare', 'income']
temp = df2.loc[:, vars2].values
pca2 = PCA(n_components=2)

latent_vars = pca2.fit_transform(temp)

print ("Variance explained by each latent variable in PCA: ", pca2.explained_variance_ratio_)
print ("\n")


# In[13]:


print(temp[2])
df2['welfare']


# In[14]:


df2['pca2'] = latent_vars[:,0]
x_norm = np.append(x_norm,latent_vars,1)


# In[15]:


df2.loc[:, ['welfare','pca2']]
print(pca2.components_)
print(pca2.singular_values_)


# In[16]:


IVs = ['pca2','gaym','edu','gender','fp']

from sklearn.model_selection import KFold

X_train, X_test, y_train, y_test = train_test_split(df2.loc[:, IVs], df2.loc[:, 'obama'], test_size=0.2)
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)


# In[19]:


# testing linear and polynomial of different degrees
linear_model = LinearRegression(normalize=True)
p2_model = LinearRegression(normalize=True)
p3_model = LinearRegression(normalize=True)


p2_features = PolynomialFeatures(degree=2)
p2_train = p2_features.fit_transform(X_train)
p2_test = p2_features.fit_transform(X_test)

p3_features = PolynomialFeatures(degree=3)
p3_train = p3_features.fit_transform(X_train)
p3_test = p3_features.fit_transform(X_test)


lin_1 = linear_model.fit(X_train, y_train)
p2_1 = p2_model.fit(p2_train, y_train)
p3_1 = p3_model.fit(p3_train, y_train)



# In[20]:


lin1_predict = lin_1.predict(X_test)
p2_predict = p2_1.predict(p2_test)
p3_predict = p3_1.predict(p3_test)


# In[21]:


print("Linear K-Fold:", cross_val_score(linear_model, df2.loc[:, IVs], df2.loc[:, 'obama'], cv=10))
print("Poly 2 K-Fold:", cross_val_score(p2_model, p2_train, y_train, cv=10))
print("Poly 3 K-Fold:", cross_val_score(p3_model, p3_train, y_train, cv=10))


# In[22]:



print (len(lin_1.coef_))
print (len(p2_1.coef_))
print (len(p3_1.coef_))
print (lin_1.coef_)
print (lin_1.intercept_)
print (X_train.columns)


# In[23]:



print ("linear train / test rmse: ", mean_squared_error(y_train, lin_1.predict(X_train))**(.5), " / ", mean_squared_error(y_test, lin1_predict)**(.5))
print ("poly degree 2 train / test rmse: ", mean_squared_error(y_train, p2_1.predict(p2_train))**(.5), " / ", mean_squared_error(y_test, p2_predict)**(.5))
print ("poly degree 3 train / test rmse: ", mean_squared_error(y_train, p3_1.predict(p3_train))**(.5), " / ", mean_squared_error(y_test, p3_predict)**(.5))

print ("linear train / test r^2: ", r2_score(y_train, lin_1.predict(X_train)), " / ", r2_score(y_test, lin1_predict))
print ("poly degree 2 train / test r^2: ", r2_score(y_train, p2_1.predict(p2_train)), " / ", r2_score(y_test, p2_predict))
print ("poly degree 3 train / test r^2: ", r2_score(y_train, p3_1.predict(p3_train)), " / ", r2_score(y_test, p3_predict))


# In[24]:


for i in range(0, len(p2_features.get_feature_names())):
    print (p2_features.get_feature_names(X_train.columns)[i], ", ", (p2_1.coef_)[i])


# In[719]:





# In[25]:


#lasso regularized regression test values
lambdas = (.1, .5, 1, 2.5, 5, 7.5, 10, 20, 50, 100, 200)

for i in lambdas:    
    lasso_reg = Lasso(alpha = i, max_iter=10000)
    lasso1 = lasso_reg.fit(p2_train, y_train)
    lasso1_predict = lasso1.predict(p2_test)
    print (mean_squared_error(y_test, lasso1_predict)**(.5))


# In[26]:


lasso_reg = Lasso(alpha = 5, max_iter=10000)
lasso1 = lasso_reg.fit(p2_train, y_train)
lasso1_predict = lasso1.predict(p2_test)
print (mean_squared_error(y_test, lasso1_predict)**(.5))


# In[43]:


print("Intercept:", p2_1.intercept_)
for i in range(0, len(p2_features.get_feature_names())):
    print (p2_features.get_feature_names(X_train.columns)[i], ", ", (lasso1.coef_)[i])


# In[44]:


for i in range(0, len(p2_features.get_feature_names())):
    if (abs(lasso1.coef_[i])) > 0.05:
        print (p2_features.get_feature_names(X_train.columns)[i], ", ", (lasso1.coef_)[i])


# In[46]:


# retesting interaction terms for any hidden interactions
df2['gaymfp'] = df2.loc[:, 'gaym'] * df2.loc[:, 'fp']
df2['gaym^2'] = df2.loc[:, 'gaym'] * df2.loc[:, 'gaym']
df2['gaymgender'] = df2.loc[:, 'gaym'] * df2.loc[:, 'gender']
df2['gaymedu'] = df2.loc[:, 'gaym'] * df2.loc[:, 'edu']
df2['pca2^2'] = df2.loc[:, 'pca2'] * df2.loc[:, 'pca2']
df2['pca2gender'] = df2.loc[:, 'pca2'] * df2.loc[:, 'gender']
df2['pca2fp'] = df2.loc[:, 'pca2'] * df2.loc[:, 'fp']
df2['edufp'] = df2.loc[:, 'edu'] * df2.loc[:, 'fp']
df2['edupca2'] = df2.loc[:, 'edu'] * df2.loc[:, 'pca2']
df2['pca2gaym'] = df2.loc[:, 'pca2'] * df2.loc[:, 'gaym']
df2['genderfp'] = df2.loc[:, 'gender'] * df2.loc[:, 'fp']
df2['gender^2'] = df2.loc[:, 'gender'] * df2.loc[:, 'gender']
df2['fp^2'] = df2.loc[:, 'fp'] * df2.loc[:, 'fp']
df2['edu^2'] = df2.loc[:, 'edu'] * df2.loc[:, 'edu']

x_final = ['pca2','gaym', 'edu', 'fp','edu^2', 'genderfp', 'gaymgender', 'edupca2']
y_final = ['obama']

x_train_f, x_test_f, y_train_f, y_test_f = train_test_split(df2.loc[:, x_final], df2.loc[:, y_final], test_size=0.33)



# In[47]:


model = LinearRegression().fit(x_train_f, y_train_f)
y_pred = model.predict(x_test_f)
r_sq = model.score(x_test_f, y_test_f)
print(r_sq)


# In[48]:


print("intercept:", model.intercept_)
print(x_final[0], ", ", (model.coef_)[0])


# In[49]:





# In[42]:





# In[624]:


for i in range(0, len(x_final)):
    print (x_final[i], ", ", (model.coef_)[i])
print("intercept:", model.intercept_)


# In[ ]:




