#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# # Loading Data

# In[3]:


df = pd.read_csv("Hotel_Booking.csv")
df


# In[4]:


# Checking Dupicates
df.duplicated().value_counts()


# In[5]:


# Checking number of rows and columns
df.shape


# In[6]:


df.info()


# In[7]:


df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'])


# In[8]:


df.info()


# In[9]:


df.describe(include = 'object').T


# In[10]:


#Categories presents in columns
for col in df.describe (include = 'object').columns:
    print(col)
    print(df[col].unique())
    print('-'*50)


# # Data Cleaning

# In[11]:


df.isnull().sum()


# In[12]:


df.drop(['company', 'agent'], axis = 1, inplace = True)
df.dropna(inplace = True)


# In[13]:


# Checking after cleaning null values
df.isnull().sum()


# In[14]:


df.shape


# # Statistical Summary 

# In[15]:


df.describe()


# In[16]:


df.shape


# In[17]:


# Correlation between numerical column
co = df.corr(min_periods=2, numeric_only=True)
sns.heatmap(co, cmap='coolwarm')
plt.show()


# In[18]:


# Select numerical columns for outlier detection
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns

# Calculate the number of rows needed based on five subplots per row
num_rows = (len(numerical_columns) + 5) // 5

# Set up subplots with specified rows and columns
fig, axes = plt.subplots(nrows=num_rows, ncols=5, figsize=(12, 3 * num_rows))

# Flatten the 2D array of subplots for easier iteration
axes = axes.flatten()

# Iterate through each numerical column
for i, column in enumerate(numerical_columns):
    # Calculate the IQR
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    # Define the lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]

    # Plot boxplot with outliers highlighted
    sns.boxplot(x=df[column], ax=axes[i])
    
# Remove empty subplots if there are any
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Adjust layout for better readability
plt.tight_layout()
plt.show()


# In[19]:


# Select numerical columns for outlier detection
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns

# Iterate through each numerical column
for column in numerical_columns:
    # Calculate the IQR
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    # Define the lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers and remove them in-place
    df.drop(df[(df[column] < lower_bound) | (df[column] > upper_bound)].index, inplace=True)


# In[20]:


df.shape


# # Data Visualization

# In[21]:


#Reservation counts cancelled and not cancelled
print(df['is_canceled'].value_counts())
plt.figure(figsize = (5,4))
plt.title('Reservation status count')
plt.pie(df['is_canceled'].value_counts(), labels=['Not Canceled', 'Canceled'], autopct='%1.1f%%')
plt.show()


# In[22]:


#Checking Reservation Status in Different Hotels
print('Resort Hotel Percentage')
resort_hotel = df[df['hotel'] == 'Resort Hotel']
print(resort_hotel['is_canceled'].value_counts(normalize = True) * 100 )

print('City Hotel Percentage')
city_hotel = df[df['hotel'] == 'City Hotel']
print(city_hotel['is_canceled'].value_counts(normalize = True) * 100 )

plt.figure(figsize = (8,4))
ax = sns.countplot(x = 'hotel', hue = 'is_canceled',data=df, palette= 'Blues')
legend_labels,_ = ax. get_legend_handles_labels()
plt.title('Reservation status in different hotels', size=20)
plt.xlabel('hotel')
plt.ylabel('number of reservations')
plt.legend(['not_canceled','canceled'])
plt.show()


# In[23]:


#Groupby Data
resort_hotel = resort_hotel.groupby('reservation_status_date')[['adr']].mean()
city_hotel = city_hotel.groupby('reservation_status_date')[['adr']].mean()


# In[24]:


#Average Daily Rate in City and Resort Hotel
plt.figure(figsize = (20,10))
plt.title('Average Daily Rate in City and Resort Hotel', fontsize=30)
plt.plot(resort_hotel.index, resort_hotel['adr'], label = 'Resort Hotel')
plt.plot(city_hotel.index, city_hotel['adr'], label = 'City Hotel')
plt.legend(fontsize = 20)
plt.show()


# In[25]:


#Reservation status per month
df['month'] = df['reservation_status_date'].dt.month
plt.figure(figsize = (8,4))
ax1 = sns.countplot(x = 'month', hue = 'is_canceled', data = df, palette = 'bright')
legend_labels,_ = ax1. get_legend_handles_labels()
ax1.legend(bbox_to_anchor=(1,1))
plt.title('Reservation status per month', size = 20)
plt.xlabel('month')
plt.ylabel('number of reservations')
plt.legend(['not canceled', 'canceled'])
plt.show()


# In[26]:


#ADR per month
plt.figure(figsize=(15, 8))
plt.title('ADR per month', fontsize=30)

# Filter the DataFrame for canceled bookings and group by month, summing up the 'adr' values
canceled_df = df[df['is_canceled'] == 1].groupby('month')[['adr']].sum().reset_index()

# Use sns.barplot with x='month' and y='adr', specifying data=canceled_df
sns.barplot(x='month', y='adr', data=canceled_df)

plt.show()


# In[27]:


#Top 10 countries with reservation canceled
cancelled_data = df[df['is_canceled'] == 1]
top_10_country = cancelled_data['country'].value_counts()[:10]
plt.figure(figsize=(7, 7))
plt.title('Top 10 countries with reservation canceled')
plt.pie(top_10_country, autopct='%.2f',labels = top_10_country.index)
plt.show()


# In[28]:


df['market_segment'].value_counts()


# In[29]:


df['market_segment'].value_counts(normalize=True) * 100


# In[30]:


cancelled_data['market_segment'].value_counts(normalize=True) * 100


# In[31]:


#Average daily rate
cancelled_data_adr = cancelled_data.groupby('reservation_status_date')[['adr']].mean()
cancelled_data_adr.reset_index(inplace=True)
cancelled_data_adr.sort_values('reservation_status_date',inplace=True)

not_cancelled_data = df[df['is_canceled'] == 0]
not_cancelled_data_adr = not_cancelled_data.groupby('reservation_status_date')[['adr']].mean()
not_cancelled_data_adr.reset_index(inplace=True)
not_cancelled_data_adr.sort_values('reservation_status_date',inplace=True)

plt.figure(figsize = (20,10))
plt.title('Average Daily Rate', fontsize = 30)
plt.plot(not_cancelled_data_adr['reservation_status_date'],not_cancelled_data_adr['adr'],label='not cancelled')
plt.plot(cancelled_data_adr['reservation_status_date'],cancelled_data_adr['adr'],label='cancelled')
plt.legend(fontsize = 20)
plt.show()


# In[32]:


df.describe()


# In[34]:


a = df.select_dtypes(object).columns
for i in a:
    print (i, df[i].nunique())


# In[35]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'])

df['year'] = df['reservation_status_date'].dt.year
df['month'] = df['reservation_status_date'].dt.month
df['day'] = df['reservation_status_date'].dt.day

df.drop(['reservation_status_date','arrival_date_month'] , axis = 1, inplace = True)


# In[36]:


a = df.select_dtypes(object).columns
cat_list = []
for i in a:
    print (i, df[i].nunique())
    cat_list.append(i)


# In[37]:


for i in cat_list:
    df[i] = le.fit_transform(df[i])
df['year'] = le.fit_transform(df['year'])
df['month'] = le.fit_transform(df['month'])
df['day'] = le.fit_transform(df['day'])


# In[38]:


from sklearn.model_selection import train_test_split
y = df['is_canceled']
X = df.drop('is_canceled', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=101,test_size=0.3)


# In[72]:


X


# In[39]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[73]:


from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


# In[74]:


lr = LogisticRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

acc_lr = accuracy_score(y_test, y_pred_lr)
print(f"Accuracy Score of Logistic Regression is : {acc_lr:.2f}")


# In[75]:


knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)

acc_knn = accuracy_score(y_test, y_pred_knn)

print(f"Accuracy Score of KNN is : {acc_knn}")


# In[76]:


rd_clf = RandomForestClassifier()
rd_clf.fit(X_train, y_train)

y_pred_rd_clf = rd_clf.predict(X_test)

acc_rd_clf = accuracy_score(y_test, y_pred_rd_clf)


print(f"Accuracy Score of Random Forest is : {acc_rd_clf}")


# In[77]:


models = pd.DataFrame({
    'Model' : ['Logistic Regression', 'KNN', 'Random Forest Classifier'],
    'Score' : [acc_lr, acc_knn, acc_rd_clf]
})
models.sort_values(by='Score', ascending=False)


# In[78]:


plt.barh(models['Model'], models['Score'], color='blue')
plt.xlabel('Score')
plt.ylabel('Model')
plt.title('Models Comparison')
plt.show()


# In[ ]:




