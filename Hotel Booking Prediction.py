#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # Loading Data

# In[2]:


df = pd.read_csv("Hotel_Booking.csv")
df


# In[3]:


# Checking Dupicates
df.duplicated().value_counts()


# In[4]:


# Checking number of rows and columns
df.shape


# In[5]:


df.info()


# In[6]:


#change the datatype to datetime
df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'])


# In[7]:


df.info()


# In[8]:


df.describe(include = 'object').T


# In[9]:


#Categories presents in columns
for col in df.describe (include = 'object').columns:
    print(col)
    print(df[col].unique())
    print('-'*50)


# # Data Cleaning

# In[10]:


df.isnull().sum()


# In[11]:


df.drop(['company', 'agent'], axis = 1, inplace = True)
df.dropna(inplace = True)


# In[12]:


# Checking after cleaning null values
df.isnull().sum()


# In[13]:


df.shape


# # Statistical Summary 

# In[14]:


df.describe()


# # EDA - Exploratory Data Analysis

# In[15]:


# Correlation between numerical column
co = df.corr(min_periods=2, numeric_only=True)
sns.heatmap(co, cmap='coolwarm')
plt.show()


# # Data Visualization

# In[16]:


#Reservation counts cancelled and not cancelled
print(df['is_canceled'].value_counts())
plt.figure(figsize = (5,4))
plt.title('Reservation status count')
plt.pie(df['is_canceled'].value_counts(), labels=['Not Canceled', 'Canceled'], autopct='%1.1f%%')
plt.show()


# In[17]:


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


# In[18]:


#Groupby Data
resort_hotel = resort_hotel.groupby('reservation_status_date')[['adr']].mean()
city_hotel = city_hotel.groupby('reservation_status_date')[['adr']].mean()


# In[19]:


#Average Daily Rate in City and Resort Hotel
plt.figure(figsize = (20,10))
plt.title('Average Daily Rate in City and Resort Hotel', fontsize=30)
plt.plot(resort_hotel.index, resort_hotel['adr'], label = 'Resort Hotel')
plt.plot(city_hotel.index, city_hotel['adr'], label = 'City Hotel')
plt.legend(fontsize = 20)
plt.show()


# In[20]:


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


# In[21]:


#ADR per month
plt.figure(figsize=(15, 8))
plt.title('ADR per month', fontsize=30)

# Filter the DataFrame for canceled bookings and group by month, summing up the 'adr' values
canceled_df = df[df['is_canceled'] == 1].groupby('month')[['adr']].sum().reset_index()

# Use sns.barplot with x='month' and y='adr', specifying data=canceled_df
sns.barplot(x='month', y='adr', data=canceled_df)

plt.show()


# In[22]:


#Top 10 countries with reservation canceled
cancelled_data = df[df['is_canceled'] == 1]
top_10_country = cancelled_data['country'].value_counts()[:10]
plt.figure(figsize=(7, 7))
plt.title('Top 10 countries with reservation canceled')
plt.pie(top_10_country, autopct='%.2f',labels = top_10_country.index)
plt.show()


# In[23]:


df['market_segment'].value_counts()


# In[24]:


df['market_segment'].value_counts(normalize=True) * 100


# In[25]:


cancelled_data['market_segment'].value_counts(normalize=True) * 100


# In[26]:


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


# # Conclusion

# ### 1. Cancellation rates rise as the price does. In order to prevent cancellations of reservations, hotels could work on their pricing strategies and try to lower the rates for specific hotels based on locations. They can also provide some discounts to the consumers.
# 
# ### 2. As the ratio of the cancellation and not cancellation of the resort hotel is higher in the resort hotel than the city hotels. So the hotels should provide a reasonable discount on the room prices on weekends or on holidays.
# 
# ### 3. In the month of January, hotels can start campaigns or marketing with a reasonable amount to increase their revenue as the cancellation is the highest in this month.
# 
# ### 4. They can also increase the quality of their hotels and their services mainly in Portugal to reduce the cancellation rate.

# In[ ]:




