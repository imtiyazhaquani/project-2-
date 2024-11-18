#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import statsmodels.api as sm
from scipy import stats
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn import metrics

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize'] = (10,5)
plt.rcParams['figure.dpi'] = 250
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
import lightgbm as lgb
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,roc_auc_score,roc_curve
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


books = pd.read_csv('Books.csv', encoding='latin-1')
ratings = pd.read_csv('Ratings.csv', encoding='latin-1')
users= pd.read_csv('Users.csv', encoding='latin-1')


# In[9]:


books.head()


# In[10]:


books.shape


# In[11]:


ratings.head()


# In[12]:


ratings.shape


# In[13]:


users.head()


#  DATA CLEANING
#  CHECKING BOOKS DATA

# In[15]:


books.info()


# In[16]:


books.dtypes


# In[18]:


books.isnull().sum()


# In[19]:


books['Book-Author'] = books['Book-Author'].fillna('Unknown')

books['Publisher'] = books['Publisher'].fillna('Unknown')

books['Image-URL-S'] = books['Image-URL-S'].fillna('Not Available')
books['Image-URL-M'] = books['Image-URL-M'].fillna('Not Available')
books['Image-URL-L'] = books['Image-URL-L'].fillna('Not Available')

books.isnull().sum()


# In[20]:


for col in books.columns:
    print(f"Number of {col} is {books[col].nunique()}")


# In[21]:


# droping the url
books.drop(['Image-URL-S', 'Image-URL-M', 'Image-URL-L'], axis= 1, inplace= True)


# In[22]:


# Extracting and fixing mismatch in feature 'year_of_publication', 'publisher', 'book_author', 'book_title'
books[books['Year-Of-Publication'] == 'DK Publishing Inc']


# In[23]:


books[books['Year-Of-Publication'] == 'Gallimard']


# In[24]:


def replace_df_value(df, idx, col_name, val):
    df.loc[idx, col_name] = val
    return df


# In[25]:


replace_df_value(books, 209538, 'Book-Title', 'DK Readers: Creating the X-Men, How It All Began (Level 4: Proficient Readers)')
replace_df_value(books, 209538, 'Book-Author', 'Michael Teitelbaum')
replace_df_value(books, 209538, 'Year-Of-Publication', 2000)
replace_df_value(books, 209538, 'Publisher', 'DK Publishing Inc')

replace_df_value(books, 221678, 'Book-Title', 'DK Readers: Creating the X-Men, How Comic Books Come to Life (Level 4: Proficient Readers)')
replace_df_value(books, 221678, 'Book-Author', 'James Buckley')
replace_df_value(books, 221678, 'Year-Of-Publication', 2000)
replace_df_value(books, 221678, 'Publisher', 'DK Publishing Inc')

replace_df_value(books, 220731,'Book-Title', "Peuple du ciel, suivi de 'Les Bergers")
replace_df_value(books, 220731, 'Book-Author', 'Jean-Marie Gustave Le ClÃ?Â©zio')
replace_df_value(books, 220731, 'Year-Of-Publication', 2003)
replace_df_value(books, 220731, 'Publisher', 'Gallimard')


# In[26]:


books.loc[209538]


# In[27]:


books.loc[221678]


# In[28]:


books.loc[220731]


# In[29]:


books.isnull().sum()


# CHECKING USERS DATA

# In[30]:


users.head(3)


# In[31]:


users.shape


# In[32]:


users.info()


# In[33]:


# unique value in age
users['Age'].unique()


# In[34]:


users.isnull().sum()


# In[35]:


# replacing nan with average of 'age'
users['Age'].fillna((users['Age'].mean()), inplace=True)


# In[36]:


users['Age'].unique()


# In[37]:


# retrieving age data between 5 to 90
users.loc[(users['Age'] > 90) | (users['Age'] < 5)] = np.nan


# In[38]:


users['Age'].fillna((users['Age'].mean()), inplace=True)
users['Age'].unique()


# CHECKING RATINGS DATA

# In[39]:


ratings.head()


# In[40]:


ratings.shape


# In[41]:


ratings.info()


# In[42]:


# finding unique ISBNs from rating and book dataset
unique_ratings = ratings[ratings.ISBN.isin(books.ISBN)]
unique_ratings


# In[43]:


print(ratings.shape)
print(unique_ratings.shape)


# In[44]:


# unique ratings from 'book_rating' feature
unique_ratings['Book-Rating'].unique()


# DATA VISUALIZATION /
# BOOKS DATASET
# 

# In[45]:


# Top Authors with no.of books

plt.figure(figsize=(12,6))
sns.countplot(y="Book-Author",palette = 'Paired', data=books,order=books['Book-Author'].value_counts().index[0:20])
plt.title("Top 20 author with number of books")


# In[46]:


#Top publishers with published books

plt.figure(figsize=(12,6))
sns.countplot(y="Publisher",palette = 'Paired', data=books,order=books['Publisher'].value_counts().index[0:20])
plt.title("Top 20 Publishers with number of books published")


# In[47]:


#no.of books published yearly

publications = {}
for year in books['Year-Of-Publication']:
    if str(year) not in publications:
        publications[str(year)] = 0
    publications[str(year)] +=1

publications = {k:v for k, v in sorted(publications.items())}

fig = plt.figure(figsize =(55, 15))
plt.bar(list(publications.keys()),list(publications.values()), color = 'blue')
plt.ylabel("Number of books published")
plt.xlabel("Year of Publication")
plt.title("Number of books published yearly")
plt.margins(x = 0)
plt.show()


# In[48]:


zero_year_rows = books[books['Year-Of-Publication'] == 0]

# Get the number of rows where 'Year-Of-Publication' is 0
num_zero_years = len(zero_year_rows)
print(f"\nNumber of books with 'Year-Of-Publication' equal to 0: {num_zero_years}")


# In[49]:


# Replace 0 in 'Year-Of-Publication' with NaN
books['Year-Of-Publication'] = books['Year-Of-Publication'].replace(0, np.nan)

# Verify the replacement
zero_year_rows = books[books['Year-Of-Publication'] == 0]
print(zero_year_rows)

num_zero_years = len(zero_year_rows)
print(f"\nNumber of books with 'Year-Of-Publication' equal to 0: {num_zero_years}")


# In[50]:


books['Year-Of-Publication'] = pd.to_numeric(books['Year-Of-Publication'], errors='coerce')

year = books['Year-Of-Publication'].value_counts().sort_index()
year = year.where(year>5)
plt.figure(figsize=(6, 4))
plt.rcParams.update({'font.size': 4})
plt.bar(year.index, year.values)
plt.xlabel('Year of Publication')
plt.ylabel('counts')
plt.show()


# So we can see publication years are somewhat between 1950 - 2005 here.The publication of books got vital when it starts emerging from 1950. We can get some hyothesis key points:-
# 
# It might happen people starts to understand the importance of books and gradually got productivity habits in their life.
# Every user has their own taste to read books based on what particular subject Author uses. The subject of writing books got emerge from late 1940 slowly. Till 1970 it has got the opportunity to recommend books to people or users what they love to read.
# The highest peak we can observe is between 1995-2001 year. The user understand what they like to read. Looking towards the raise the recommendation is also increase to understand their interest.

# USER DATASET

# In[51]:


#Age distribution

plt.figure(figsize=(4,2))
users.Age.hist(bins=[10*i for i in range(1, 10)], color = 'cyan')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()


# Looking towards the users age between 30-40 prefer more and somewhat we can also view between 20-30. Let's make some hypothesis analysis:-
# 
# It is obvious that most of the user books are from Age 30 to 40.
# It might happen that the users are more interested on that subject what Authors are publishing in the market.
# The age group between 20-30 are immensely attracted to read books published by Author.
# We can observe same pitch for Age group between 10-20 and 50-60. There are can be lot of different reasons.

# RATINGS DATASET

# In[52]:


#Top 20 Books as per no. of ratings

plt.figure(figsize=(4,2))
sns.countplot(y="Book-Title",palette = 'Paired',data= books, order=books['Book-Title'].value_counts().index[0:15])
plt.title("Top 20 books as per number of ratings")


# As per ratings "Selected Poems" has been rated most followed by "Little Women".
# 
# Selected Poems are most favourable to users as per ratings.
# Three of the books 'The Secret Garden', 'Dracula','Adventures of Huckleberry Finn'are struggling to compete with each other. Similarly, we can observe in 'Masquerade','Black Beauty','Frankenstein'.

# In[53]:


plt.figure(figsize=(4,2))
sns.countplot(x="Book-Rating",palette = 'Paired',data= unique_ratings)


# Firstly the above ratings are unique ratings from 'ratings_data' and 'books_data' dataset. We have to separate the explicit ratings represented by 1–10 and implicit ratings represented by 0. Let's make some hypothesis assumptions :-
# 
# This countplot shows users have rated 0 the most, which means they haven't rated books at all.
# Still we can see pattern to recognize in ratings from 1-10.
# Mostly the users have rated 8 ratings out of 10 as per books. It might happen that the feedback is positive but not extremely positive as 10 ratings (i.e best books ever).

# In[54]:


# Explicit Ratings
plt.figure(figsize=(4,2))
rate_data = unique_ratings[unique_ratings['Book-Rating'] != 0]
sns.countplot(x="Book-Rating",palette = 'Paired',data=rate_data)
plt.title("Explicit Ratings")


# Now this countplot of bookRating indicates that higher ratings are more common amongst users and rating 8 has been rated highest number of times. There can be many assumptions based on ratings of users :-
# 
# Let's take ratings group from 1-4. This can be negative impact for books been published if they have ratings from 1 to 4. It can be issues related to - 1. Language 2. Offend by any chapter's incident/paragraph/Author 3. They've read worst book ever.
# 
# If we think analytical about rating 5, it might happen some same reason as above key points mention.
# 
# For 5 ratings the users might not sure about book ratings whether it's positive or negative impact.
# 
# Let's take ratings group from 6-10. This are positive feedback - 1. It can happen that not every book is perfect in all desire. So, the user's have decided to rate 8.
# 2. Since 6 ratings is very low among other ratings. 3. As we can aspect 7 and 8 are average and more ratings from users. 4. 9 and 10 ratings are top best ratings based on Author's, Publisher's and Books been published.

# In[55]:


# average rating per user
av_rating_user = ratings.groupby('User-ID')['Book-Rating'].mean().reset_index()


# In[56]:


plt.figure(figsize=(4,2))
sns.histplot(data=av_rating_user, x='Book-Rating', color='#0047AB')
plt.title('Distribution of average ratings per user', weight='bold', fontsize=8)
plt.xlabel('Average ratings given', fontsize=8)
plt.ylabel('Number of user', fontsize=8)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.show()


# A little bit different from above. Here, we tried to understand the average rating given by user.
# 
# There are few low ratings observed and the peak seems to be at 0.

# In[57]:


# agregate ratings by user
book_per_user = ratings.groupby('User-ID')['Book-Rating'].count().reset_index()

# visualize
plt.figure(figsize=(4,2))
sns.distplot(book_per_user['Book-Rating'], color='#0047AB')
plt.title('Number of books that a person usually review', weight='bold', fontsize=8)
plt.xlabel('Number of books reviewed', fontsize=8)
plt.ylabel('Density', fontsize=8)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.show()


# RECOMMENDATION SYSTEM 

# Now let's try to build recommendation system based on popularity (i.e ratings). This recommendations are usually given to every user irrespective of personal characterization.
# 
# We have to separate the explicit ratings represented by 1–10 and implicit ratings represented by 0.

# In[58]:


ratings_explicit= unique_ratings[unique_ratings['Book-Rating'] != 0]   # explicit ratings represented by 1–10
ratings_implicit= unique_ratings[unique_ratings['Book-Rating'] == 0]   # implicit ratings represented by 0


# In[59]:


ratings_explicit.head()


# In[60]:


print(unique_ratings.shape)
print(ratings_explicit.shape)


# In[61]:


# Merging book_data dataset and ratings_explicit
new_book_df= pd.merge(books, ratings_explicit, on='ISBN')
new_book_df.head()


# In[62]:


print(new_book_df.shape)


# In[63]:


new_book_df['Book-Title'].count()


# In[64]:


new_book_df['Book-Title'].nunique()


# LETS TAKE TOP 10 RECOMMENDATION BOOKS

# In[65]:


# top ten books as per book ratings and recommendation
top_ten_books= pd.DataFrame(new_book_df.groupby('Book-Title')['Book-Rating'].count()
                         .sort_values(ascending=False).head(10))
print('The top ten books as per ratings : ')
top_ten_books


# In[66]:


# Import Libraries
#Importing modules
import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

# This is to supress the warning messages (if any) generated in our code
import warnings
warnings.filterwarnings('ignore')

import scipy
import math
import sklearn
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import correlation
from sklearn.metrics.pairwise import pairwise_distances
import ipywidgets as widgets
from IPython.display import display, clear_output
from contextlib import contextmanager
import numpy as np
import os, sys
import re
from scipy.sparse import csr_matrix


# In[67]:


books = pd.read_csv('Books.csv', encoding='latin-1')
ratings = pd.read_csv('Ratings.csv', encoding='latin-1')
users= pd.read_csv('Users.csv', encoding='latin-1')


# In[68]:


# User Dataset First Look
users.head()


# In[69]:


# Book Dataset First Look
books.head()


# In[70]:


# Rating Dataset First Look
ratings.head()


# In[71]:


# Dataset Rows & Columns count
print(f'Users Dataset shape: {users.shape}')
print(f'Books Dataset shape: {books.shape}')
print(f'Ratings Dataset shape: {ratings.shape}')


# In[72]:


# User Dataset Info
print(users.info())
print("\n")
print(books.info())
print("\n")
print(ratings.info())


# In[73]:


# Dataset Duplicate Value Count
print(f'Duplicates in Users: {users.duplicated().sum()}')
print(f'Duplicates in Books: {books.duplicated().sum()}')
print(f'Duplicates in Ratings: {ratings.duplicated().sum()}')


# In[74]:


# Missing Values/Null Values Count
print(f'Null values in Users:\n{users.isnull().sum()}\n')
print(f'Null values in Books:\n{books.isnull().sum()}\n')
print(f'Null values in Ratings:\n{ratings.isnull().sum()}\n')


# In[75]:



# Dataset Columns
print(f'Columns in Users: {users.columns}')
print(f'Columns in Books: {books.columns}')
print(f'Columns in Ratings: {ratings.columns}')


# In[76]:


# Dataset Describe
users.describe(include='all')


# In[77]:


# Dataset Describe
books.describe(include='all')


# In[78]:


# Dataset Describe
ratings.describe(include='all')


# In[79]:


# Checking missing value
users.isnull().sum()


# In[80]:


# Unique values in user
print(f'unique value in age: ', sorted(users['Age'].unique()))
print(f'unique value in Location: ', users['Location'].nunique())


# In[81]:


# Take out country from Location
users['Country']=users.Location.str.extract(r'\,+\s?(\w*\s?\w*)\"*$')


# In[82]:


users.head()


# In[83]:


# drop location column
users.drop('Location',axis=1,inplace=True)


# In[84]:


users.head()


# In[85]:


# Checking for Null values in new column Country
users.isnull().sum()


# In[86]:


users['Country']=users['Country'].astype('str')


# In[87]:


# To check Country column
a=list(users.Country.unique())
a=set(a)
a=list(a)
a = [x for x in a if x is not None]
a.sort()
print(len(a))
print(a)


# In[88]:


# Replace mispelled word with correct one.
users['Country'].replace(['','01776','02458','19104','23232','30064','85021','87510','alachua','america','austria','autralia','cananda','geermany','italia','united kindgonm','united sates','united staes','united state','united states','us'],
                           ['other','usa','usa','usa','usa','usa','usa','usa','usa','usa','australia','australia','canada','germany','italy','united kingdom','usa','usa','usa','usa','usa'],inplace=True)


# In[89]:


users['Country'].value_counts()[:10]


# In[90]:


# Top 10 Countries having most users.
sns.countplot(y='Country',data=users,order=pd.value_counts(users['Country']).iloc[:10].index)
plt.title('Count of users Country wise')


# In[91]:


# Plotting histogram for age column
plt.hist(users['Age'],bins=[0,10,20,30,40,50,100])
plt.show()


# In[92]:


# finding outlier in age
sns.boxplot(y='Age', data=users)
plt.title('Find outlier data in Age column')


# In[93]:


# To get distribution plot
sns.distplot(users['Age'])
plt.title('Age Distribution Plot')


# In[94]:


# outlier data into NaN
users.loc[(users.Age > 100) | (users.Age < 5), 'Age'] = np.nan


# In[95]:


users.isnull().sum()


# In[96]:


# Code for filling null with median age on the basis of their Country
users['Age'] = users['Age'].fillna(users.groupby('Country')['Age'].transform('median'))


# In[97]:


users.isnull().sum()


# In[98]:


# Filling remaining Null with Mean of Age
users['Age'].fillna(users['Age'].mean(),inplace=True)


# In[99]:


users.isnull().sum()


# In[100]:


books.head()


# In[101]:


# Top 10 Authors which have written the most books.
sns.countplot(y='Book-Author',data=books,order=pd.value_counts(books['Book-Author']).iloc[:10].index)
plt.title('Top 10 Authors')


# In[102]:


# Top 10 Publisher which have published the most books.
sns.countplot(y='Publisher',data=books,order=pd.value_counts(books['Publisher']).iloc[:10].index)
plt.title('Top 10 Publishers')


# In[103]:


# Converting Year of Publication into string type
books['Year-Of-Publication']=books['Year-Of-Publication'].astype('str')

# for getting unique from Year of publication
a=list(books['Year-Of-Publication'].unique())
a=set(a)
a=list(a)
a = [x for x in a if x is not None]
a.sort()
print(a)


# In[104]:


# investigating the rows having 'DK Publishing Inc' as yearOfPublication
books.loc[books['Year-Of-Publication'] == 'DK Publishing Inc',:]


# In[105]:


#From above, it is seen that bookAuthor is incorrectly loaded with bookTitle, hence making required corrections
#ISBN '0789466953'
books.loc[books.ISBN == '0789466953','Year-Of-Publication'] = 2000
books.loc[books.ISBN == '0789466953','Book-Author'] = "James Buckley"
books.loc[books.ISBN == '0789466953','Publisher'] = "DK Publishing Inc"
books.loc[books.ISBN == '0789466953','Book-Title'] = "DK Readers: Creating the X-Men, How Comic Books Come to Life (Level 4: Proficient Readers)"

#ISBN '078946697X'
books.loc[books.ISBN == '078946697X','Year-Of-Publication'] = 2000
books.loc[books.ISBN == '078946697X','Book-Author'] = "Michael Teitelbaum"
books.loc[books.ISBN == '078946697X','Publisher'] = "DK Publishing Inc"
books.loc[books.ISBN == '078946697X','Book-Title'] = "DK Readers: Creating the X-Men, How It All Began (Level 4: Proficient Readers)"

#rechecking
books.loc[(books.ISBN == '0789466953') | (books.ISBN == '078946697X'),:]
#corrections done


# In[106]:


#investigating the rows having 'Gallimard' as yearOfPublication
books.loc[books['Year-Of-Publication'] == 'Gallimard',:]


# In[107]:


# making corrections
#ISBN '2070426769'
books.loc[books.ISBN == '2070426769','Year-Of-Publication'] = 2003
books.loc[books.ISBN == '2070426769','Book-Author'] = "Jean-Marie Gustave Le ClÃ?Â©zio"
books.loc[books.ISBN == '2070426769','Publisher'] = "Gallimard"
books.loc[books.ISBN == '2070426769','Book-Title'] = "Peuple du ciel, suivi de 'Les Bergers"


# In[108]:


books.loc[books.ISBN == '2070426769',:]


# In[109]:


# making Year again as Integer
books['Year-Of-Publication']=books['Year-Of-Publication'].astype(int)


# In[110]:


print(sorted(books['Year-Of-Publication'].unique()))
#Now it can be seen that yearOfPublication has all values as integers


# In[111]:


# For replacing year as 0 or greater than 2021 to Nan
books.loc[(books['Year-Of-Publication'] > 2021) | (books['Year-Of-Publication'] == 0),'Year-Of-Publication'] = np.NAN

# Replacing NaNs with median value of Year-Of-Publication
books['Year-Of-Publication'].fillna(round(books['Year-Of-Publication'].median()), inplace=True)


# In[112]:


# dropping last three columns containing image URLs which will not be required for analysis
books.drop(['Image-URL-S', 'Image-URL-M', 'Image-URL-L'],axis=1,inplace=True)


# In[113]:


books.isnull().sum()


# In[114]:


# Looking for NaN in Author
books[books['Book-Author'].isnull()]


# In[115]:


# Filling Nan of Book-Author with others
books['Book-Author'].fillna('other',inplace=True)


# In[116]:


# Looking for NaN 'publisher' column
books.loc[books.Publisher.isnull()]


# In[117]:


# Filling Nan of Publisher with others
books['Publisher'].fillna('other',inplace=True)


# In[118]:


books.isnull().sum()


# In[119]:


ratings.head()


# In[120]:


# Checking for any null vlaues
ratings.isnull().sum()


# In[121]:


# Making new dataset which has rating of books that exist in our dataset
new_rating = ratings[ratings["ISBN"].isin(books["ISBN"])]
ratings.shape,new_rating.shape


# In[122]:


# Checking for users in rating with our user dataset
print("Shape of dataset before dropping",new_rating.shape)
new_rating = new_rating[new_rating['User-ID'].isin(users['User-ID'])]
print("shape of dataset after dropping",new_rating.shape)


# In[123]:


# Plotting graph for distribution of ratings
new_rating['Book-Rating'].value_counts(sort=False).plot(kind='bar')
plt.title('Rating Distribution\n')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()


# In[124]:


# Segragating implicit and explict ratings datasets
ratings_explicit = new_rating[new_rating['Book-Rating'] != 0]
ratings_implicit = new_rating[new_rating['Book-Rating'] == 0]


# In[125]:


print('ratings_explicit dataset shape',ratings_explicit.shape)
print('ratings_implicit dataset',ratings_implicit.shape)


# In[126]:


# Countplot for explicit ratings
sns.countplot(data=ratings_explicit , x='Book-Rating', palette='rocket_r')


# In[127]:


# for getting books that are most count of rating
rating_count = pd.DataFrame(ratings_explicit.groupby('ISBN')['Book-Rating'].count())
rating_count.sort_values('Book-Rating', ascending=False).head()


# In[128]:


# For getting details about Books from rating_count
most_rated_books = pd.DataFrame(['0316666343', '0971880107', '0385504209', '0312195516', '0060928336'], index=np.arange(5), columns = ['ISBN'])
most_rated_books_summary = pd.merge(most_rated_books, books, on='ISBN')
most_rated_books_summary


# In[129]:


# Create column Rating average
ratings_explicit['Avg_Rating']=ratings_explicit.groupby('ISBN')['Book-Rating'].transform("mean")
# Create column Rating sum
ratings_explicit['Total_No_Of_Users_Rated']=ratings_explicit.groupby('ISBN')['Book-Rating'].transform('count')


# In[130]:


ratings_explicit.head()


# In[131]:


# Merging all dataset to get a Final Dataset with all infromation
Final_Dataset=users.copy()
Final_Dataset=Final_Dataset.merge(ratings_explicit,on="User-ID")
Final_Dataset=Final_Dataset.merge(books,on="ISBN")


# In[132]:


Final_Dataset.head()


# In[133]:


Final_Dataset.isnull().sum()


# In[134]:


Final_Dataset.shape


# In[135]:


# getting C and m
C= Final_Dataset['Avg_Rating'].mean()
m= Final_Dataset['Total_No_Of_Users_Rated'].quantile(0.90)

# getting a books who had ratings more than 90 percentile
Top_Books = Final_Dataset.loc[Final_Dataset['Total_No_Of_Users_Rated'] >= m]
print(f'C={C} , m={m}')
Top_Books.shape


# In[136]:


# function for weighted average
def weighted_rating(x,C=C,m=m):
  v=x['Total_No_Of_Users_Rated']
  R=x['Avg_Rating']
  return (v/(v+m) * R) + (m/(m+v) * C)


# In[137]:


# Apply function of Top_Books dataset
Top_Books["Score"]=Top_Books.apply(weighted_rating,axis=1)


# In[138]:


# Sorting Dataset on the basis of Score
Top_Books.sort_values("Score",ascending=False,inplace=True)


# In[139]:


# Dropping duplicates from data
Top_Books.drop_duplicates('ISBN',inplace=True)


# In[140]:


# Getting DataFrame of Top 20 Books to recommend every new user.
Top_Books[['Book-Title', 'Total_No_Of_Users_Rated', 'Avg_Rating', 'Score']].reset_index(drop=True).head(20)


# In[141]:


ratings_explicit.head()


# In[142]:


users_interactions_count_df = ratings_explicit.groupby(['ISBN', 'User-ID']).size().groupby('User-ID').size()
print('# of users: %d' % len(users_interactions_count_df))

users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df >= 100].reset_index()[['User-ID']]
print('# of users with at least 100 interactions: %d' % len(users_with_enough_interactions_df))


# In[143]:


print('# of interactions: %d' % len(ratings_explicit))
interactions_from_selected_users_df = ratings_explicit.merge(users_with_enough_interactions_df,
               how = 'right',
               left_on = 'User-ID',
               right_on = 'User-ID')
print('# of interactions from users with at least 100 interactions: %d' % len(interactions_from_selected_users_df))


# In[144]:


interactions_from_selected_users_df.head(10)


# In[145]:


import math


# In[146]:


def smooth_user_preference(x):
    return math.log(1+x, 2)

ratings_full_df = interactions_from_selected_users_df.groupby(['User-ID','ISBN'])['Book-Rating'].sum().apply(smooth_user_preference).reset_index()
print('# of unique user/book ratings: %d' % len(ratings_full_df))
ratings_full_df.head()


# In[147]:


ratings_train_df, ratings_test_df = train_test_split(ratings_full_df,
                                   stratify=ratings_full_df['User-ID'],
                                   test_size=0.20,
                                   random_state=42)

print('# interactions on Train set: %d' % len(ratings_train_df))
print('# interactions on Test set: %d' % len(ratings_test_df))


# In[148]:


#Creating a sparse pivot table with users in rows and items in columns
users_items_pivot_matrix_df = ratings_train_df.pivot(index='User-ID',
                                                          columns='ISBN',
                                                          values='Book-Rating').fillna(0)

users_items_pivot_matrix_df.head()


# In[149]:


users_items_pivot_matrix = users_items_pivot_matrix_df.values
users_items_pivot_matrix[:10]


# In[150]:


users_ids = list(users_items_pivot_matrix_df.index)
users_ids[:10]


# In[151]:


from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds


# In[152]:


# The number of factors to factor the user-item matrix.
NUMBER_OF_FACTORS_MF = 15

#Performs matrix factorization of the original user item matrix
U, sigma, Vt = svds(users_items_pivot_matrix, k = NUMBER_OF_FACTORS_MF)


# In[153]:


users_items_pivot_matrix.shape


# In[154]:


U.shape


# In[155]:


sigma = np.diag(sigma)
sigma.shape


# In[156]:


Vt.shape


# In[157]:


all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)
all_user_predicted_ratings


# In[158]:


all_user_predicted_ratings.shape


# In[159]:


cf_preds_df = pd.DataFrame(all_user_predicted_ratings, columns = users_items_pivot_matrix_df.columns, index=users_ids).transpose()
cf_preds_df.head()


# In[160]:


len(cf_preds_df.columns)


# In[161]:
books.head()


# In[162]:


class CFRecommender:

    MODEL_NAME = 'Collaborative Filtering'

    def __init__(self, cf_predictions_df):
        self.cf_predictions_df = cf_predictions_df

    def get_model_name(self):
        return self.MODEL_NAME

    def recommend_items(self, user_id, items_to_ignore=[], topn=10):
        # Get and sort the user's predictions
        sorted_user_predictions = self.cf_predictions_df[user_id].sort_values(ascending=False).reset_index().rename(columns={user_id: 'recStrength'})

        # Recommend the highest predicted rating content that the user hasn't seen yet.
        recommendations_df = sorted_user_predictions[~sorted_user_predictions['ISBN'].isin(items_to_ignore)].sort_values('recStrength', ascending = False).head(topn)
        recommendations_df=recommendations_df.merge(books,on='ISBN',how='inner')
        recommendations_df=recommendations_df[['ISBN','Book-Title','recStrength']]

        return recommendations_df



cf_recommender_model = CFRecommender(cf_preds_df)


# In[163]:


#Indexing by personId to speed up the searches during evaluation
ratings_full_indexed_df = ratings_full_df.set_index('User-ID')
ratings_train_indexed_df = ratings_train_df.set_index('User-ID')
ratings_test_indexed_df = ratings_test_df.set_index('User-ID')


# In[164]:


def get_items_interacted(UserID, interactions_df):
    interacted_items = interactions_df.loc[UserID]['ISBN']
    return set(interacted_items if type(interacted_items) == pd.Series else [interacted_items])


# In[165]:


# 1
class ModelRecommender:

    # Function for getting the set of items which a user has not interacted with
    def get_not_interacted_items_sample(self, UserID, sample_size, seed=42):
        interacted_items = get_items_interacted(UserID, ratings_full_indexed_df)
        all_items = set(ratings_explicit['ISBN'])
        non_interacted_items = all_items - interacted_items

        random.seed(seed)
        non_interacted_items_sample = random.sample(non_interacted_items, sample_size)
        return set(non_interacted_items_sample)

    # Function to verify whether a particular item_id was present in the set of top N recommended items
    def _verify_hit_top_n(self, item_id, recommended_items, topn):
            try:
                index = next(i for i, c in enumerate(recommended_items) if c == item_id)
            except:
                index = -1
            hit = int(index in range(0, topn))
            return hit, index

    # Function to evaluate the performance of model for each user
    def evaluate_model_for_user(self, model, person_id):

        # Getting the items in test set
        interacted_values_testset = ratings_test_indexed_df.loc[person_id]

        if type(interacted_values_testset['ISBN']) == pd.Series:
            person_interacted_items_testset = set(interacted_values_testset['ISBN'])
        else:
            person_interacted_items_testset = set([int(interacted_values_testset['ISBN'])])

        interacted_items_count_testset = len(person_interacted_items_testset)

        # Getting a ranked recommendation list from the model for a given user
        person_recs_df = model.recommend_items(person_id, items_to_ignore=get_items_interacted(person_id, ratings_train_indexed_df),topn=10000000000)
        print('Recommendation for User-ID = ',person_id)
        print(person_recs_df.head(10))

        # Function to evaluate the performance of model at overall level
    def recommend_book(self, model ,userid):

        person_metrics = self.evaluate_model_for_user(model, userid)
        return

model_recommender = ModelRecommender()


# In[166]:


# 2
print(list(ratings_full_indexed_df.index.values))


# In[170]:


# 3
user=int(input("Enter User ID from above list for book recommendation  "))
model_recommender.recommend_book(cf_recommender_model,user)


# In[205]:


#Top-N accuracy metrics consts
EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS = 100

class ModelEvaluator:

    # Function for getting the set of items which a user has not interacted with
    def get_not_interacted_items_sample(self, UserID, sample_size, seed=42):
        interacted_items = get_items_interacted(UserID, ratings_full_indexed_df)
        all_items = set(ratings_explicit['ISBN'])
        non_interacted_items = all_items - interacted_items

        random.seed(seed)
        non_interacted_items_sample = random.sample(non_interacted_items, sample_size)
        return set(non_interacted_items_sample)

    # Function to verify whether a particular item_id was present in the set of top N recommended items
    def _verify_hit_top_n(self, item_id, recommended_items, topn):
            try:
                index = next(i for i, c in enumerate(recommended_items) if c == item_id)
            except:
                index = -1
            hit = int(index in range(0, topn))
            return hit, index

    # Function to evaluate the performance of model for each user
    def evaluate_model_for_user(self, model, UserID):

        # Getting the items in test set
        interacted_values_testset = ratings_test_indexed_df.loc[UserID]

        if type(interacted_values_testset['ISBN']) == pd.Series:
            person_interacted_items_testset = set(interacted_values_testset['ISBN'])
        else:
            person_interacted_items_testset = set([int(interacted_values_testset['ISBN'])])

        interacted_items_count_testset = len(person_interacted_items_testset)

        # Getting a ranked recommendation list from the model for a given user
        person_recs_df = model.recommend_items(UserID, items_to_ignore=get_items_interacted(UserID, ratings_train_indexed_df),topn=10000000000)

        hits_at_5_count = 0
        hits_at_10_count = 0

        # For each item the user has interacted in test set
        for item_id in person_interacted_items_testset:

            # Getting a random sample of 100 items the user has not interacted with
            non_interacted_items_sample = self.get_not_interacted_items_sample(UserID, sample_size=EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS, seed=item_id)

            # Combining the current interacted item with the 100 random items
            items_to_filter_recs = non_interacted_items_sample.union(set([item_id]))

            # Filtering only recommendations that are either the interacted item or from a random sample of 100 non-interacted items
            valid_recs_df = person_recs_df[person_recs_df['ISBN'].isin(items_to_filter_recs)]
            valid_recs = valid_recs_df['ISBN'].values

            # Verifying if the current interacted item is among the Top-N recommended items
            hit_at_5, index_at_5 = self._verify_hit_top_n(item_id, valid_recs, 5)
            hits_at_5_count += hit_at_5
            hit_at_10, index_at_10 = self._verify_hit_top_n(item_id, valid_recs, 10)
            hits_at_10_count += hit_at_10

        # Recall is the rate of the interacted items that are ranked among the Top-N recommended items
        recall_at_5 = hits_at_5_count / float(interacted_items_count_testset)
        recall_at_10 = hits_at_10_count / float(interacted_items_count_testset)

        person_metrics = {'hits@5_count':hits_at_5_count,
                          'hits@10_count':hits_at_10_count,
                          'interacted_count': interacted_items_count_testset,
                          'recall@5': recall_at_5,
                          'recall@10': recall_at_10}
        return person_metrics


    # Function to evaluate the performance of model at overall level
    def evaluate_model(self, model):

        people_metrics = []

        for idx, person_id in enumerate(list(ratings_test_indexed_df.index.unique().values)):
            person_metrics = self.evaluate_model_for_user(model, person_id)
            person_metrics['User-ID'] = person_id
            people_metrics.append(person_metrics)

        print('%d users processed' % idx)

        detailed_results_df = pd.DataFrame(people_metrics).sort_values('interacted_count', ascending=False)

        global_recall_at_5 = detailed_results_df['hits@5_count'].sum() / float(detailed_results_df['interacted_count'].sum())
        global_recall_at_10 = detailed_results_df['hits@10_count'].sum() / float(detailed_results_df['interacted_count'].sum())

        global_metrics = {'modelName': model.get_model_name(),
                          'recall@5': global_recall_at_5,
                          'recall@10': global_recall_at_10}
        return global_metrics, detailed_results_df

model_evaluator = ModelEvaluator()


# In[206]:


print('Evaluating Collaborative Filtering (SVD Matrix Factorization) model...')
cf_global_metrics, cf_detailed_results_df = model_evaluator.evaluate_model(cf_recommender_model)

print('\nGlobal metrics:\n%s' % cf_global_metrics)
cf_detailed_results_df.head(10)


# In[217]:


# In[23]:


import pickle
import numpy as np


# Defining the Function to Fetch Book Poster URLs
def fetch_poster(suggestion):
    book_name = []
    ids_index = []
    poster_url = []

    for book_id in suggestion:
        book_name.append(filtered_data_pivot_table.index[book_id])

    for name in book_name[0]: 
        ids = np.where(filtered_data['Book-Title'] == name)[0][0]
        ids_index.append(ids)

    for idx in ids_index:
        url = filtered_data.iloc[idx]['Image-URL-M']
        poster_url.append(url)

    return poster_url



# In[24]:


# Defining the Book Recommendation Function
def recommend_book(book_name):
    books_list = []
    book_id = np.where(filtered_data_pivot_table.index == book_name)[0][0]
    distance, suggestion = model.kneighbors(filtered_data_pivot_table.iloc[book_id,:].values.reshape(1,-1), n_neighbors=6 )

    poster_url = fetch_poster(suggestion)
    
    for i in range(len(suggestion)):
            books = filtered_data_pivot_table.index[suggestion[i]]
            for j in books:
                books_list.append(j)
    return books_list , poster_url       



# In[27]:


# In[6]:


# Creating the Streamlit UI
import streamlit as st

book_name=[]
st.header('Book Recommender System')
selected_books = st.selectbox(
    "Type or select a book from the dropdown",
                              book_name)



# In[2]:


#Displaying the Recommendations
if st.button('Show Recommendation'):
   recommended_books,poster_url = recommend_book(selected_books)
   col1, col2, col3, col4, col5 = st.columns(5)
   with col1:
       st.text(recommended_books[1])
       st.image(poster_url[1])
   with col2:
       st.text(recommended_books[2])
       st.image(poster_url[2])
   with col3:
       st.text(recommended_books[3])
       st.image(poster_url[3])
   with col4:
       st.text(recommended_books[4])
       st.image(poster_url[4])
   with col5:
       st.text(recommended_books[5])
       st.image(poster_url[5])


# In[ ]:




