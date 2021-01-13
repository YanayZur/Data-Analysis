#!/usr/bin/env python
# coding: utf-8

# <div style="border:solid Chocolate 2px; padding: 40px">
# 
# <b>Hello, Zuriel! Nice to meet you! 👋</b>
# 
# My name is Arina Uksusova and I am glad to be your reviewer in this project!<br />
# 
# You can find my comments in <font color='green'>green</font>, <font color='gold'>yellow</font> and <font color='red'>red</font> boxes. Examples you can see below:
#     
# <div class="alert alert-block alert-success">
#     <h2>Reviewer's commentary</h2>
#     <br/>
# <b>Well done!👍:</b> In case if task is completely correct and everything is alright!
# </div>
# 
# 
# <div class="alert alert-block alert-warning">
#    <h2>Reviewer's commentary</h2>
#     <br/>
# <b>Remarks📌:</b>  In case when I can give some advice that can help you to improve your work or recommend you some useful links and resources that can help you widen your knowledge and help in future tasks.
# </div>
# 
# <div class="alert alert-block alert-danger">
#     <h2>Reviewer's commentary</h2>
#     <br/>
# <b>Needs fixing!🤔:</b> In case when step requires some extra work and some corrections. Project can't be accepted with comments in the red boxes
# </div>
# 
# Please pay attention to not move or change my comments. It would be good to highlight your answers in some way. 
# 
# Having seen your mistake, for the first time I will only indicate its presence and give you the opportunity to find and correct it yourself. In a real job, your boss will do the same, and I'm trying to prepare you for the job of an analyst. But if you cannot cope with such a task yet, at the next iteration of the review I will give a more accurate hint.
#     
# And one more thing: please, pay attention to both the yellow and red boxes with my comments, and if I directly ask you to fix something, please, try to do it!💪
# 
# #### <font color='Purple'>Student's commentary:</font> for example like this</div>
#     
# OK, now let's go!😏
# </font>

# <div style="border:groove 5px; padding: 40px">
#     <b>Hello Arina! Nice to meet you too! </b>
# You can find my comments in <font color='#33ccff'>Blue</font> <br />
# <div class="alert alert-block alert-info">
# 
# <h2>Student's commentary (me!):</h2>
#     <b>My comment!</b> ✍️:Explanation of a repair that made or If I have something to say to you that need your att attention ☺.</div>
#     
# <div class="alert alert-block alert-info">
# <h2>Student's commentary (me!):</h2>
#     Hope you enjoy!🍹.</div>

# <div class="alert alert-block alert-info">
# <h2>Student's commentary (me!):</h2>
# <h3>Note❕</h3>
# Hello Arina😃,<br>
# After going through all the notes you wrote down for me, I noticed that I had <b>two main problem</b> in the data analysis that affected me throughout the data analysis process, <b>and therefore - also in the conclusions I drew at each and every step from the step 2 - "Data Preparation".</b>
# 
# <b>The main problems are:</b>
#     
# 1. Dealing with missing values (Step 2).
# 2. Determining the period of years appropriate for data analysis (Step 3).
#     
# 
# I would like to inform you that I understood the problems, and corrected accordingly also the steps that <b>were affected</b> by the incorrect analysis I did.
# 
# Hope everything is fine now.🤞
# 
# Thanks,
#     Zuriel👦
# </div>

# # Research on video games sales
# 
# <div style="border:groove 5px; padding: 40px">
# <h4>I work for the online store "Ice", which sells video games all over the world.<br>
# I'll identify patterns that determine whether a game succeeds or not. This will allow me spot potential big winners and plan advertising campaigns for our company.</h4>
# I have  the data from the year that ended and i'm planning a campaign for next year.<br>
# The dataset contains the abbreviation ESRB. The Entertainment Software Rating Board evaluates a game's content and assigns an age rating such as Teen or Mature.

# ### Table of Contents
# 
# * [Step 1. Open the data file and study the general information.](#step1)
# * [step 1. conclusion](#step1.1)<br><br>
#     
# * [Step 2. Prepare the data](#step2.)
# * [step 2. conclusion](#step2.2)<br><br>
#     
# * [Step 3. Analyze the data](#step3.)
#     * [Conclusion on Games release by year:](#step3.a)
#     * [Conclusion varied platforms sales:](#step3.b)
#     * [Conclusion How long does it generally take for new platforms to appear and old ones to fade:](#step3.c)
#     * [Conclusion  Finding the relevant period:](#step3.d)
#     * [Conclusion differences in sales between platforms:](#step3.e)
#     * [Conclusions critics influences:](#step3.f)
#     * [Conclusions games by genre](#step3.g)
#         * [step 3. conclusion](#step3.3)<br><br>
#     
# * [Step 4.Create a user profile for each region](#step4.)
#     * [A)Top 5 profitable platform by regions](#step4.a)
#     * [A)Conclusions platform sales by region](#step4.a.c)
#     * [B)Top 5 profitable genres by regions](#step4.b)
#     * [B)Conclusions platform sales by genre](#step4.b.c)
#     * [C)Check if the ESRB rating affect sales by regions](#step4.c)
#     * [c)Conclusions sales by ESRB rating](#step4.c.c)
#         * [step 4. conclusion](#step4.4)<br><br>
#         
# * [Step 5. Test the hypotheses](#step5.)
#     * [Hypothesis 1](#Hypothesis1.)
#     * [Hypothesis 2](#Hypothesis2.)
#         * [step 5. conclusion](#step5.5)
#         
# * [Step 6. Overall conclusion](#step6.)

# <div class="alert alert-block alert-success">
#     <h2>Reviewer's commentary</h2>
#     <br/>
# <b>Well done!👍:</b> 
#     
# Amazing introduction to the project! It is great that you add the content plan, good job!

# **We will need to install this plugin for visualization later-**

# In[1]:


pip install plotly==4.13.0


# In[2]:


get_ipython().system('pip3 install squarify')


# ## Step 1. Open the data file and study the general information <a class="anchor" id="step1"></a>

# **We have a lot of libraries, and one DataFrame, we will check it:**

# In[3]:


import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as st
import matplotlib.patches as mpatches
import sys
import squarify
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.style as style
from matplotlib.colors import ListedColormap
from matplotlib import cm
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.colors import n_colors
from plotly.offline import init_notebook_mode, iplot
from plotly import tools
import math


df_games = pd.read_csv('/datasets/games.csv')


# **We will create a function that checks whether there are values equal to 0 and if so what is their relative share:**

# In[4]:


def zeros(data):
    for i in data.columns:
        if len(data[data[i]==0]) == 0:
            print(i , len(data[data[i]==0]))
        else:
            print(i,len(data[data[i]==0]),(round(len(data[data[i]==0])/len(data[i]),3)))


# <div class="alert alert-block alert-success">
#     <h2>Reviewer's commentary</h2>
#     <br/>
# <b>Well done!👍:</b> 
#     
# Interesting function! For checking the data for zero values you can also use isin() method and any()

# <div class="alert alert-block alert-info">
# <h2>Student's commentary (me!):</h2>
#     Thank you for the tip ✔️</div>

# ### We will do a series of tests for the data:

# In[5]:


df_games.info()


# There seem to be a lot of null values and columns not of the desired type, we will fix this soon.

# In[6]:


df_games.head(5)


# In[7]:


df_games.tail(5)


# In[8]:


df_games.sample(5)


# **We will rename the columns to lowercase:**

# In[9]:


df_games.columns = df_games.columns.str.lower()


# In[10]:


df_games.describe()


# In[11]:


df_games.describe(include = 'object')


# We see here the score of the type 'tbd', which means "to be determined", that is, it can be treated as a missing value , null.

# **Zero's check:**

# In[12]:


zeros(df_games)


# There are 0 values only in **sales-related columns**.<br>
# I assume that these values are **correct** and do not need to be corrected, or get rid of the rows.<br>
# 
# **We will test the 3 sigmas to be sure:**

# I have created a function that will do this for me:

# In[13]:


def three_sigmas (df,column):
    x1 = df[column].mean()-3*np.std(df[column])
    x2 = df[column].mean()+3*np.std(df[column]) 
    filters = df[(df[column]<x2) & (df[column]>x1) ]
    return round(len(filters)/len(df),5)


# In[14]:


print("3 sigmas test of na_sales result = ",three_sigmas(df_games,'na_sales'))
print("3 sigmas test of eu_sales result = ",three_sigmas(df_games,'eu_sales'))
print("3 sigmas test of jp_sales result = ",three_sigmas(df_games,'jp_sales'))


# #### Conclusion Examining the three sigmas:
# 
# The whole data seems to be in the range of **+- 3 sigmas**.
# 
# It can be concluded that our suspicious values (the 0 values) are **true values** and not outliers that need to be delited. We will consider them as a 0's sells and will not remove them..

# **Null's check:**

# <div class="alert alert-block alert-success">
#     <h2>Reviewer's commentary</h2>
#     <br/>
# <b>Well done!👍:</b> 
#     
# Yes, you are absolutely right! It is interesting that you use 3 sigmas rule at this step of the analysis! Great knowledge of statistics!

# In[15]:


print(df_games.isnull().sum())
print(df_games.isnull().mean())


# About **40-50 percents!** of the 'Critic_Score','User_Score' columns has a null's values. 
# 
# **My hypothesis and way of treatment:**<br>
# **- Critic_Score and User_Score columns:**
# Since we do not have access to the original data, and we do not have the ability to ask the person who collected the information to correct it for us- <br>We will fill in the Null values of those columns with 0 and pay attention to these values in any future analysis.<br> 
# 
# **- Rating column**:Since we do not have access to the original data, and we do not have the ability to ask the person who collected the information to correct it for us and of curse there may not have been a rating for these games, we will fill the Null values With "Unknown". 
# 
# **- year_of_release column**
# As for this column we see that the relative share of the missing values is really small (1.6%) and therefore we will fill them using a method "transform".
# 
# 

# **Let's take a look at the 'platform' column:**

# In[16]:


df_games['platform'].value_counts().head(10)


# ### Conclusions Step 1. <a class="anchor" id="step1.1"></a>
# The tests we did-
# 1. Examination of the data.
# 2. Check NULL values.
# 3. Check suspects 0's (3 sigma test).
# 4. Check null's values.
# 5. Check statistics.
# 
# **According those tests:**<br>
# - It can be concluded that the 0's we found in the sales columns are correct and show zero sales.<br>
# - There are a lot of missing values in the review's columns, we will have to deal with them in the next step.
# - There are a few missing values in the 'year of release' column, we will fill it in the next step.
# **Now we go to the next step, and we will prepare the data for analysis:**

# <div class="alert alert-block alert-success">
#     <h2>Reviewer's commentary</h2>
#     <br/>
# <b>Well done!👍:</b> 
#     
# Ok, you have successfully finished the first step of the analysis! Let's see what you got at next steps!👀

# ## Step 2. Prepare the data <a class="anchor" id="step2."></a>

# We will start by examining the 'name' column and see what can be done with the 2 null's values:

# **'name' column Null's check:**

# In[17]:


df_games[df_games['name'].isnull()]


# The quantity is very small, and the information is missing in a lot of columns.<br>
# **We will get rid of these rows:**

# In[18]:


df_games = df_games[df_games['name'].notna()]


# <div class="alert alert-block alert-success">
#     <h2>Reviewer's commentary</h2>
#     <br/>
# <b>Well done!👍:</b> 
#     
# Yes, you can delete these data, this action will not greatly affect the data!

# **We will continue with the fill of the missing values in the column 'year_of_release' using the transform method:**
# 

# In[19]:


df_games['year_of_release'] = (df_games.groupby('platform')['year_of_release']
                               .transform(lambda grp: grp.fillna(grp.median())))


# **As part of the preparation, we will convert the column 'year_of_release' type to 'int':**

# In[20]:


df_games['year_of_release'] = df_games['year_of_release'].astype('int')


# ### Now, as I explained in the previous step, we will approach the next three columns:  'critic_score' ,'user_score', 'rating'  and deal with their missing values:

# **We will now fill in the missing values with 0 in the 'critic_score' column:**
# We will check the distribution before and after the addition:

# In[21]:


df_games.critic_score.hist(bins = 20)
plt.title('Critic score before', size = 20 )
plt.xlabel('Critic mark',size =15)
plt.ylabel('Frequency',size = 15);
print(df_games.critic_score.describe())


# We see that the scatter is normal, and I assume that after we fill in the missing values with 0, we will get a big "pick" at the value 0.

# In[22]:


df_games['critic_score'] = df_games['critic_score'].fillna(0)


# In[23]:


df_games.critic_score.hist(bins = 50,color = 'y')
plt.title('Critic score after', size = 20 )
plt.xlabel('Critic mark',size =15)
plt.ylabel('Frequency',size = 15);
print(df_games.critic_score.describe())


# As we assumed earlier, we got a normal scatter except around the value -0.

# **We will now handle with the column 'user_score'.<br>**
# The treatment will be a bit complicated since the column has values of type 'object', and type 'float'.
# The value - 'tbd' says that a rating has not yet been set, so I consider it as the rest of the null values, and then we will fill all the values with 0 as in the column 'critic_score'-
# We will check the distribution before and after the addition:

# In[24]:


df_games['user_score'] =df_games['user_score'].replace('tbd',np.nan)
df_games.user_score = df_games.user_score.astype('float')

df_games.user_score.hist()
plt.title('User score before', size = 20 )
plt.xlabel('User mark',size =15)
plt.ylabel('Frequency',size = 15);
print(df_games.user_score.describe())


# We see that the scatter is normal, and I assume that after we fill in the missing values with 0, we will get a big "pick" at the value 0.

# In[25]:


df_games['user_score'] = df_games['user_score'].fillna(0)


# In[26]:


df_games.user_score.hist(color = 'y')
plt.title('User score after', size = 20 )
plt.xlabel('User mark',size =15)
plt.ylabel('Frequency',size = 15);
print(df_games.user_score.describe())


# As we assumed earlier, we got a normal scatter except around the value -0.

# **We will now fill in the missing values with "Unknown" in the 'rating' column:**<br>
# We will fill in the values with "Unknown" because we will later examine the effect of the rating on the sales of the games, and also and perhaps more - the effect of the non-rating on sales.

# In[27]:


df_games['rating'] = df_games['rating'].fillna("Unknown")


# <div class="alert alert-block alert-warning">
#    <h2>Reviewer's commentary</h2>
#     <br/>
# <b>Remarks📌:</b>  
# 
# And yet the data before and after filling in the missing vales differs, yes, this difference does not seem critical at first glance, but it is there and it is important to understand this. As for histograms, this is a rather imprecise way to estimate the distribution, and in this case, the graphs are still different, although perhaps not significantly.
# 

# <div class="alert alert-block alert-warning">
#    <h2>Reviewer's commentary</h2>
#     <br/>
# <b>Remarks📌:</b>  
# 
# As for filling the missing values in columns user_score and critic_score and year of release: it is not very good and correct strategy to replace NaN values with the mean (or median) in case of critic_score and user_score column. Because the mean or median can provide us with the bias estimate at the final stages of our analysis. Why is it so? 
# 
# 
# Because filling the NaN value of critic_score (user_score) for games, that have not the criritc score (user_score), with the mean (or median) that was calculated based on the critic_score (user_score) values for the other games we create the artificial critic score (user_score) for games that initially didn't have this score yet. Doing so we distort the original data. And you can easily check out that in this dataset we see a simultaneous lack of data in the critics score, user score and ratings columns.
# 
#     
# This means that this data was simply not added to the dataset. Therefore, it is better not to fill in the gaps in the user score and critic score indicators with artificial statistics, but leave it as it is. So, please, improve this step of analysis.
# 
# In fact, we do not know why the data in this column is missing (we can only hypothesize), and we do not have access to a source that would allow us to replenish the data, restore it, so it is better leave missing values in the data. Missing values are also some kind of signal worth paying attention to and analyzing, it is valuable in its own way! The analyst's goal when working with missing values is not about filling the 100% of missing values! When filling missing values, it is important to proceed from the specifics of the data, take into account the risks of data distortion and the introduction of artificial trends and characteristics.
#     
# As you have identified the amount of missing values in columns user_score and critic_score is about 40-50%, it is too big! We cal fill the missings only when their amount is not bigger that 10%
# 
# Even if the statistical check showed that everything is more or less normal, when filling inmissing values, one must focus on adequacy and common sense, as well as the risks of distortion of the initial data, this distortion cannot always be detected on the distribution graph or by comparing the median and standard deviation, this is the distortion will be in terms of the quality of the data and semantic content, as well as the conclusions that we will eventually receive.
# 
# One possible strategy is to fill the NaN values with 0 (critic_score and user_score) and 'unknown' in case of ratings. Because the other option - is to delete such values, but I think you will agree with me that it could lead to a tremendous loss of data, at the same time, filling with zero values allows to preprocess and analyze data using these columns.

# <div class="alert alert-block alert-info">
# <h2>Student's commentary (me!):</h2>
# Thanks so much for the extension and explanation!<br>
# I was really wondering to myself while preparing the project what is the correct way to analyze such corrupt information that is full of null values.<br>
# I understood your explanation and as you can see I did implement it.<br>
# I followed the method you suggested and I will emphasize the values that I filled , if required later.
#    </div>

# <div class="alert alert-block alert-success">
#     <h2>Reviewer's commentary:second iteration of the review</h2>
#     <br/>
# <b>Well done!👍:</b> 
#     
# Zuriel, yes, I see your corrections here, good job! Thank you for responding to my comments, getting feedback is very important for a reviewer!

# ### We will add a column that calculates the global amount of sales in all regions: 'total_sales'

# In[28]:


df_games['total_sales'] = df_games[['na_sales','eu_sales','jp_sales','other_sales']].sum(axis = 1)
df_games.sample(3)


# In[29]:


plt.figure(figsize=(15,3))
ax = sns.distplot(df_games.total_sales)
plt.xlim(0, 4)
plt.title('Total sales in m$', size = 20 )
plt.xlabel('Sales in m$',size =15)
plt.ylabel('Frequency',size = 15)
print(df_games.total_sales.describe())
print("3 sigmas test of total_sales result   = ",three_sigmas(df_games,'total_sales'))


# <a class="anchor" id="step2.2"></a> 
# ### Conclusions Step 2.
# Based on what we did we can say this:
# 
# 1. Since it is not possible to know or restore the missing values in the columns 'critic_score' and 'user_score' we filled them in with 0's and we willpay attention to them later if required.
# 2. We have added a column that calculates sales in all regions, which will allow us to later calculate profitability based on various parameters.
# 3. There is an interesting statistic, seen in the column 'total_sales'  between 0 and 1.6 million, the distribution is similar to a uniform distribution.
# 
# **Now on the basis of the prepared table it is possible to further examine the behavior of customers.**

# ### Step 2 Readiness Checklist
# 
# - [X] Replace the column names (make them lowercase).
# - [X] Convert the data to the required types.
# - [X] Describe the columns where the data types have been changed and why.
# - [X] If necessary, decide how to deal with missing values:Explain why you filled in the missing values as you did or why you decided to leave them blank.
# - [X] Why do you think the values are missing? Give possible reasons.
# - [X] Pay attention to the abbreviation TBD (to be determined). Specify how you intend to handle such cases.
# - [X] Calculate the total sales (the sum of sales in all regions) for each game and put these values in a separate column.

# <a class="anchor" id="step3."></a>
# ## Step 3. Analyze the data

# **Now that the data is ready, we will dive a little deeper to get a picture of the trends so that we can in the end decide on which platform to invest in next year.**

# **Let's take a look at how many games were released in different years:**

# In[30]:


df = df_games[['year_of_release','name']].groupby('year_of_release').count().sort_values(by = 'year_of_release').reset_index()
df = df[df['year_of_release'] != 0]


# ## (I could not decide which way to view the data, so I left both ways ☺)

# In[31]:


fig,ax = plt.subplots(figsize=(17,10))
ax.vlines(x=df.year_of_release, ymin=0, ymax=df.name, color='orange', alpha = 0.7, linewidth=2)
ax.scatter(x=df.year_of_release, y=df.name, s=75, color='black', alpha =0.7)
ax.set_title('Games released by year', fontdict ={'size':17})
ax.set_ylabel('Number of games',fontdict ={'size':15})
ax.set_xlabel('Year of release',fontdict ={'size':15})
ax.set_xticks(df.year_of_release)
ax.set_xticklabels(df.year_of_release, rotation = 45 , fontdict ={'horizontalalignment':'right','size':12});

for row in df.itertuples():
    ax.text(row.year_of_release,row.name+30, s=round(row.name,2),ha='center')


# <div class="alert alert-block alert-success">
#     <h2>Reviewer's commentary</h2>
#     <br/>
# <b>Well done!👍:</b> 
#     
# Ok, very interesting and informative graph!

# In[32]:


df = df_games[['year_of_release','name']].groupby('year_of_release').count().sort_values(by = 'year_of_release').reset_index()
df = df[df['year_of_release'] != 0]

plt.figure(figsize=(12,8))
ax = sns.barplot(y=df.year_of_release, x=df.name, orient='h',edgecolor='black')
ax.set_xlabel(xlabel='Number of games', fontsize=16)
ax.set_ylabel(ylabel='Year of release', fontsize=16)
ax.set_title(label='Games released by year', fontsize=20)
plt.xticks(np.arange(0, 1600,100))
for row in df.itertuples():
    ax.text(row.name+30,row.year_of_release-1979.75, s=round(row.name,2),ha='center', color='maroon')
plt.show();


# <div class="alert alert-block alert-success">
#     <h2>Reviewer's commentary</h2>
#     <br/>
# <b>Well done!👍:</b> 
#     
# Also good one and what is important - well formatted, good job!

# <a class="anchor" id="step3.a"></a>
# ### Conclusion on Games release by year:
# - **It can be seen that there has been a consistent but slow increase since 1990.**<br>
# - **It can also be seen that the peak years were in 2007- 2008 and after that in 2011 a big fall.**<br>
# - **In the years 2000-2001 there seems to be a significant jump of game release, and reaching new heights. Later in the analysis, we would like to check the sales information from these years, since it seems that there is more activity in them.**

# ### Now we will take a look and analise how sales varied from platform to platform,after that and base on that analysis, we'll choose the platforms with the greatest total sales and build a distribution of greatest platform sales for each year:

# **In order to check which platform is with the greatest sales we will use the "Z score" analisies:<br>**
# Z score show us how far each value in the total sale column is from the mean in terms of std(more then 3 it mean that its outlier),<br> far and positive indicate best seller platform!

# In[33]:


df_sales = df_games[['platform','total_sales']].groupby('platform').sum().sort_values(by = 'total_sales').reset_index()


# Convert to std terms:

# In[34]:


df_sales['sales_z']=(df_sales['total_sales']-df_sales['total_sales'].mean())/df_sales['total_sales'].std()


# Set indicators:

# In[35]:


df_sales['color'] = ['red' if x<0 else 'green' for x in df_sales['sales_z']]


# ### Show the varied platforms sales: 

# In[36]:


plt.figure(figsize=(14,10))
plt.hlines(y=df_sales.platform , xmin=0 ,xmax=df_sales.sales_z, color=df_sales.color , alpha = 0.5 , linewidth=10);


# **It can be clearly seen that the PS2 platform is the platform with the highest Z score.**<br>
# There are almost 3 positive deviations from the average.

# **Following the analysis, we would also like to see numerical values, we will represent this by a treemap:**

# In[37]:


df_tree = df_sales
sizes   = df_tree.total_sales.values.tolist()
labels  = df_tree.apply(lambda x: str(x[0])+"\n"+"$"+str(round(x[1])), axis = 1)
plt.figure(figsize=(15,9))
squarify.plot(sizes=sizes, label = labels, alpha = 0.5,edgecolor = 'white' ,linewidth = 4 );


# <a class="anchor" id="step3.b"></a>
# ### Conclusion varied platforms sales: 
# 
# - **In the analysis performed, it is clear that the platform with the biggest sales over the years is "PS2".**<br>
# - **Over the years, PS2 sales have totaled over 1 billion dollars** <br>
# 
# We will examine the sales distribution of "PS2" by year:

# In[38]:


df_ps2 = df_games[df_games['platform'] == 'PS2']
df_ps2 = df_ps2[['year_of_release','total_sales']].groupby('year_of_release').sum().reset_index()


# In[39]:


fig = go.Figure(go.Bar(x=df_ps2['year_of_release'],y=df_ps2['total_sales'],
                       marker={'color': df_ps2['total_sales'],'colorscale': 'peach'}))
fig.update_layout(
                title_text='Revenue of PS2 by Years',
                xaxis_title="Year",
                yaxis_title="$ Millions",
                xaxis = dict(
        tickmode = 'linear'))
fig.show()


# Contrary to the trend of all the data, in the years 2007-2008 there was actually a **decrease**.<br>
# The peack is between the years 2002-2004 with over $ 200 million profit per year.

# <div class="alert alert-block alert-info">
# <h2>Student's commentary (me!):</h2>
#     <b>My comment!</b> <br>
# I have corrected the erroneous logic regarding the selection of the relevant period for the analysis.<br>
# The result is here👇🏻👇🏻👇🏻👇🏻👇🏻</div>

# <div style="border:groove 5px; padding: 40px">
# <h2>Finding the relevant period for analyzing by examining the growth and shrink of the platforms sales over the years:</h2>
#     
# <h3>To find the relevant period for analyzing the data, we will work in three steps:</h3>
# 
# **Step 1: Years in the market:** We will prepare a "Dynamic table" showing the profit and loss of each platform compared to the previous year. With the help of the "Dynamic table", We will examine how many years on average it takes for the platform to be in the market by measuring the years it took for each platform to disappear from the market.<br>
# 
# **Step 2 Filtering the data:** Once we have identified the average number of years the platform has been on the market, we will want to filter the data so that it shows us from the **current year, minus the number of years we found in Step 1**.
# 
# **Step 3 Check the relevance:** After the filtering in step 2,  We will then transfer the data from a filtered "Dynamic table" by the years-  to a hitmap, and with the help of the hitmap we created, we will determine the duration of **relevance** of the platform in the market and **subtract that time from the current year and this will be the time period we continue to analyze the data.**

# <div class="alert alert-block alert-success">
#     <h2>Reviewer's commentary:second iteration of the review</h2>
#     <br/>
# <b>Well done!👍:</b> 
#     
# Zuriel, ok, it is great that you provided so comprehensove explanation of your strategy! Let's see, what results you got!👀🤓

# ### Step 1:
# We will prepare a "Dynamic table" showing the profit and loss of each platform compared to the previous year.

# In[40]:


df_change   = pd.pivot_table(df_games, index = 'year_of_release', 
                            columns = 'platform', values = 'total_sales', 
                            aggfunc = 'sum', fill_value = 0)

df_dynamics = df_change - df_change.shift(+1)
df_dynamics


# **With the help of the "Dynamic table", We will examine how many years on average it takes for the platform to be in the market by measuring the years it took for each platform to disappear from the market:**
# 

# In[41]:


fade_years = df_dynamics.T.astype(bool).sum(axis=1).sort_values(ascending = False)
fade_years


# In[42]:


print(f'The average time it takes to fade from the moment the platform appears is: {round(fade_years.mean(),2)} years.')


# ### We will represent this in the following graph:

# In[43]:


fig = plt.figure(figsize = (10,6))
ax = fig.add_axes([0,0,1,1])

platform = fade_years.index
year = fade_years.values

plt.yticks(np.arange(0, year.max()+1))
sns.barplot(x=platform, y=year, palette="RdBu")
plt.xticks(rotation= 30)
plt.xlabel("Platform", fontsize=16, color="blue")
plt.ylabel("Years in the market", fontsize=16, color="blue")
plt.axhline(fade_years.mean())
plt.title("Number of years in the market by Platform", fontsize=18, color="red")
plt.show()


# <a class="anchor" id="step3.c"></a>
# 
# ### Conclusion Step 1. 
# (How long does it generally take for new platforms to appear and old ones to fade)
# 
# - **The average "life time" (in years) of the platform from the moment it appears on the market until the moment it disappear is almost 10 years.
# - **The most stable platform is not surprisingly the PC, we can say that this platform is immortal!**

# ### Step 2:
# We identified that the average number of years the platform has been on the market is 10 years.<br>
# Now we will filter the data so that it shows us from the **current year, minus 10 year (2016 - 10 = 2006):**

# In[44]:


df_filtered_dynamics = df_dynamics.query('year_of_release >2006')
df_filtered_dynamics.loc['total',:]= df_filtered_dynamics.sum(axis=0)
df_filtered_dynamics = df_filtered_dynamics.T.query('total!=0')
df_filtered_dynamics.drop("total",axis = 1,inplace=True)


# ### Step 3 Check the relevance:
# After the filtering in step 2,  We now transfer the data to a hitmap, and after -  We will determine the duration of **relevance** of the platform in the market and **subtract that time from the current year and this will be the time period we continue to analyze the data.**

# Building a hitmap : 

# In[45]:


plt.figure(figsize = (13,9))
sns.heatmap(df_filtered_dynamics, cmap = 'RdBu_r',annot=True, fmt=".2f",annot_kws={'size':8})
ax.set_ylim((0,15))
plt.title("Annual profit or loss compared to the previous year by Platform\n", fontsize = 17,color = 'r', fontstyle='italic')
plt.xlabel('Years 2007 - 2016',fontsize=16, color="blue")
plt.ylabel('Platform',fontsize=16, color="blue")
plt.show();


# According to the hitmap it is clear that the number of years relevant to each platform is 4 years + - (it is easy to identify this based on the darker red colors).<br>
# **Hence-**
# If we want to determine which game or platform is more "powerful" for our campaign, we would like to filter the data **back 4 years in order to identify current trends!**

# **Filtering the data:**

# In[46]:


df_games_filtered = df_games.query('year_of_release >= 2012')
df_games_filtered.sort_values('year_of_release').head(1)


# In[47]:


df_games_filtered_1 = df_games_filtered.groupby('platform')['total_sales'].sum().reset_index().sort_values('total_sales',ascending = False)

fig = go.Figure(go.Bar(x=df_games_filtered_1['platform'],y=df_games_filtered_1['total_sales'],
                       marker={'color': df_games_filtered_1['total_sales'],'colorscale': 'magma'}))
fig.update_layout(
                title_text='Revenue of Platform by Years (2012 - 2016)',
                xaxis_title="Year",
                yaxis_title="$ Millions",
                xaxis = dict(
        tickmode = 'linear'))
fig.show()


# <div class="alert alert-block alert-warning">
#    <h2>Reviewer's commentary</h2>
#     <br/>
# <b>Remarks📌:</b>
# 
# It is the right strategy that you tried to determine the current period, however, you did not do it quite accurately. The lifespan of the game is about 10 years, but lifespan does not equal "relevance". The relevance of the game falls according to the charts on average for 3-4 years (since games quickly become obsolete, although they can continue to be sold in stores) Accordingly, if we choose a longer period, for example, as in your case, we risk capturing a lagging trend, i.e. we include in the data irrelevant pre-period platforms that should not remain in the data for the predicted period.
# 
# <br/>
# For example, the case of PS2. We see that the leader in sales in the entire history of the PS2. But sales by 2011 are already equal to zero. Or X360: it was at its peak in 2010, but by 2016 it had dropped to almost 0. You can see this if you build the histogram of total_sales for each platform. That is why the choice of 1998 year as a start year can catch the outdated tendency and bias the results of your analysis at the next steps. 
#  
# In this project we want to PREDICT the platfoms sales and popularity, so, when we talk about prediction it is very important to use the most recent data.
# 
# <br/>
# It is worth adjusting the time period here to obtain more accurate results. So , please, try to correct the actual period! <br><br>   
# I would like to recommend you to choose 2012 or 2013 year as a lower border of the current period
#     
# The year that can be considered the boundary of the current period can be determined by building histograms for each platform. The histograms will show that games lose relevance within 3-4, maximum 5 years, at the same time the lifespan of the platform is on average about 10 years. The final year of the dataset is 2016, so we subtract 3-4 years from 2016 and get a year, which can be considered the boundary of the current period.
# 
# **The example of platforms that remain in dataframe after selecting the actual years (here from 2012)🙌**    
# ![image.png](attachment:image.png)

# <a class="anchor" id="step3.d"></a>
# ### Conclusion  Finding the relevant period:
# 
# After the three steps it can be concluded that:
# 
# 1. The average period that a platform survives in the market is 10 years.
# 2. During the time the platform is in existence, the time it is relevant and profitable is a period of 4 years.
# 
# Based on these conclusions, we filtered our data for the years 2012 to 2016 in order to identify the trends relevant to our research.
# #### The potential platforms I would mark for the future:
# 1. PC - Because of its stability over the years.
# 2. PS4 -Leads in sales in the relevant period.
# 3. XOne - The leading competitor in sales in the relevant period.

# <div class="alert alert-block alert-success">
#     <h2>Reviewer's commentary</h2>
#     <br/>
# <b>Well done!👍:</b> 
#     
# Ok, you have identified the profitable platforms for future correctly!

# ### In order to see if there is significant differences in sales between platforms, We'll build a box plot for the global sales of all games, broken down by platform:

# In[48]:


platform_sales = df_games_filtered.groupby(['platform','year_of_release'])['total_sales'].sum().reset_index()
ordered = platform_sales.groupby(['platform'])['total_sales'].sum().sort_values().reset_index()['platform']


# In[49]:


plt.figure(figsize=(13,10))
ax =sns.boxplot(x='platform', y='total_sales',data= platform_sales,palette="RdBu", order =ordered )
ax.set_xticklabels(labels= ordered , fontsize=12, rotation=50)
ax.set_ylabel(ylabel='Revenue in $ Millions', fontsize=16)
ax.set_xlabel(xlabel='Platforms', fontsize=16)
ax.set_title(label='Distribution of Revenue Per Platform in $ Millions', fontsize=20)
plt.show();


# <a class="anchor" id="step3.e"></a>
# ### Conclusion differences in sales between platforms :
# 
# - **The PS4 leads in all parameters over all competitors in the relevant period (median and top sales).**
# - **The top four platforms (after PS4) have almost the same median.**
# 
# **To summarize the analysis - there is a significant difference between the averages of the various platforms, and the sales data. Very significant changes.**

# ## Now it's interesting to know ,if the critics have any influences?
# **We will filter the data according to a very leading platform -the PS4 and measure the effects of the critics on the sales and maybe themselfs?.**

# We will see the influences of the critics on sales, and the professional critics on that of the users and show that with the help of scatter plots:

# <div class="alert alert-block alert-info">
# <h2>Student's commentary (me!):</h2>
# ⚠️For the sake of accuracy the correlation test, only the values with the real reviews were taken (not the 0 values we added earlier)
#     </div>

# <div class="alert alert-block alert-success">
#     <h2>Reviewer's commentary:second iteration of the review</h2>
#     <br/>
# <b>Well done!👍:</b> 
#     
# Yes, it is absolutely right! Good job!

# In[50]:


df_PS4 = df_games_filtered[(df_games_filtered['platform'] == 'PS4')&(df_games_filtered['critic_score']> 0)]
df_PS4 = df_PS4.loc[:,["critic_score","user_score","total_sales" ]]

df_PS4["index"] = np.arange(1,len(df_PS4)+1)
# scatter matrix
fig = ff.create_scatterplotmatrix(df_PS4, diag='box', index='index',colormap='YlOrRd',
                                  colormap_type='seq',title = 'Correlation test between critics and "PS4" total sales',
                                  height=700, width=800)
iplot(fig)


# <div class="alert alert-block alert-warning">
#    <h2>Reviewer's commentary</h2>
#     <br/>
# <b>Remarks📌:</b> 
# 
# Ok, why are you choose X360 here? It is better to choose the most popular platform that you can identify only after the choosing the right actual period. When you correct your decision in the previous step, you will see that in fact, the most popular platform is not the X360 (a tiny hint: pay attention to PS4 after choosing the correct actual period)🙄

# <div class="alert alert-block alert-info">
# <h2>Student's commentary (me!):</h2>
# A more popular platform was chosen, of course further to address the main issue regarding the duration of the period.
# Thanks for the comment! ✔️</div>

# <div class="alert alert-block alert-success">
#     <h2>Reviewer's commentary:second iteration of the review</h2>
#     <br/>
# <b>Well done!👍:</b> 
#     
# Now everything is alright!

#   <a class="anchor" id="step3.f"></a>
# 
# ## Conclusions critics influences
# - **It can be seen that there is a (but relatively small) influences of the professional critics on the sales, the higher the scores the higher the sales.**
# 
# We will check this in a measurable way with the help of the correlation table:

# In[51]:


df_PS4.corr()['total_sales']


# As we observed, there is a weak positive relationship between the professional critics and sales, although it is not really strong (close to 0.40).

# ## We will examine the correlation between visitors and sales on the rest of the platforms, are the correlations similar for everyone?

# In[52]:


#ordered = platform_sales.groupby(['platform'])['total_sales'].sum().sort_values().reset_index()['platform']
df_scatt = df_games_filtered[df_games_filtered['critic_score']> 0]
df_scatt = df_scatt[['platform', 'critic_score','user_score', 'total_sales']]


# In[53]:


def corr(platform):
    
    df = df_scatt[df_scatt['platform']== platform]
    f,ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(df.corr() ,annot=True, linewidths=.9, linecolor='white', fmt= '.2f',ax=ax,vmin=-1,vmax=1)
    plt.show();


# In[54]:


lst = ordered.tail(9).tolist()

for i in lst:
    print(f'The correlation between {i} total sales and critics reviews :\n')
    print(corr(i))


# **Thanks to the hitmap at a glance we were able to notice that most of the platforms have the same correlation between sales and reviews. It can be seen that the strongest connection between the professional critics and the sales is in the XOne and PS4 platform.**

# <div class="alert alert-block alert-success">
#     <h2>Reviewer's commentary</h2>
#     <br/>
# <b>Well done!👍:</b> 
#     
# Ok, you have interpreted the heatmaps correctly and you visualization looks really good!

# <div class="alert alert-block alert-warning">
#    <h2>Reviewer's commentary</h2>
#     <br/>
# <b>Remarks📌:</b> 
# 
# Zuriel, it is not a very correct strategy! You need to slice the data when you try to determine the actual period and then use only this slice not without referring to the original data! So, please, try to correct this step!

# <div class="alert alert-block alert-info">
# <h2>Student's commentary (me!):</h2>
# Thanks for the comment!<br>
# I did fix that too and now we'll continue the analysis based on the filtered information -
#     </div>

# <div class="alert alert-block alert-success">
#     <h2>Reviewer's commentary:second iteration of the review</h2>
#     <br/>
# <b>Well done!👍:</b> 
#     
# Ok, thank you for this correction!

# ## Determine which genre is the most profitable:

# 1. First we will look at the  total sales by genre.
# 2. Then we will look at the average sales value per genre.

# In[55]:


# Total Sales by Genre 
genre = df_games_filtered.loc[:,['genre','total_sales']]
genre['total_sales'] = genre.groupby('genre')['total_sales'].transform('sum')
genre['mean'] = genre['total_sales']/genre['total_sales'].sum()
genre = genre.drop_duplicates()
fig = px.pie(genre, names='genre', values='total_sales', template='seaborn')
fig.update_traces(  hoverinfo ="label",
                    rotation=45,
                    pull=[0.03,0,0.05,0.03,0.04,0.08,0,0.07,0.2,0,0.03,0,0.2],
                    textinfo="label+percent",
                    textfont_size=10
                  )
fig.update_layout(title="The Sales of Games Released by Genre (In M$)",title_x=0)
fig.show()


# Video Game avarage sells by Genre
genre_with_game = df_games_filtered.pivot_table(index = 'genre', values = ['total_sales'], aggfunc =['count','mean'])
genre_with_game = genre_with_game.reset_index()
genre_with_game.columns = ['genre','count', 'mean']

fig = go.Figure([go.Pie(labels=genre_with_game['genre'], 
                        values= round(genre_with_game['mean'],2)*1000000,
                        hole=0.3)])  

fig.update_traces(
                   hoverinfo='label+percent+value', 
                   textinfo='value+label', 
                   textfont_size=10,
                   pull=[0.0,0,0.0,0.0,0.0,0.0,0,0.0,0.2,0,0.03,0,0.2]
                    )
fig.update_layout(title="The Aaverage Sales per Genre (In M$)",title_x=0)
fig.show()        


# <div class="alert alert-block alert-danger">
#     <h2>Reviewer's commentary</h2>
#     <br/>
# <b>Needs fixing!🤔:</b> 
# 
# At this step your visualization provides an error. Please, try to understand what is going wrong and fix it!👀
# 
# ![image.png](attachment:image.png)

# <div class="alert alert-block alert-info">
# <h2>Student's commentary (me!):</h2>
#     Oh 😞... really a pity  he didn't show up at first! Had to install the latest version of matplotlib what I just put up :) enjoy!  ☺.</div>

# <div class="alert alert-block alert-warning">
#    <h2>Reviewer's commentary</h2>
#     <br/>
# <b>Remarks📌:</b> 
# 
# Zuriel, your results at this step is not very correct. Misc cannot be at the third place, please, correct the actual period and use only corrected period here to get the more correct results: the second position should be occupied by the shooter genre!👀

# <div class="alert alert-block alert-warning">
#    <h2>Reviewer's commentary</h2>
#     <br/>
# <b>Remarks📌:</b>
# 
# In terms of total sales, the Action genre is really in the leader, this is because, in general, the number of games sold in this genre surpasses all others. But if you look at the average in sales, Shooter will take the leading position, and Action will be inferior to this genre.<br>

# <div class="alert alert-block alert-info">
# <h2>Student's commentary (me!):</h2>
#  I have corrected the conclusions in accordance with the relevant period.
# Thanks for the comment!✔️</div>

# <div class="alert alert-block alert-success">
#     <h2>Reviewer's commentary:second iteration of the review</h2>
#     <br/>
# <b>Well done!👍:</b> 
#     
# Zuriel, amazing pie charts! Very beautiful and informative! Great job!

# <div class="alert alert-block alert-success">
#     <h2>Reviewer's commentary:second iteration of the review</h2>
#     <br/>
# <b>Well done!👍:</b> 
#     
# Ok, now the conclusions you got are absolutely correct!

# <a class="anchor" id="step3.g"></a>
# ## Conclusions games by genre
# - **The largest market share genre is the genre "Action" constitutes about 30.5%!**
# - **The largest average sales per genre is the genre "Shotter" with 1,290,000 dollars! .**
# 
# 
# - **The smallest market share genre is the genre "Puzzle" constitutes about just 0.338%**
# - **The smallest average sales per genre is the genre "Adventure" with just 100,000 dollars.**
# 

# # Conclusions Step 3.<a class="anchor" id="step3.3"></a> 
# 
# **After conducting an in-depth investigation of the data we conclude a number of things:**
# 
# 1. **The years in which the most games were released are 2007-2008. Maybe there is a connection between this and the launch of new  platforms?**
# 
# 2. **We found that the average time that a platform "survives" in the market is about 10 years, it is already recommended to start looking to invest in games that play them on platforms that are in progress (4 years).**
# 
# 3. **Based on the analysis, we have seen that the relevant period for the platform in the market is 4 years and this is the period in which it is correct to check the data.**
# 
# 4. **here is no strong connection between reviews and sales, we can think of how to elevate it to profit in our meeting on the campaign.**
# 
# 5. **The best-selling genre is "Action", we can start and check which games can be invested in which they are considered in this genre. Alternatively, we can actually look at an unsaturated market, and perhaps invest in genres that fail to sell.**
# 
# 
# After all the knowledge we have accumulated it is possible to build more profiled profiles based on regions:
# 

# ### Step 3 Readiness Checklist
# 
# - [X] Look at how many games were released in different years. Is the data for every period significant?
# - [X] Look at how sales varied from platform to platform. 
# - [X] Choose the platforms with the greatest total sales and build a distribution based on data for each year. 
# - [X] Find platforms that used to be popular but now have zero sales. How long does it generally take for new platforms to appear and old ones to fade?
# - [X] Determine what period you should take data for. To do so, look at your answers to the previous questions. The data should allow you to build a prognosis for 2017.
# - [X] Work only with the data that you've decided is relevant. Disregard the data for previous years.
# - [X] Which platforms are leading in sales? Which ones are growing or shrinking? Select several potentially profitable platforms.
# - [X] Build a box plot for the global sales of all games, broken down by platform. Are the differences in sales significant? What about average sales on various platforms? Describe your findings.
# - [X] Take a look at how user and professional reviews affect sales for one popular platform (you choose). Build a scatter plot and calculate the correlation between reviews and sales. Draw conclusions.
# - [X] Keeping your conclusions in mind, compare the sales of the same games on other platforms.
# - [X] Take a look at the general distribution of games by genre. What can we say about the most profitable genres? Can you generalize about genres with high and low sales?

# <a class="anchor" id="step4."></a> 
# # Step 4. Create a user profile for each region

# **We will look for high-profit parameters video game features with the help of the following analysis-**

# <a class="anchor" id="step4.a"></a> 
# ## A) Top 5 profitable platform by regions-
# 
# **We will create a profile for each region according to the sum of sales over the years by regions:<br>**
# We will create this in a convenient way-

# In[56]:


EU = df_games_filtered.pivot_table('eu_sales', columns='platform', aggfunc='sum').T
EU = EU.sort_values(by='eu_sales', ascending=False).iloc[0:5]
EU_platform = EU.index

JP = df_games_filtered.pivot_table('jp_sales', columns='platform', aggfunc='sum').T
JP = JP.sort_values(by='jp_sales', ascending=False).iloc[0:5]
JP_platform = JP.index

NA = df_games_filtered.pivot_table('na_sales', columns='platform', aggfunc='sum').T
NA = NA.sort_values(by='na_sales', ascending=False).iloc[0:5]
NA_platform = NA.index

Other = df_games_filtered.pivot_table('other_sales', columns='platform', aggfunc='sum').T
Other = Other.sort_values(by='other_sales', ascending=False).iloc[0:5]
Other_platform = Other.index

Total = df_games_filtered.pivot_table('total_sales', columns='platform', aggfunc='sum').T
Total = Total.sort_values(by='total_sales', ascending=False).iloc[0:5]
Total_platform = Total.index


# In[57]:


# Initialize figure
fig = go.Figure()

# Add Traces

fig.add_trace(
    go.Bar(y=NA['na_sales'],
           x=NA_platform,
           name="North America",
          marker={'color': NA['na_sales'],'colorscale': 'burgyl'}))
fig.add_trace(
    go.Bar(y=EU['eu_sales'],
           x=EU_platform,
           name="Europe",
           marker={'color': EU['eu_sales'],'colorscale': 'burgyl'},
           visible=False))
fig.add_trace(
    go.Bar(y=JP['jp_sales'],
           x=JP_platform,
           name="Japan",
           marker={'color': JP['jp_sales'],'colorscale': 'burgyl'},
           visible=False))

fig.add_trace(
    go.Bar(y=Other['other_sales'],
           x=Other_platform,
           name="Others",
           marker={'color': Other['other_sales'],'colorscale': 'burgyl'},
           visible=False))

fig.add_trace(
    go.Bar(y=Total['total_sales'],
           x=Total_platform,
           name="Total",
           marker={'color': Total['total_sales'],'colorscale': 'Portland'},
               visible=False ))

fig.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            direction="right",
            active=0,
            x=1,
            y=1.2,
            buttons=list([
                dict(label="North America",
                     method="update",
                     args=[{"visible": [True, False,False, False, False]},
                           {"title": "Top 5 Platforms for North America"}]),
                dict(label="Europe",
                     method="update",
                     args=[{"visible": [False,True, False, False, False]},
                           {"title": "Top 5 Platforms for Europe"}]),
                dict(label="Japan",
                     method="update",
                     args=[{"visible": [False,False, True, False, False]},
                           {"title": "Top 5 Platforms for Japan"}]),
                dict(label="Others",
                     method="update",
                     args=[{"visible": [False,False, False, True, False]},
                           {"title": "Top 5 Platforms for Other Region"}]),
                dict(label="Total",
                     method="update",
                     args=[{"visible": [False,False, False, False, True]},
                           {"title": "Top 5 Platforms for Total"}]),
            ]),
        )
    ])

# Set title
fig.update_layout(
    title_text="Top 5 Platforms per region",
    xaxis_domain=[0.05, 1.0],
    xaxis_title="Platforms",
    yaxis_title="Revenue in $ Milions",
        font=dict(
        family="monospace",
        size=13,
        color="RebeccaPurple")
    )    


fig.show()


# <a class="anchor" id="step4.a.c"></a> 
# 
# ## A) Conclusions platform sales by region
# 
# - **There are differences between the leading platforms between the regions:**
# - In NA leading X360.
# - In Europ leading PS4.
# - And in Japan leading 3DS.
# 
# - **In Japan, Sony is the leading manufacturer, but in first place is Nintendo (with 3DS Platform).**
# - **The top five platforms in Europ are a reflection of the top five in Other regions.

# <div class="alert alert-block alert-warning">
#     <h2>Reviewer's commentary</h2>
#     <br/>
# <b>Remarks📌:</b> 
# 
# As for the most popular platforms in Europe, the USA and Japan, you got not very correct results and again this inaccuracy is connected with the identification of the actual period. Please, try to reconsider your analysis and get the new results. More correct result can look like this one:
#     
# ![image.png](attachment:image.png)

# <div class="alert alert-block alert-info">
# <h2>Student's commentary (me!):</h2>
# I do not know how to get to the data you presented in the picture, but I believe the data I have just presented is correct. (They are now based on the relevant period)✔️</div>

# <div class="alert alert-block alert-success">
#     <h2>Reviewer's commentary:second iteration of the review</h2>
#     <br/>
# <b>Well done!👍:</b> 
#     
# Zuriel, yes, you got the correct result! This picture (table) was only for an example for you, as a landmark! It was created with groupby() method. Decision you provided much more great, difficult and cool!😉

# <a class="anchor" id="step4.b"></a> 
# ## B) Top 5 profitable genres by regions
# 
# **We will create a profile for each region according to the sum of sales over the years by genres:<br>**
# We will create this in a subplots-

# In[58]:


genres_list = df_games_filtered.groupby(['genre'])['total_sales'].sum().sort_values(ascending=False).head(5).index

genres_list = df_games_filtered[df_games_filtered.genre.isin(genres_list)]

fig, (ax0,ax1) = plt.subplots(2,2, figsize=(17,10))
fig.suptitle('Top 5 Genres and their Sales (in Millions) Respective to their region', fontsize=20, fontweight = 'bold', y=1.03)


sns.lineplot(x='year_of_release', y='na_sales', hue='genre', data=genres_list, ci=None, ax=ax0[0], palette='Set1',marker="o")

sns.lineplot(x='year_of_release', y='eu_sales', hue='genre', data=genres_list, ci=None, ax=ax0[1], palette='Set1',marker="o")

sns.lineplot(x='year_of_release', y='jp_sales', hue='genre', data=genres_list, ci=None, ax=ax1[0], palette='Set1',marker="o")

sns.lineplot(x='year_of_release', y='other_sales', hue='genre', data=genres_list, ci=None, ax=ax1[1], palette='Set1',marker="o")

ax0[0].legend(loc='upper left')
ax0[1].legend(loc='upper left')
ax1[0].legend(loc='upper left')
ax1[1].legend(loc='upper left')

ax0[0].set_ylim(-0.1,1)
ax0[1].set_ylim(-0.1,1)
ax1[0].set_ylim(-0.1,1)
ax1[1].set_ylim(-0.1,1)

ax0[0].set_xlim(2012,2016)
ax0[1].set_xlim(2012,2016)
ax1[0].set_xlim(2012,2016)
ax1[1].set_xlim(2012,2016)

ax0[0].set_ylabel('NA Sales (in Millions)', fontsize=16)
ax0[1].set_ylabel('EU Sales (in Millions)', fontsize=16)
ax1[0].set_ylabel('Japan Sales (in Millions)', fontsize=16)
ax1[1].set_ylabel('Other Sales (in Millions)', fontsize=16)

ax0[0].set_title('North America', fontsize=16)
ax0[1].set_title('Europ', fontsize=16)
ax1[0].set_title('Japan', fontsize=16)
ax1[1].set_title('Other', fontsize=16)

ax0[0].set_xlabel('Year', fontsize=13)
ax0[1].set_xlabel('Year', fontsize=13)
ax1[0].set_xlabel('Year', fontsize=13)
ax1[1].set_xlabel('Year', fontsize=13)

plt.tight_layout()
plt.show()


# <a class="anchor" id="step4.b.c"></a> 
# 
# ## B) Conclusions platform sales by genre
# 
# - **There are differences between the leading platforms between the regions:**
# - In NA leading "Shooter".
# - In Europ leading "Shooter".
# - And in Japan leading "Role-Playing".
# 
# - **While in Europe and North America the leading genres are "Shooter","Sports", in Japan these genres are the least profitable, quite the opposite.**
# - **The most profitable genres in Japan are "Role-Playing"**
# 

# <div class="alert alert-block alert-success">
#     <h2>Reviewer's commentary</h2>
#     <br/>
# <b>Well done!👍:</b> 
# 
# Yes, here you are absolutely right!

# ## C) Check if the ESRB rating affect sales by regions -<a class="anchor" id="step4.c"></a> 
# 
# 
# 
# **We will create a profile for each region according to the ESRB rating by sales:<br>**
# We will create this in a 100% interactive stack -

# In[59]:


esrb = df_games_filtered.pivot_table(index = 'rating', values = ['eu_sales','na_sales','jp_sales','other_sales'], aggfunc = 'sum')


# In[60]:


rating = ['rating'].unique()

na_sales=[]
eu_sales=[]
jp_sales=[]
other_sales=[]
total_sales=[]
for i in rating:
    val=df_games_filtered[df_games_filtered.rating==i]
    na_sales.append(val.na_sales.sum())
    eu_sales.append(val.eu_sales.sum())
    jp_sales.append(val.jp_sales.sum())
    other_sales.append(val.other_sales.sum())


# In[61]:


fig = go.Figure()
fig.add_trace(go.Bar(x=na_sales,
                     y=rating,
                     name='North America Sales',
                     marker_color='teal',
                     orientation='h'))

fig.add_trace(go.Bar(x=eu_sales,
                     y=rating,
                     name='Europe Sales',
                     marker_color='purple',
                     orientation='h'))

fig.add_trace(go.Bar(x=jp_sales,
                     y=rating,
                     name='Japan Sales',
                     marker_color='gold',
                     orientation='h'))

fig.add_trace(go.Bar(x=other_sales,
                     y=rating,
                     name='Other Region Sales',
                     marker_color='deepskyblue',
                     orientation='h'))

fig.update_layout(title_text='Regions Total Sales by ESRB Rating',
                  xaxis_title="Sales in $M",
                  yaxis_title="Rating Code",
                  barmode='overlay')

fig.show()
#     ['stack', 'group', 'overlay', 'relative']


# <a class="anchor" id="step4.c.c"></a> 
# ## Conclusions sales by ESRB rating
# 
# First let's make an order:<br>
# E –    Everyone<br>
# M –    Mature<br>
# T –    Teen<br>
# E10+ – Everyone 10 and Older<br>
# EC –   Early Childhood<br>
# Ao –   Adults only<br>
# RP –   Rating Pending<br>
# K-A –  Kids to Adults<br>
# 
# **From the diagram it can be concluded that:**
# 
# **North America-<br>**
#     The most purchased category is - M<br>
#     The least bought category is - T<br>
# 
# **Europe-<br>**
#     The most purchased category is - M<br>
#     The least bought category is - T<br>
# 
# **Japan-<br>**
#     The most purchased category is - E<br>
#     The least bought category is - E + 10<br>
# 
# **It can be assumed that the category definitely affects whether to buy the game or not.**
# 
# Regarding the Unknown category - the interesting statistic is that in the three areas sales are quite similar. (Between 92M and 109M).

# <div class="alert alert-block alert-warning">
#     <h2>Reviewer's commentary</h2>
#     <br/>
# <b>Remarks📌:</b> 
# 
# It is very important to pay attention to the games that have no rating. If we do not take into account unrated games, a huge part of games simply drops out of analysis. And it is important to find a way to take such games into account when grouping values and building analytics at this step.

# <div class="alert alert-block alert-info">
# <h2>Student's commentary (me!):</h2>
# Thanks for the comment!
# After filling in the missing values unknown, I got a more real picture about the effect of the ESRB rating on sales.✔️</div>

# <div class="alert alert-block alert-success">
#     <h2>Reviewer's commentary:second iteration of the review</h2>
#     <br/>
# <b>Well done!👍:</b> 
#     
# Ok, great job! The devil is in the detail!🙄

# <a class="anchor" id="step4.4"></a> 
# 
# # Conclusions Step 4.
# ### We created profiles based on various parameters from the data:
# 
# 1. **It seems that the best-selling platform is Sony's PS4.**
# 2. **The most played genre is "Shooter" while in Japan it is "Role Playing"**
# 3. **The M-rated game in NA and EU is the best-selling.**
# 
# ### We can already say that we have reached the mature audience in NA and the EU, while in Japan the target audience will actually be wider.
# ***And that's a pretty strong conclusion, we've created characteristics for a profitable game, we have something to tell the boss!***
# 

# ### Step 4 Readiness Checklist
# 
# - [X] For each region (NA, EU, JP), determine:
#  - [X] The top five platforms. Describe variations in their market shares from region to region.
#  - [X] The top five genres. Explain the difference.
#  - [X] Do ESRB ratings affect sales in individual regions?

# <a class ='anchor' id="step5."></a>
# ## Step 5. Test the following hypotheses:

# **<span style="color:green">In this step we will  examine the  hypotheses on averages ratings.<br> First by some two platforms and second by two genres  , we will examine whether the averages is equal or not.</span>**
# 

# ### General explanation:
# Since we were asked to test the user rating mean in 2  different groups,**the sample from both groups does not have the same amount, and the samples are not interdependent** ,we would like to perform a **hypothesis on the equality of two Population means**  and for two different populations variance.<br>
# With the help of this test we can answer on this question:<br>Is the average user ratings of the Xbox One and PC platforms are the same?<br>
# That depends on the variance of the samples the values are calculated from.<br>
# Instead of basing our comparison on the averages alone, we use the data sets to perform a statistical test.<br>
# Because We've already checked  that the variances is different, and in general - probably the variance will be different in both populations unless the groups examined overlap.
# 
# **So all that left us to do is to write the hypotheses formally:**

# <h3><span style="color:orange">We will now test the first hypotheses:
# </span></h3>

# <a class ='anchor' id="Hypothesis1."></a>
# <h3><span style="color:darkblue">Hypothesis 1: Average user ratings of the Xbox One and PC platforms are the same:.</span></h3>
# <h4><span style="color:green">H0: Average user ratings of the Xbox One and PC platforms are equal</span></h4> 
# <h4><span style="color:red">H1: Average user ratings of the Xbox One and PC platforms are not equal</span></h4><br>
# I choose the critical statistical significance level to be 5%.<br><br>
# I chose to use the t test (or student test)-<br>
# More specifically-ind t test, ind for independent Samples and that is exactly our case.<br>
# In this type of test, we  comparing the average of two independent unrelated groups.
# Meaning, we are comparing samples from two different populations and are testing whether or not they have a different average.

# In[62]:


XOne = df_games[(df_games['platform'] == 'XOne')&(df_games['user_score'] > 0)]['user_score'] #247
PC = df_games[(df_games['platform'] == 'PC')&(df_games['user_score'] > 0)]['user_score'] #974


# In[63]:


sample_1 = XOne
sample_2 = PC

alpha = 0.05 # critical statistical significance level

results = st.ttest_ind(
         sample_1, 
         sample_2,
         equal_var = False )



#show the to samples distributions:

plt.figure(figsize=(16,9))
ax1 = sns.distplot(sample_1, color = 'b')
ax2 = sns.distplot(sample_2, color = 'r')
#
plt.title('Visual T-test of the Hypotheses', size = 20)
plt.xlabel("Distribution of the two samples: (XOne,PC)", size = 15)
plt.ylabel("Frequency", size = 15)
#
b_patch = mpatches.Patch(color='b', label='XOne Critics Distribution')
r_patch = mpatches.Patch(color='r', label='PC Critics Distribution')
plt.legend(handles=[b_patch,r_patch])
#
#Set Vlines:#
plt.axvline(np.mean(sample_1), color='b', linestyle='dotted', linewidth=2.5)
_, max_ = plt.ylim()

plt.text(
        sample_1.mean()-1.5,
        max_ - max_ / 20,
        "XOne Critics Mean:{:.2f}".format(sample_1.mean()),
        color = 'b',
    )

plt.axvline(np.median(sample_1), color='green', linestyle='dotted', linewidth=2.5)
_, max_ = plt.ylim()

plt.text(
        sample_1.median()-1.75,
        max_ - 0.04,
        "XOne Critics Median:{:.2f}".format(sample_1.median()),
        color = 'green',
    )

plt.axvline(np.mean(sample_2), color='r', linestyle='dotted', linewidth=2.5, )
plt.text(
        sample_2.mean() +0.1,
        max_ - 0.02,
        "PC Critics Mean:{:.2f}".format(sample_2.mean()),
        color = 'r',style = "italic"
    );


plt.axvline(np.median(sample_2), color='y', linestyle='dotted', linewidth=2.5, )
plt.text(
        sample_2.median() +0.075,
        max_ - 0.04,
        "PC Critics Median:{:.2f}".format(sample_2.median()),
        color = 'y',style = "italic"
    );

plt.axvline(results.pvalue, color='black', linestyle='dotted', linewidth=2.5, );
plt.text(
        results.pvalue +0.075,
        max_ - 0.04,
        "PValue:{}".format(results.pvalue),
        color = 'black',style = "italic"
    );

plt.axvline(alpha, color='orange', linestyle='solid', linewidth=2.5, );
plt.text(
        alpha+0.075,
        max_ - 0.02,
        "alpha = {}".format(alpha),
        color = 'orange',style = "italic"
    );


#conclusion:


print('The p-value is: ',results.pvalue)


if (results.pvalue < alpha):
        print(results.pvalue,"<",alpha,"\n"+"We reject the null hypothesis (H0).")
else:
        print(results.pvalue,">",alpha,"\n"+"We can't reject the null hypothesis (H0).") 


# <div class="alert alert-block alert-warning">
#     <h2>Reviewer's commentary</h2>
#     <br/>
# <b>Remarks📌:</b> 
# 
# To check if the assumption of homogeneity of variances is met, you can also use the Levene's test, which is implemented in scipy. The method allows you to estimate the equality of variances using the p-level of significance. You can read more about this method at the link: 
# https://medium.com/@kyawsawhtoon/levenes-test-the-assessment-for-equality-of-variances-94503b695a57

# <div class="alert alert-block alert-success">
#     <h2>Reviewer's commentary</h2>
#     <br/>
# <b>Well done!👍:</b> 
# 
# Visualization is really exciting! You are a real master of plotting the graph, you have created a very difficult code for this task! Great job!

# <div class="alert alert-block alert-info">
# <h2>Student's commentary (me!):</h2>
# Thank you so much Arina!!!! 🤩 🥳🤩 🥳🤩 🥳</div>

# <div class="alert alert-block alert-success">
#     <h2>Reviewer's commentary:second iteration of the review</h2>
#     <br/>
# <b>Well done!👍:</b> 
#     
# I am glad to share something interesting and possibly useful for your future work!

# <h3><span  style="color:red">It can be clearly seen that the average user ratings of the Xbox One and PC platforms are not equal<br>
# The PValue “falls” out of the value that we defined (5%) of the distributions .<br>##We reject the null hypothesis.##</span>
# </h3>

# <h3><span style="color:orange">We will now test the second hypotheses:
# </span></h3>

# In this hypothesis we are asked to test if the average user ratings for the Action and Sports genres are different.<br>
# In this case we need again to exclude two 'populations' from the whole Data and test it.<br>
# **The characteristics are similar** -<br> The size of the samples is different, the variance is different (we will check this of course) and there is no dependence between the two samples!<br>**Therefore everything that was said at the beginning of the previous test is also valid now.**
# 
# 
# **We will prepare the data and then test the hypotheses-**

# <a class='anchor' id="Hypothesis2."></a>
# <h3><span style="color:darkblue">Hypothesis 2: Average user ratings for the Action and Sports genres are differents.</span></h3>
# <h4><span style="color:green">H0: The average user ratings for the Action and Sports genres are equal </span></h4> 
# <h4><span style="color:red">H1: The average user ratings for the Action and Sports genres are not equal</span></h4><br>
# I choose the critical statistical significance level to be 15%.<br><br>
# Same here- I chose to use the t test (or student test)-<br>
# More specifically-ind t test, ind for independent Samples and that is exactly our case.<br>
# In this type of test, we  comparing the average of two independent unrelated groups.
# Meaning, we are comparing samples from two different populations and are testing whether or not they have a different average.

# In[64]:


Action = df_games[(df_games['genre'] == 'Action')&(df_games['user_score'] > 0)]['user_score'] #3369
Sports = df_games[(df_games['genre'] == 'Sports')&(df_games['user_score'] > 0)]['user_score'] #2348


# In[65]:


sample_1 = Action
sample_2 = Sports

alpha = 0.15 # critical statistical significance level

results = st.ttest_ind(
         sample_1, 
         sample_2,
         equal_var = False )



#show the to samples distributions:

plt.figure(figsize=(16,9))
ax1 = sns.distplot(sample_1, color = 'g')
ax2 = sns.distplot(sample_2, color = 'b')
#
plt.title('Visual T-test of the Hypotheses', size = 20)
plt.xlabel("Distribution of the two samples: (Action,Sports)", size = 15)
plt.ylabel("Frequency", size = 15)
#
g_patch = mpatches.Patch(color='g', label='Action Critics Distribution')
b_patch = mpatches.Patch(color='b', label='Sports Critics Distribution')
plt.legend(handles=[g_patch,b_patch],loc = (0.81,0.83))
#
#Set Vlines:#
plt.axvline(np.mean(sample_1), color='g', linestyle='dotted', linewidth=2.5)
_, max_ = plt.ylim()

plt.text(
        sample_1.mean()-2,
        max_ - 0.05,
        "Action Critics Mean:{:.2f}".format(sample_1.mean())+"---->",
        color = 'g',
    )

plt.axvline(np.median(sample_1), color='m', linestyle='dotted', linewidth=2.5)
_, max_ = plt.ylim()

plt.text(
        sample_1.median()-1.9,
        max_ - 0.02,
        "Action Critics Median:{:.2f}".format(sample_1.median()),
        color = 'm',
    )

plt.axvline(np.mean(sample_2), color='b', linestyle='dotted', linewidth=2.5, )
plt.text(
        sample_2.mean(),
        max_ - 0.01,
        "<----------Sports Critics Mean:{:.2f}".format(sample_2.mean()),
        color = 'b',style = "italic"
    );


plt.axvline(np.median(sample_2)+0.03, color='y', linestyle='dotted', linewidth=2.5, )
plt.text(
        sample_2.median() +0.07,
        max_ - 0.04,
        "<-----------------Sports Critics Median:{:.2f}".format(sample_2.median()),
        color = 'y',style = "italic"
    );

plt.axvline(results.pvalue, color='black', linestyle='dotted', linewidth=2.5, );
plt.text(
        results.pvalue +0.075,
        max_ - 0.04,
        "PValue:{}".format(results.pvalue),
        color = 'black',style = "italic"
    );

plt.axvline(alpha, color='purple', linestyle='solid', linewidth=2.5, );
plt.text(
        alpha+0.075,
        max_ - 0.02,
        "alpha = {}".format(alpha),
        color = 'purple',style = "italic"
    );




#conclusion:


print('The p-value is: ',results.pvalue)


if (results.pvalue < alpha):
        print(results.pvalue,"<",alpha,"\n"+"We reject the null hypothesis (H0).")
else:
        print(results.pvalue,">",alpha,"\n"+"We can't reject the null hypothesis (H0).") 


# <h3><span  style="color:red">It can be clearly seen that the the average user ratings for the Action and Sports genres are not equal<br>
# The PValue “falls” out of the value that we defined (15%) of the distributions .<br>##We reject the null hypothesis.##</span>
# </h3>

# <a class = 'anchor' id="step5.5"></a>
# ### Conclusions Step 5.
# The findings we found are -
# 1. The claim that the average user ratings of the Xbox One and PC platforms are the same can be rejected.
# 2. The claim that the average user ratings for the Action and Sports genres are the same can be rejected.

# ### Step 5 completion checklist
# 
# - [X] Average user ratings of the Xbox One and PC platforms are the same.
# - [X] Average user ratings for the Action and Sports genres are different.
# - [X] Set the alpha threshold value yourself.
# - [X] How you formulated the null and alternative hypotheses
# - [X] What significance level you chose to test the hypotheses, and why

# <a class = 'anchor' id="step6."></a>
# ## Step 6. Overall conclusions
# **After reviewing the data, we went over the missing values, checked that they were indeed correct, went over and researched the important parameters, and even created a profile for what kind of video game we should invest in next year we came to the following conclusions:**<br>
# ## 1.
# ### The ultimate game investment profile that i recommend to the management board is:
# 
# |**EU and NA**|category        |
# |------       |------          |
# |Genre        |"Shooter"       |
# |Platform     |"PS4" and "X60" |
# |ESRB Rating  |      "M"       |
# 
# |**Japan**    |category        |
# |------       |------          |
# |Genre        |"Role-Play"     |
# |Platform     |"3DS"           |
# |ESRB Rating  |      "E"       |
# 
# ## 2.
# #### It may also be worthwhile to consider an investment option of buying the platforms themselves as they are in high demand in some regions, and may enter the market as a supplier.
# 
# ## 3. 
# #### At the moment the priority is to focus mainly on the large and prifitable regions (NA,EU,JP) and not Other regions, as their relative profit is relatively small.
# 

# <div style="border:solid Chocolate 2px; padding: 40px">
# 
# **The overall review conclusion**
# 
# Zuriel, thank you for submitting your project! I can see that you've worked really hard on it! I'm really impressed this the amount of work you have done. But there are several things in your project which I suppose it is important to fix in order to make your project really ideal! My comments will navigate you!
# 
# 
# **Good things and practices in your project I want to highlight🙂**:
# * I want to point out your code style: it is very well formatted, neat and understandale, you introduction to the project is really amazing, well done!
# * You use different methods for working with your data: you correctly use groupby(),pivot_table(), unique(), query(), corr() and other methods. It is really great that you can apply these methods on practice, keep it up!
# * You worked with the visualization a lot, it is really good that you use plots in your work! You are a master of plotting the graphs, they look amazing!
# 
# 
# **What is worth working on🙄**:
# * Please, try to specify more correctly the actual period for the analysis and choosing of the actual games and fix the conclusions from several parts of analysis that depend on the correct choice of actual period.
# * Pay attention to the step of preprocessing data: it is not quite the right strategy to fill in the missing values where we do not have access to the source of the data and do not know the exact origin of these gaps.
# 
# **Good luck! I will wait for your work for a second iteration of the review!😉**

# <div style="border:solid Chocolate 2px; padding: 40px">
# 
# **The overall review conclusion: second iteration of the review**
# 
# Zuriel, thank you for submitting your project with corrections!
# 
# It's great that you left your comments in response to mine, your reaction is very valuable, I was very pleased to review your project! You managed to improve your work significantly, now it is really perfect! I hasten to inform you that your project has been accepted and you can proceed to the next sprint!
# 
#     
# **I wish you exciting and cool projects in the next sprints😉**
# 
# ![gif](https://media.giphy.com/media/l49JHz7kJvl6MCj3G/giphy.gif)    

# ### Project completion checklist
# 
# - [X] How do you describe the problems you identify in the data?
# - [X] How do you prepare a dataset for analysis?
# - [X] How do you build distribution graphs and how do you explain them?
# - [X] How do you calculate standard deviation and variance?
# - [X] Do you formulate alternative and null hypotheses?
# - [X] What methods do you apply when testing them?
# - [X] Do you explain the results of your hypothesis tests?
# - [X] Do you follow the project structure and keep your code neat and comprehensible?
# - [X] Which conclusions do you reach?
# - [X] Did you leave clear, relevant comments at each step?
