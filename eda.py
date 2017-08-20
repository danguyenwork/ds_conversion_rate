import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn

df = pd.read_csv('conversion_data.csv')
df.head()

df.info()
# No null values. Data types are correct.

df.country.unique()
df.age.unique()
df.total_pages_visited.unique()
df.source.unique()
df.new_user.unique()
df.converted.unique()

df.describe()
# Max age is 123 which does not seem right. A lot of people don't like to put their real birthday or find it annoying to do so. This could be due to such random entry. Should investigate histogram and potentially excluded anything above a certain threshold.

# Only 3% converted. This is an extremely unbalanced data set. Some type of sampling is likely needed to amplify the signal of the positive class.

plt.hist(df.age, bins=10)
plt.show()
plt.savefig('age_hist.png')

df.boxplot(column='age')
plt.savefig('age_boxplot.png')

df[df.age > 60].describe()
df.shape

237 * 1. / df.shape[0] * 100

# clear outliers for age > 60
df = df[df.age < 60]
df['cnt'] = 1
df.head()

# data cleaning for age

df.boxplot(column='age')
plt.show()

plt.hist(df.age, bins=10)
plt.show()

# conversion rate vs country
# germany and UK have the highest conversion rate. US is slightly behind. China is way behind
df_country = df[['country','total_pages_visited','converted','cnt']].groupby('country').sum().reset_index()
df_country['user_ctr'] = df_country['converted'] * 1. / df_country['cnt']
df_country['page_ctr'] = df_country['converted'] * 1. / df_country['total_pages_visited']
df_country


# conversion rate vs age
# there seems to be a linear relationship between the log age and the conversion rate
df_age = df[['age','total_pages_visited','converted','cnt']].groupby('age').sum().reset_index()
df_age['user_ctr'] = df_age['converted'] * 1. / df_age['cnt']
df_age['page_ctr'] = df_age['converted'] * 1. / df_age['total_pages_visited']
plt.scatter(np.log(df_age.age), df_age.user_ctr)
plt.xlabel('Age (log)')
plt.ylabel('CVR')
plt.show()
plt.savefig('logage_vs_conversion.png')
plt.scatter(df_age.age, df_age.user_ctr)
plt.xlabel('Age')
plt.ylabel('CVR')
plt.savefig('age_vs_conversion.png')



# conversion rate vs. source
# ads and seo seem to perform better than direct
df_source = df[['source','total_pages_visited','converted','cnt']].groupby('source').sum().reset_index()
df_source['user_ctr'] = df_source['converted'] * 1. / df_source['cnt']
df_source['page_ctr'] = df_source['converted'] * 1. / df_source['total_pages_visited']
df_source
plt.show()

# conversion rate vs. page visited
# there is a very clear relationship between number of pages viewed and conversion rate
# users with less than 7 page views tend to not convert
# users with more than 19 page views tend to converted
# we can create three bins: 0-7, 8-18, 19-up

df_page = df[['total_pages_visited','converted','cnt']].groupby('total_pages_visited').sum().reset_index()
df_page['user_ctr'] = df_page['converted'] * 1. / df_page['cnt']
df_page['page_ctr'] = df_page['converted'] * 1. / df_page['total_pages_visited']
df_page
plt.show()
plt.scatter(df_page.total_pages_visited, df_page.user_ctr)
plt.xlabel('Pages Visited')
plt.ylabel('CVR')
plt.savefig('age_vs_page.png')
plt.show()
plt.bar(df_page.total_pages_visited, df_page.cnt)
plt.xlabel('Pages Visited')
plt.ylabel('User Count')
plt.savefig('page_bar.png')

# conversion rate versus new user
# existing user converts significantly more
df_new_user = df[['new_user','total_pages_visited','converted','cnt']].groupby('new_user').sum().reset_index()
df_new_user['user_ctr'] = df_new_user['converted'] * 1. / df_new_user['cnt']
df_new_user['page_ctr'] = df_new_user['converted'] * 1. / df_new_user['total_pages_visited']
df_new_user
# we can consider the following for feature engineering:
# - take the log transform of age
# - dummify the country, user and source variables
# - divide total pages viewed into 3 bins
