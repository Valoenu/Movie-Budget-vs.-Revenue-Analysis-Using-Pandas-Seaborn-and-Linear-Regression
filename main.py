import pandas as pd
import numpy as np

# Import necessary libraries for visualization and chart creation
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# Remember: use .values.any() to check if *any* value in the DataFrame is True

df = pd.read_csv('cost_revenue_dirty.csv')

df.head()  # Preview the first few rows of the data
df.tail()  # Preview the last few rows of the data

df.shape  # Check number of rows and columns in the dataset

df.isna().values.any()  # Check for any NaN (missing) values in the dataset

df.dropna()  # Remove rows with missing values (returns new DataFrame, not in-place)

df[df.duplicated().values.any()]  # Check if any rows are duplicated (Incorrect usage, will be corrected below)

rows_duplicated = df[df.duplicated()]  # Store duplicated rows

df.dtypes  # Check data types of each column

df.info()  # Get concise summary of the DataFrame (including datatypes and nulls)


# Convert important columns to numeric format by removing '$' signs and commas
# (Used to clean financial columns before analysis)

unnecesery_characters = [',', '$']
columns_cleaning = ['USD_Production_Budget', 
                    'USD_Worldwide_Gross',
                    'USD_Domestic_Gross']

for column in columns_cleaning:
    for character in unnecesery_characters:
        df[column] = df[column].astype(str).str.replace(character, "")
        
    df[column] = pd.to_numeric(df[column])  # Convert to numeric data type


# Convert the 'Release_Date' column to Pandas datetime format
df['Release_Date'] = pd.to_datetime(df['Release_Date'])

df.describe()  # View summary statistics (mean, min, max, etc.)


# Which film had the lowest production budget?
# We can see from df.describe that the lowest budget was $1,100
df[df.USD_Production_Budget == 1100.00]


# Find the movie with the highest production budget
df[df.USD_Production_Budget == 425000000.00]


# Count how many movies made $0 in domestic revenue
zero_revenue_us = df[df.USD_Domestic_Gross == 0]
print(f'Number of movies that earned $0 domestically: {len(zero_revenue_us)}')

zero_revenue_us.sort_values('USD_Production_Budget', ascending=False)  # Sort by budget descending


# Count how many movies made $0 in worldwide revenue
zero_revenue_worldwide = df[df.USD_Worldwide_Gross == 0]
print(f'Number of movies that earned $0 worldwide: {len(zero_revenue_worldwide)}')

zero_revenue_worldwide.sort_values('USD_Production_Budget', ascending=False)  # Sort by budget descending


# Use multiple conditions (e.g. & for "and") in one line

# First method (using .query)
# Query movies that made $0 domestically but earned revenue worldwide
international = df.query('USD_Domestic_Gross == 0 and USD_Worldwide_Gross != 0')


# Second method (using .loc)
international = df.loc[(df.USD_Domestic_Gross == 0) & 
                       (df.USD_Worldwide_Gross != 0)]


# Define the current date used during data collection
df_date = pd.Timestamp('2018-5-1')

# Find films not released yet at the time of data collection
not_released_yet = df[df.Release_Date >= df_date]

# How many films were unreleased at the time of analysis?
not_released_yet.shape
not_released_yet.info()  # Get information on unreleased films


# Create a new DataFrame excluding unreleased films
clean_dataframe = df.drop(not_released_yet.index)  # .index to drop by row number


# Find movies where the budget is higher than the total worldwide revenue
movies_with_losing = clean_dataframe.loc[
    clean_dataframe.USD_Production_Budget > clean_dataframe.USD_Worldwide_Gross]

# Calculate the proportion of losing movies
len(movies_with_losing) / len(clean_dataframe)


# Visualize data

# Create a scatterplot using seaborn
plt.figure(figsize=(15, 7), dpi=130)

with sns.axes_style('white'):  # Styling the plot using seaborn themes
    scatterplot = sns.scatterplot(data=clean_dataframe,
                                  y='USD_Worldwide_Gross',
                                  x='USD_Production_Budget',
                                  size='USD_Worldwide_Gross',  # Dot size reflects revenue
                                  hue='USD_Worldwide_Gross')   # Dot color reflects revenue

    scatterplot.set(xlim=(0, 400000000),  # X-axis range
                    ylim=(0, 300000000),  # Y-axis range
                    xlabel='Total Budget in $100M scale',  # X-axis label
                    ylabel='Total Revenue in $100M scale')  # Y-axis label

plt.show()


# Second chart with 3D-style data representation (scatterplot)
plt.figure(figsize=(8, 4), dpi=200)

with sns.axes_style("darkgrid"):
    second_chart = sns.scatterplot(data=clean_dataframe,
                                   x='Release_Date', 
                                   y='USD_Production_Budget',
                                   hue='USD_Worldwide_Gross',
                                   size='USD_Worldwide_Gross')

    second_chart.set(ylim=(0, 450000000),
                     xlim=(clean_dataframe.Release_Date.min(), 
                           clean_dataframe.Release_Date.max()),  # Get min and max dates
                     xlabel='Release Year',
                     ylabel='Budget in $100M scale')


# Convert release date into decades and add a new column

index_dataframe = pd.DatetimeIndex(clean_dataframe['Release_Date'])  # Convert to datetime index
years = index_dataframe.year  # Extract year

# Create decade from year using integer division
decades_date = (years // 10) * 10

# Add a new column for decades
clean_dataframe['Decades_date'] = decades_date


# Filter new and old movies based on year
new_movies = clean_dataframe[clean_dataframe.Release_Date.dt.year > 1990]
old_movies = clean_dataframe[clean_dataframe.Decades_date <= 1990]


# Show statistical summary for new movies
new_movies.describe()

# Sort and show top 5 most expensive new movies
new_movies.sort_values('USD_Production_Budget', ascending=False).head(5)

# Show 5 lowest-budget new movies
new_movies.sort_values('USD_Production_Budget').head(5)


# Train a univariate linear regression model

# Relationship between budget and revenue (new movies only)
plt.figure(figsize=(15, 8), dpi=120)

with sns.axes_style('white'):
    regression_chart = sns.regplot(data=new_movies,
                                   x='USD_Production_Budget',
                                   y='USD_Worldwide_Gross',
                                   line_kws={'color': '#4d4d4d'},  # Regression line color
                                   color='#1f77b4',                # Dot color
                                   scatter_kws={'alpha': 0.5})     # Dot transparency

    regression_chart.set(xlim=(0, 400000000),
                         ylim=(0, 300000000),
                         xlabel='Total Budget in $100M scale',
                         ylabel='Total Revenue in $Billions')


# Analyze results
# The y-intercept represents estimated revenue if the budget were zero
# The slope indicates the revenue increase per $1 budget increase


# Create and fit a regression model (using sklearn)
regr = LinearRegression()

# Define features (X) and target (y)
X = pd.DataFrame(new_movies, columns=['USD_Production_Budget'])
y = pd.DataFrame(new_movies, columns=['USD_Worldwide_Gross'])

regr.fit(X, y)  # Fit the regression model

regr.score(X, y)  # R² value - how well the model explains the variance

regr.coef_  # Theta one (slope)
regr.intercept_  # Theta zero (intercept)


# Linear regression for old movies
old_movies_X = pd.DataFrame(old_movies, columns=['USD_Production_Budget'])
old_movies_y = pd.DataFrame(old_movies, columns=['USD_Worldwide_Gross'])

regr.fit(old_movies_X, old_movies_y)

# Display regression results
print(f'The R-squared value is: {regr.score(old_movies_X, old_movies_y)}')
print(f'The intercept is: {regr.intercept_[0]}')
print(f'The slope coefficient is: {regr.coef_[0][0]}')


# Make a prediction

budget = 100000000  # Example input budget value in USD

# Estimate revenue using the regression equation: REVENUE = intercept + slope * budget
estimate = regr.intercept_[0] + regr.coef_[0][0] * budget
estimate = round(estimate, -6)  # Round to nearest million

print(f'The estimated revenue for a ${budget} film is around ${estimate:.0f}.')  # Display result