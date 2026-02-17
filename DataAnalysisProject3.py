#learning how this stuff works, adding graphs
import pandas 
import matplotlib.pyplot as plt
import seaborn as sns

#pandas functions are basically excel inside of python, with a bunch of functionalities
#matplotlib is used to graph everything
#seaborn is a specialized python tool to create statistical graphs

#select the open source data set
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

#reads in the csv from the url to the data variable and seperates each variable based on the presence of the ;
data = pandas.read_csv(url, sep=';')

#look at the first X rows
print("----- Data Preview -----")

#takes the data and displays a number of rows. the .head() command will by default take the first 5 rows as well as columns and indexes
print(data.head())

#examine data to locate missing values
print("\n--- Missing Values ---")

#makes a copy of the csv that is all "false". Switches with "true" when the spot is empty 
print(data.isnull().sum())

#determine and show statistics
print("\n--- Data Summary ---")

#examine the data and uses the math functionality in pandas to find mean,standard deviation(how spread out the data is)
#Shows the minimum,the percentile values for 25%,50%,75% and then the max for each header value.
print(data.describe())


#section to visualize the data in a graph

#make a boxplot, good for discrete X values
#make heatmap size and initialize the object
plt.figure(figsize=(8,5))
#draw the boxplot
sns.boxplot(x='quality',y='alcohol',data=data,palette='vlag')
#name the boxplot
plt.title('Alcohol Content Spread by Wine Quality')
#show the graph
plt.show()

#make a scatter plot to see if there's a tie between alcohol content and quality
sns.scatterplot(x='quality',y='alcohol',data=data)
#make the title
plt.title('Alcohol Content vs Quality')
#show the graph
plt.show()

#correlation heatmap
#make the heatmap size to avoid overlap
plt.figure(figsize=(10,8))
#data,corr calculations the correlation coefficient of all columns
sns.heatmap(data.corr(),annot=True,fmt=".2f",cmap='coolwarm')
#name the graph
plt.title('Correlation Heatmap of Wine Features')
#show the graph
plt.show()