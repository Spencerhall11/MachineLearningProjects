#learning how this stuff works
import pandas 

#pandas functions are basically excel inside of python, with a bunch of functionalities


#select the open source data set
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

#reads in the csv from the url to the data variable and seperates each variable based on the presence of the ;
data = pandas.read_csv(url, sep=';')


#look at the first X rows
print("----- Data Preview -----")

#takes the data and displays a number of rows. the .head command will by default take the first 5 rows as well as columns and indexes
print(data.head())

#examine data to locate missing values
print("\n--- Missing Values ---")

#makes a copy of the csv that is all "true". Switches with "false" when the spot is empty 
print(data.isnull().sum())



#determine and show statistics
print("\n--- Data Summary ---")

#examine the data and uses the math functionality in pandas to find mean,standard deviation, minimum,the percentile values for 25%,50%,75% and then the max for each header value.
print(data.describe())

