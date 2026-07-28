import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#create dataFrame
df = sns.load_dataset('titanic')

#Clean data(drop unimportant columns, fill blanks, convert string values to 0/1)
df = df.drop(['alive', 'class','deck', 'embark_town', 'who','embarked'], axis=1)
df['age'] = df['age'].fillna(df['age'].mean())
df['sex'] = (df['sex'] == 'female').astype(int)

#print amount of non-null values per column and data type
print(df.info())

#Assign data and what we're looking for to variables
y = df['survived']
X = df.drop(['survived'], axis = 1)

#Split training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#Create model
model = LogisticRegression(max_iter = 1000)

#train model
model.fit(X_train, y_train)

#Print accuracy
print(model.score(X_train, y_train))
print(model.score(X_test, y_test))
      