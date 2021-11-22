import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump,load

csv_file = pd.read_csv('dataset.csv')

#  Useful prints to analyze the data
# print(csv_file.values)
# print(csv_file.describe())

data = csv_file.drop(columns=['music'])
results =  csv_file['music']
data_train, data_test, results_train, results_test = train_test_split(data, results, test_size=0.2)

#  In order to train the model
# model = DecisionTreeClassifier()
# model.fit(data_train, results_train)

#  Loads the model
model = load('music-taste-detector.joblib')


print (data_test)
predictions = model.predict(data_test)
print(predictions)
model_acc = accuracy_score(results_test, predictions)
print(model_acc)