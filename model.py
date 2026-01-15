import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from pickle import dump

data = pd.read_csv('Dataset/Crop_recommendation.csv')
print(data.head())
data.shape

x = data.iloc[:,:-1] #features
y = data.iloc[:,-1] #labels

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model = RandomForestClassifier()
model.fit(x_train,y_train)

predictions = model.predict(x_test)
accuracy = model.score(x_test,y_test)
print("Accuracy : ",accuracy)

#generate a pickle file (load)
f = open("model.pkl", "wb")
dump(model, f)
f.close()
print("model saved")