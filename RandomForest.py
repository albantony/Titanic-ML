from sklearn.ensemble import RandomForestClassifier
import pandas as pd

train = pd.read_csv('data/raw/train.csv').copy()
test = pd.read_csv('data/raw/test.csv').copy()

y = train['Survived']

features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']
X = pd.get_dummies(train[features], drop_first=True)
X_test = pd.get_dummies(test[features], drop_first=True)

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})
print(X.columns)
print(model.feature_importances_)
#output.to_csv('submission2.csv', index=False)
#print("Your submission was successfully saved!")