import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


train = pd.read_csv('data/processed/train_cleaned.csv').copy()
test = pd.read_csv('data/processed/test_cleaned.csv').copy()

y = train['Survived']
categorical_features = ['Pclass', 'Sex', 'SibSp', 'Parch']
X_cat_train = pd.get_dummies(train[categorical_features], drop_first=True)
X_cat_test = pd.get_dummies(test[categorical_features], drop_first=True)
numerical_features = ['Age', 'Fare']

X_train = pd.concat([X_cat_train, train[numerical_features]], axis=1)
X_test = pd.concat([X_cat_test, test[numerical_features]], axis=1)

model = LogisticRegression(max_iter=1000, random_state=1)
model.fit(X_train, y)
predictions = model.predict(X_test)
predictions_train = model.predict(X_train)

# Evaluate the model on the training set
accuracy = accuracy_score(train['Survived'], predictions_train)
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion_matrix(train['Survived'], predictions_train))
print("Classification Report:\n", classification_report(train['Survived'], predictions_train))

#output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})
#output.to_csv('logisticReg.csv', index=False)
