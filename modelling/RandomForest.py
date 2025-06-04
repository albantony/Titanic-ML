from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


train = pd.read_csv('data/processed/train_cleaned.csv').copy()
test = pd.read_csv('data/processed/test_cleaned.csv').copy()

y = train['Survived']

features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Has_Cabin', 'FamilySize', 'IsAlone', 'CategoricalFare', 'CategoricalAge']
X = pd.get_dummies(train[features], drop_first=True)
X_test = pd.get_dummies(test[features], drop_first=True)
X_test = X_test.reindex(columns=X.columns, fill_value=0)

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)
predictions_train = model.predict(X)

output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})
print(X.columns)
print(model.feature_importances_)

# Evaluate the model on the training set
accuracy = accuracy_score(train['Survived'], predictions_train)
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion_matrix(train['Survived'], predictions_train))
print("Classification Report:\n", classification_report(train['Survived'], predictions_train))

output.to_csv('data\submissions\RandomForest_submission.csv', index=False)
print("Your submission was successfully saved!")