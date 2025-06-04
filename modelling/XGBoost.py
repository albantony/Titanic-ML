import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


train = pd.read_csv('data/processed/train_cleaned.csv')
test = pd.read_csv('data/processed/test_cleaned.csv')

drop_columns = ['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin', 'Age','Fare']
drop_test = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Age','Fare']


y_train = train['Survived']
X = train.drop(drop_columns, axis=1)
X_test = test.drop(drop_test, axis=1)
x_train = X.values # Creates an array of the train data
x_test = X_test.values 

gbm = xgb.XGBClassifier(
    #learning_rate = 0.02,
    n_estimators= 2000,
    max_depth= 4,
    min_child_weight= 2,
    #gamma=1,
    gamma=0.9,                        
    subsample=0.8,
    colsample_bytree=0.8,
    objective= 'binary:logistic',
    nthread= -1,
    scale_pos_weight=1).fit(x_train, y_train)

xgb_predictions = gbm.predict(x_test)
predictions_train = gbm.predict(x_train)

PassengerId = test['PassengerId']
StackingSubmission = pd.DataFrame({ 'PassengerId': PassengerId, 'Survived': xgb_predictions })

# Evaluate the model on the training set
accuracy = accuracy_score(train['Survived'], predictions_train)
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion_matrix(train['Survived'], predictions_train))
print("Classification Report:\n", classification_report(train['Survived'], predictions_train))

#StackingSubmission.to_csv('data/submissions/XGBoost_submission.csv', index=False)

