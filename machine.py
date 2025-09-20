import pandas

from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.model_selection import train_test_split as trainTestSplit, GridSearchCV, KFold, cross_val_score as crossValScore
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import classification_report

dataFrame = pandas.read_csv('data.csv')

# Feature engineering, drop all the columns that create noise.
trainingData = dataFrame.drop(columns={
   'STAT_CAUSE_DESCR', 
    'STAT_CAUSE_CODE',
    'OWNER_CODE',
    'OWNER_DESCR',
    'FIPS_CODE',
    'FIPS_NAME',
    'MTBS_ID',
    'ICS_209_INCIDENT_NUMBER',
    'NWCG_REPORTING_UNIT_ID',
    'NWCG_REPORTING_UNIT_NAME',
    'FOD_ID',
    'FPA_ID',
    'SOURCE_SYSTEM_TYPE',
    'SOURCE_SYSTEM',
    'SOURCE_REPORTING_UNIT',
    'LOCAL_FIRE_REPORT_ID',
    'LOCAL_INCIDENT_ID',
    'FIRE_YEAR',
    'COUNTY',
    'STATE',
})

# Map the cause to a 0 or 1 -> This is because the metrics work well with numbers.
def classifyCause(cause: str):
    if cause == "Lightning":
        return 0 # Nature
    else:
        return 1 # Human

# Preprocess the data so the model can classify it correctly.
targetData = dataFrame['STAT_CAUSE_DESCR'].apply(classifyCause)
trainingData = pandas.get_dummies(trainingData)
trainingData = trainingData.fillna(0)

# Build the model.
stackedModel = StackingClassifier(
    estimators=[
        ('gradientBooster', GradientBoostingClassifier(n_estimators=50, min_samples_leaf=8, max_depth=10)),
        ('extraTrees', ExtraTreesClassifier(n_estimators=50, min_samples_leaf=4, max_depth=10))
    ],

    n_jobs=-1
)

# Split the data
xTrain, xTest, yTrain, yTest = trainTestSplit(
    trainingData,
    targetData,
    stratify=targetData,
    random_state=42
)

'''
paramGrid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10],
    'min_samples_leaf': [3, 5],
} 

search = GridSearchCV(estimator=GradientBoostingClassifier, param_grid=paramGrid)
search.fit(xTrain, yTrain)
Output: {'max_depth': 10, 'min_samples_leaf': 3, 'n_estimators': 100}

search = GridSearchCV(estimator=ExtraTreesClassifier, param_grid=paramGrid)
search.fit(xTrain, yTrain)
Output: {'max_depth': 10, 'min_samples_leaf': 3, 'n_estimators': 100}
'''

# Trains the data and predicts.
stackedModel.fit(xTrain, yTrain)
forecast = stackedModel.predict(xTest)

# How well did the prediction do?
print(classification_report(yTest, forecast))

# Cross validation using a KFold model.
# Outputs: [0.89333333 0.89333333 0.84       0.888      0.864]
'''print("Cross validating the model.....")
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
print(crossValScore(stackedModel, xTrain, yTrain, cv=kfold))'''

# Creates evaluation metrics and prints them
MAPE = mean_absolute_percentage_error(yTest, forecast)
RMSE = root_mean_squared_error(yTest, forecast)
R2 = r2_score(yTest, forecast)
MAE = mean_absolute_error(yTest, forecast)
print("#########################################")
print("-> The predictions error percentile: " + str(MAPE * 100) + "%")
print("-> The predictions deviation from original values: +-" + str(RMSE) + " units")
print("-> The predictions variance score: " + str(R2 * 100) + "%")
print("-> The predictions average deviation: +-" + str(MAE) + " units")
print("#########################################")

# Exit.