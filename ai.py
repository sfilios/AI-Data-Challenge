import pandas
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold


wheat = pandas.read_csv("wheat-2013-supervised.csv")
print(wheat.describe())
print (wheat.head(5))

predictors = ["apparentTemperatureMax", "apparentTemperatureMax", "cloudCover", "dewPoint", "humidity"]

alg=LinearRegression()

kf = KFold(wheat.shape[0], n_folds=3, random_state=1)

predictions = []
for train, test in kf:
	train_predictors = (wheat[predictors].iloc[train,:])
   # The target we're using to train the algorithm.
   	train_target = wheat["Yield"].iloc[train]
   # Training the algorithm using the predictors and target.
	alg.fit(train_predictors, train_target)
   # We can now make predictions on the test fold
	test_predictions = alg.predict(wheat[predictors].iloc[test,:])
	predictions.append(test_predictions)
	
predictions = np.concatenate(predictions, axis=0)

print predictions[1:10]
# Map predictions to outcomes (only possible outcomes are 1 and 0)

#accuracy = sum(predictions[predictions == wheat["Yield"]]) / len(predictions)