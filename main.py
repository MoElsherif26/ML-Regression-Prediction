import pandas
import seaborn
import matplotlib.pyplot
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np


class LogisticRegression:
    def __init__(self, learningRate = 0.01, numberOfIterations = 1000):
        self.learningRate = learningRate
        self.numberOfIterations = numberOfIterations
        self.weights = None
        self.bias = None

    # the sigmoid function
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # fit the data to the model
    def fit(self, features, targets):
        numberOfSamples, numberOfFeatures = features.shape
        self.weights = np.zeros(numberOfFeatures)
        self.bias = 0

        for _ in range(self.numberOfIterations):
            model = np.dot(features, self.weights) + self.bias
            predictions = self.sigmoid(model)

            gradientOfWeights = (1 / numberOfSamples) * np.dot(features.T, (predictions - targets))
            gradientOfBias = (1 / numberOfSamples) * np.sum(predictions - targets)

            self.weights -= self.learningRate * gradientOfWeights
            self.bias -= self.learningRate * gradientOfBias

    # predict new targets
    def predict(self, features):
        model = np.dot(features, self.weights) + self.bias
        predictions = self.sigmoid(model)
        return np.round(predictions)

    # calculate the accuracy of the model
    def calculateAccuracy(self, actualTargets, predictedTargets):
        correct_predictions = np.sum(actualTargets == predictedTargets)
        totalSamples = len(actualTargets)
        accuracy = correct_predictions / totalSamples
        return accuracy


# load the old data
loanOldData = pandas.read_csv('loan_old.csv')

print("The data before nothing \n")
print(loanOldData)


# analysis the data
missingValues = loanOldData.isnull().sum()

print("missing values per column\n")
print(missingValues)

if missingValues.any():
    print("the data have missing values \n")
else:
    print("the data have not missing values \n")

dataTypes = loanOldData.dtypes

print("\nColumns types:")
print(dataTypes)

numericalFeatures = loanOldData.select_dtypes(include=['int64', 'float64'])

featureStatistics = numericalFeatures.describe().loc[['mean', 'std']]

print("\nMean and Standard Deviation for Numerical Features:")
print(featureStatistics)

seaborn.pairplot(numericalFeatures)
matplotlib.pyplot.show()


# preprocess the data
loanOldDataNoMissing = loanOldData.dropna()

print("\nDataFrame after removing missing values:")
print(loanOldDataNoMissing)

targetColumns = ['Max_Loan_Amount', 'Loan_Status']

features = loanOldDataNoMissing.drop(['Loan_ID', 'Max_Loan_Amount', 'Loan_Status'], axis=1)
targets = loanOldDataNoMissing[targetColumns]

print("\nfeatures")
print(features)

print("\ntargets")
print(targets)

featuresTrain, featuresTest, targetsTrain, targetsTest = train_test_split(
    features, targets, test_size=0.5
)

print("\nTraining set - Features: ", featuresTrain.shape, "Targets: ", targetsTrain.shape)
print("Testing set - Features: ", featuresTest.shape, "Targets: ", targetsTest.shape)

labelEncoder = LabelEncoder()
categoricalColumns = ['Gender', 'Married', 'Education', 'Property_Area', 'Dependents']
for col in categoricalColumns:
    featuresTrain[col] = labelEncoder.fit_transform(featuresTrain[col])
    featuresTest[col] = labelEncoder.transform(featuresTest[col])

targetsTrain['Loan_Status'] = labelEncoder.fit_transform(targetsTrain['Loan_Status'])
targetsTest['Loan_Status'] = labelEncoder.transform(targetsTest['Loan_Status'])

numericalColumns = ['Income', 'Coapplicant_Income', 'Loan_Tenor', 'Credit_History']
scaler = StandardScaler()
featuresTrain[numericalColumns] = scaler.fit_transform(featuresTrain[numericalColumns])
featuresTest[numericalColumns] = scaler.transform(featuresTest[numericalColumns])

print("\ntraining sets")
print(featuresTrain)
print(targetsTrain)

print("\ntesting sets")
print(featuresTest)
print(targetsTest)

# use linear regression
linearRegressionModel = LinearRegression()
linearRegressionModel.fit(featuresTrain, targetsTrain['Max_Loan_Amount'])

targetsPredictions = linearRegressionModel.predict(featuresTest)

r2 = r2_score(targetsTest['Max_Loan_Amount'], targetsPredictions)

print("\nR2 Score of linear model", r2)

# use logistic regression

logisticRegression = LogisticRegression()
logisticRegression.fit(featuresTrain.to_numpy(), targetsTrain['Loan_Status'].to_numpy())

loanStatusPredictions = logisticRegression.predict(featuresTest.to_numpy())

accuracy = logisticRegression.calculateAccuracy(targetsTest['Loan_Status'].to_numpy(), loanStatusPredictions)

print("logistic regression accuracy: ", accuracy)

# load new data
loanNewData = pandas.read_csv('loan_new.csv')

print("\nthe new data before nothing")
print(loanNewData)

missingValuesNew = loanNewData.isnull().sum()
print("\nmissing values per column in the new data")
print(missingValuesNew)

if missingValuesNew.any():
    print("the new data have missing values\n")
else:
    print("the new data have not missing values\n")

dataTypesNew = loanNewData.dtypes
print("Columns types in the new data:")
print(dataTypesNew)

numericalFeaturesNew = loanNewData.select_dtypes(include=['int64', 'float64'])
featureStatisticsNew = numericalFeaturesNew.describe().loc[['mean', 'std']]
print("\nmean and standard deviation for numerical features in the new data\n")
print(featureStatisticsNew)

loanNewDataNoMissing = loanNewData.dropna()

print("\ndataframe after removing missing values in the new data\n")
print(loanNewDataNoMissing)

featuresNew = loanNewDataNoMissing.drop(['Loan_ID'], axis=1)

labelEncoderNew = LabelEncoder()
categoricalColumns_new = ['Gender', 'Married', 'Education', 'Property_Area', 'Dependents']
for col in categoricalColumns_new:
    featuresNew[col] = labelEncoderNew.fit_transform(featuresNew[col])


numericalColumnsNew = ['Income', 'Coapplicant_Income', 'Loan_Tenor', 'Credit_History']
scalerNew = StandardScaler()
featuresNew[numericalColumnsNew] = scalerNew.fit_transform(featuresNew[numericalColumnsNew])

print("\npreprocessed new data ")
print(featuresNew)

# predict targets of new data
loanAmountPredictionsNew = linearRegressionModel.predict(featuresNew)

loanStatusPredictionsNew = logisticRegression.predict(featuresNew.to_numpy())

loanAmountPredictionsDataFrame = pandas.DataFrame({
    'Predicted_Max_Loan_Amount': loanAmountPredictionsNew
})

loanStatusPredictionsDataFrame = pandas.DataFrame({
    'Predicted_Loan_Status': loanStatusPredictionsNew
})

print("\npredictions on the new data are\n")
print("predicted max loan amounts in form of data frame\n")
print(loanAmountPredictionsDataFrame)
print("\npredicted max loan amounts in form of array\n")
print(loanAmountPredictionsNew)
print("\npredicted loan status in form of data frame\n")
print(loanStatusPredictionsDataFrame)
print("\npredicted loan status in form of array\n")
print(loanStatusPredictionsNew)
