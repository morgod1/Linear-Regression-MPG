import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as splitData
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as meanSquaredError, r2_score as rSquareScore
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def parseLine(line):
    parts = line.strip().split(',')
    if len(parts) < 9:
        return None
    
    carName = ','.join(parts[8:]).strip('"')
    
    return parts[:8] + [carName]

def loadAndPreprocessData():
    # Load and parse the data
    filePath = 'auto-mpg.csv'
    columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'modelYear', 'origin', 'carName']

    data = []
    with open(filePath, 'r') as file:
        next(file)  # Skip the header
        for line in file:
            parsed = parseLine(line)
            if parsed:
                data.append(parsed)

    # Create DataFrame
    df = pd.DataFrame(data, columns=columns)

    # Display the first few rows of the dataframe and its info
    print(df.head())
    print(df.info())

    # Convert columns to preferred types
    numericColumns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'modelYear']
    for col in numericColumns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # remove na rows
    df = df.dropna()

    # make categorical
    df['origin'] = df['origin'].astype('category')

    return df

def prepareData(df):
    # selecting only the features that are important
    selectedFeatures = ['displacement', 'cylinders', 'horsepower', 'weight', 'acceleration', 'modelYear']
    target = 'mpg'

    # Prepare features and target
    X = df[selectedFeatures]
    y = df[target]

    # standardize
    scaler = StandardScaler()
    xScaled = scaler.fit_transform(X)

    # splitting into 80/20
    xTrain, xTest, yTrain, yTest = splitData(xScaled, y, test_size=0.2, random_state=42)

    print("Preprocessing completed. Shape of xTrain:", xTrain.shape)
    print("Shape of xTest:", xTest.shape)

    return xTrain, xTest, yTrain, yTest, selectedFeatures

def plotCorrelationMatrix(df, features):
    correlationMatrix = df[features + ['mpg']].corr()
    plt.figure(figsize=(12, 10))
    plt.imshow(correlationMatrix, cmap='coolwarm', aspect='auto')
    plt.colorbar()
    plt.xticks(range(len(correlationMatrix.columns)), correlationMatrix.columns, rotation=45)
    plt.yticks(range(len(correlationMatrix.columns)), correlationMatrix.columns)
    plt.title('Correlation Matrix of Features')
    for i in range(len(correlationMatrix.columns)):
        for j in range(len(correlationMatrix.columns)):
            plt.text(j, i, f"{correlationMatrix.iloc[i, j]:.2f}", ha='center', va='center')
    plt.tight_layout()
    plt.savefig('correlationMatrix.png')
    plt.close()

def train(xTrain, yTrain, xTest, yTest):
    # Use sklearn's LinearRegression
    model = LinearRegression()
    model.fit(xTrain, yTrain)

    # Make predictions
    trainPredictions = model.predict(xTrain)
    testPredictions = model.predict(xTest)

    # Calculate metrics
    trainMse = meanSquaredError(yTrain, trainPredictions)
    testMse = meanSquaredError(yTest, testPredictions)
    trainR2 = rSquareScore(yTrain, trainPredictions)
    testR2 = rSquareScore(yTest, testPredictions)

    print(f"Train MSE: {trainMse:.4f}, R2: {trainR2:.4f}")
    print(f"Test MSE: {testMse:.4f}, R2: {testR2:.4f}")

    return model

def visualizeResults(model, xTest, yTest, selectedFeatures):
    testPredictions = model.predict(xTest)

    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(yTest, testPredictions)
    plt.plot([yTest.min(), yTest.max()], [yTest.min(), yTest.max()], 'r--', lw=2)
    plt.xlabel('Actual MPG')
    plt.ylabel('Predicted MPG')
    plt.title('Actual vs Predicted MPG on Test Set')
    plt.savefig('actualVsPredictedPlot.png')
    plt.close()

    # Print and plot feature importances
    featureImportance = pd.DataFrame({'Feature': selectedFeatures, 'Importance': np.abs(model.coef_)})
    featureImportance = featureImportance.sort_values('Importance', ascending=False)
    print("\nFeature Importances:")
    print(featureImportance)

    plt.figure(figsize=(10, 6))
    plt.bar(featureImportance['Feature'], featureImportance['Importance'])
    plt.title('Feature Importances')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('featureImportances.png')
    plt.close()

def main():
    df = loadAndPreprocessData()
    xTrain, xTest, yTrain, yTest, selectedFeatures = prepareData(df)

    # Plot correlation matrix
    plotCorrelationMatrix(df, selectedFeatures)

    # Train the model and evaluate
    model = train(xTrain, yTrain, xTest, yTest)

    # Visualize results
    visualizeResults(model, xTest, yTest, selectedFeatures)

    print("Analysis completed successfully.")

if __name__ == "__main__":
    main()