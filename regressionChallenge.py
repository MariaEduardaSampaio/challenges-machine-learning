import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib


def show_distribution(var_data):
    min_val = var_data.min()
    max_val = var_data.max()
    mean_val = var_data.mean()
    med_val = var_data.median()
    mod_val = var_data.mode()[0]

    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    ax[0].hist(var_data)
    ax[0].set_ylabel('Frequency')

    ax[0].axvline(x=min_val, color='green', linestyle='dashed', linewidth=2)
    ax[0].axvline(x=mean_val, color='yellow', linestyle='dashed', linewidth=2)
    ax[0].axvline(x=med_val, color='cyan', linestyle='dashed', linewidth=2)
    ax[0].axvline(x=mod_val, color='gray', linestyle='dashed', linewidth=2)
    ax[0].axvline(x=max_val, color='red', linestyle='dashed', linewidth=2)

    ax[1].boxplot(var_data, vert=False)
    ax[1].set_xlabel('Value')

    fig.suptitle(var_data.name)
    plt.show()


def show_correlations(data):
    label = data[data.columns[-1]]

    for col in data[data.columns]:
        fig = plt.figure(figsize=(9, 6))
        ax = fig.gca()
        feature = data[col]
        correlation = feature.corr(label)
        plt.scatter(x=feature, y=label)
        plt.xlabel(col)
        plt.ylabel('Correlations')
        ax.set_title('Label vs' + col + '- correlation: ' + str(correlation))

    plt.show()


def show_metrics(label_test, predictions):
    mse = mean_squared_error(label_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(label_test, predictions)
    print('MSE: %f\nRMSE: %f\nR2: %f' % (mse, rmse, r2))


def evaluate_model(label_test, predictions):
    plt.scatter(label_test, predictions)
    plt.xlabel('Actual Labels')
    plt.ylabel('Predicted Labels')
    plt.title('Predictions vs Actuals')
    z = np.polyfit(label_test, predictions, 1)
    p = np.poly1d(z)
    plt.plot(label_test, p(label_test), color='red')
    plt.show()


# Load the training dataset
df_houses = pd.read_csv('data/real_estate.csv')
print(df_houses.head())

# Cleaning data
print(df_houses.isnull().sum())

# Eliminating outliers
num_fields = ['transit_distance', 'local_convenience_stores', 'price_per_unit']

# first let's look at some graphs, so we make a better analysis
for col in num_fields:
    show_distribution(df_houses[col])

price_95pcntile = df_houses.price_per_unit.quantile(0.95)
df_houses = df_houses[df_houses.price_per_unit < price_95pcntile]

show_distribution(df_houses.price_per_unit)

# View numeric correlations
# show_correlations(df_houses)

categ_fields = df_houses[['transaction_date', 'local_convenience_stores']]
label_field = ['price_per_unit']

'''
for col in categ_fields:
    fig = plt.figure(figsize=(9, 6))
    ax = fig.gca()
    df_houses.boxplot(column=label_field, by=col, ax=ax)
    ax.set_title('Label by ' + col)
    ax.set_ylabel('Label Distribution by Categorical variable')
    plt.xticks(rotation=90)

plt.show()
'''

# Separate features and label

# did not include transaction date
features = df_houses[df_houses.columns[1:-1]].values
label = df_houses[df_houses.columns[-1]].values

feat_train, feat_test, label_train, label_test = train_test_split(features, label, test_size=0.3, random_state=0)
print('Training set: %d, rows\nTest Set: %d rows' % (feat_train.shape[0], feat_test.shape[0]))

# Preprocess the data and train a model in pipeline

numeric_features = [0, 1, 3, 4]
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features)])

# Training pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', RandomForestRegressor())])

# Fitting pipeline to train a Linear Regression model on the training set
model = pipeline.fit(feat_train, label_train)
print(model)

# Evaluating model
predictions = model.predict(feat_test)
show_metrics(label_test, predictions)
evaluate_model(label_test, predictions)

# Using the trained model

filename = './real_estate_model.pkl'
joblib.dump(model, filename)

loaded_model = joblib.load(filename)

# do not include the transaction date
new_data = np.array([[10.5, 291.83, 4, 24.98297, 121.5395],
                     [6.25, 1621.49, 2, 24.98033, 121.5401]])

results = loaded_model.predict(new_data)
print('Predictions: ')
for prediction in results:
    print(round(prediction, 2))
