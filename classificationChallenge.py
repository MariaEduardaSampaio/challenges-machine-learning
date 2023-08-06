import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, roc_auc_score

'''
"Your challenge is to explore the data and train a classification model that achieves an 
overall Recall metric of over 0.95 (95%)."
'''
# Load the training dataset
df_wines = pd.read_csv('data/wine.csv')
print(df_wines.head())

# Preparing and analysing dataset
features = list(df_wines.columns[:-1])
label = df_wines.columns[-1]

for col in features:
    df_wines.boxplot(column=col, by=label, figsize=(6, 6))
    plt.title(col)
plt.show()

# Splitting data 70%-30% into training and test set
features_values = df_wines[features]
label_values = df_wines[label]

feat_train, feat_test, label_train, label_test = train_test_split(features_values,
                                                                  label_values,
                                                                  test_size=0.3,
                                                                  random_state=0)
print('Training cases: %d\nTest cases: %d' % (feat_train.shape[0], feat_test.shape[0]))

# Defining preprocessing for numeric columns, scaling them
feature_columns = [0, 1, 2, 3, 4, 5, 6]
feature_transformer = Pipeline(steps=[('scaler', StandardScaler())])

# Creating preprocessing steps
preprocessor = ColumnTransformer(transformers=[('preprocess', feature_transformer, feature_columns)])

# Creating training pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', LogisticRegression(solver='lbfgs', multi_class='auto'))])

# Fitting pipeline to train a linear regression model
model = pipeline.fit(feat_train, label_train)
print(model)

# Evaluating model
predictions = model.predict(feat_test)

accuracy = accuracy_score(label_test, predictions)
precision = precision_score(label_test, predictions, average='macro')
recall = recall_score(label_test, predictions, average='macro')

print('Overalls\nAccuracy: %f\nPrecision: %f\nRecall: %f' % (accuracy, precision, recall))

cm = confusion_matrix(label_test, predictions)
classes = ['Variety A', 'Variety B', 'Variety C']
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Variety')
plt.ylabel('Actual Variety')
plt.show()

# Get ROC metrics for each class

probabilities = model.predict_proba(feat_test)
auc = roc_auc_score(label_test, probabilities, multi_class='ovr')
print('Average AUC: ', auc)

fpr = {}
tpr = {}
thresh = {}

for i in range(len(classes)):
    fpr[i], tpr[i], thresh[i] = roc_curve(label_test, probabilities[:, i], pos_label=i)

colors = ['orange', 'green', 'blue']
for i in range(len(classes)):
    plt.plot(fpr[i], tpr[i], linestyle='--', color=colors[i], label=classes[i] + ' vs Rest')
    plt.title('Class ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc='best')
    plt.show()

# Saving model as a pickle file
filename = './wine_classifier.pkl'
joblib.dump(model, filename)

# Load saved model
model = joblib.load(filename)

# Get predictions for two new wine samples
'''
"When you're happy with your model's predictive performance, save it and then use it to predict classes for the following two new wine samples:

[13.72,1.43,2.5,16.7,108,3.4,3.67,0.19,2.04,6.8,0.89,2.87,1285] expected = variety A
[12.37,0.94,1.36,10.6,88,1.98,0.57,0.28,0.42,1.95,1.05,1.82,520] expected = variety B"
'''
feat_new = np.array([[13.72, 1.43, 2.5, 16.7, 108, 3.4, 3.67, 0.19, 2.04, 6.8, 0.89, 2.87, 1285],
                    [12.37, 0.94, 1.36, 10.6, 88, 1.98, 0.57, 0.28, 0.42, 1.95, 1.05, 1.82, 520]])

predictions = model.predict(feat_new)

for prediction in predictions:
    print(prediction, '(' + classes[prediction] + ')')
