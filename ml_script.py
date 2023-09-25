#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import pyreadr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
plt.rcParams['figure.figsize'] = [12, 8]


os.chdir("/rds/general/project/hda-22-23/live/TDS/ek2018") # set wd
data = pyreadr.read_r('../ek2018/ML/final_data2.rds') # import data
print(data.keys())
data = data[None]

#### SETTING UP DATA ####

# one hot encoding
one_hot_encoded_cols = pd.get_dummies(data[['smoking_status', 'alcohol_drinker_status', "employment_status", "qualifications", "chronotype"]])

# concatenate the one-hot encoded columns with the remaining columns in the original DataFrame
data_encoded = pd.concat([data.drop(['smoking_status', 'alcohol_drinker_status', "employment_status", "qualifications", "chronotype"], axis=1), one_hot_encoded_cols], axis=1)


# convert some cols to integers
ordinal_cols = ['number_medications', 'days_walking', "days_moderate_activity", "days_vigorous_activity", "time_tv","time_phone", "sleep_duration", "alcohol_frequency",
               "ethnicity", "time_outdoors_summer", "time_outdoors_winter", "time_computer", "avg_household_income", "urban_rural", "maternal_smoking_at_birth", "mother_illness",
               "father_illness"]
# convert columns to integers
for col in ordinal_cols:
    data_encoded[col] = data_encoded[col].astype(int)
    
# move the outcome to be the last column and make it an integer
last_column = data_encoded.pop('incident_case')  # remove column 'A' and store it in a variable
data_encoded['incident_case'] = last_column

data_encoded['incident_case'] = data_encoded['incident_case'].astype(int)


# train and test split (80:20)
X = data_encoded.drop(columns=['incident_case'])
Y = data_encoded['incident_case']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42)




##### Clustering ####
from sklearn.decomposition import PCA

# create PCA object
pca = PCA()
# fit PCA on data
pca.fit(clusterdata)

# get explained variance ratio
explained_var_ratio = pca.explained_variance_ratio_

# plot cumulative explained variance
cumulative_var_ratio = np.cumsum(explained_var_ratio)
plt.plot(cumulative_var_ratio)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('PCA Cumulative Explained Variance')
plt.show()

# find elbow point
diffs = np.diff(cumulative_var_ratio)
elbow = np.argmin(diffs) + 1

# use elbow point as number of components
print(f'Number of components at elbow point: {elbow}')


pca = PCA(n_components=43)
dementiaPCA = pca.fit_transform(clusterdata)


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2)
dementiaPCA_labels = kmeans.fit_predict(dementiaPCA)


dfdementiaPCA = pd.DataFrame(dementiaPCA)
dfdementiaPCA['cluster'] = dementiaPCA_labels


from sklearn.manifold import TSNE
X = dfdementiaPCA.iloc[:,:-1]
Xtsne = TSNE(n_components=2, perplexity=30).fit_transform(X)
dftsne = pd.DataFrame(Xtsne)
dftsne['cluster'] = dementiaPCA_labels
dftsne.columns = ['x1','x2','cluster']


sns.scatterplot(data=dftsne,x='x1',y='x2',hue='cluster',legend="full",alpha=0.5)


from sklearn.manifold import TSNE
X = dfdementiaPCA.iloc[:,:-1]
Xtsne = TSNE(n_components=2).fit_transform(X)
dftsne = pd.DataFrame(Xtsne)
dftsne['cluster'] = dementiaPCA_labels
dftsne.columns = ['x1','x2','cluster']


clusterdata['cluster'] = dementiaPCA_labels


clusterdata = pd.DataFrame(clusterdata) # have to do this for the code below to run


# Some functions to plot just the variables that has significant deviation from global mean
def outside_limit(df, label_col, label, sensitivity):
  feature_list = clusterdata.columns[:-1]
  
  plot_list = []
  mean_overall_list = []
  mean_cluster_list = []
  
  for i,varname in enumerate(feature_list):
    
    #     get overall mean for a variable, set lower and upper limit
    mean_overall = df[varname].mean()
    lower_limit = mean_overall - (mean_overall*sensitivity)
    upper_limit = mean_overall + (mean_overall*sensitivity)

    #     get cluster mean for a variable
    cluster_filter = df[label_col]==label
    pd_cluster = df[cluster_filter]
    mean_cluster = pd_cluster[varname].mean()
    
    #     create filter to display graph with 0.5 deviation from the mean
    if mean_cluster <= lower_limit or mean_cluster >= upper_limit:
      plot_list.append(varname)
      mean_overall_std = mean_overall/mean_overall
      mean_cluster_std = mean_cluster/mean_overall
      mean_overall_list.append(mean_overall_std)
      mean_cluster_list.append(mean_cluster_std)
   
  mean_df = pd.DataFrame({'feature_list':plot_list,
                         'mean_overall_list':mean_overall_list,
                         'mean_cluster_list':mean_cluster_list})
  mean_df = mean_df.sort_values(by=['mean_cluster_list'], ascending=False)
  
  return mean_df

def plot_barchart_all_unique_features(df, label_col, label, ax, sensitivity):
  
  mean_df = outside_limit(df, label_col, label, sensitivity)
  mean_df_to_plot = mean_df.drop(['mean_overall_list'], axis=1)
  
  if len(mean_df.index) != 0:
    sns.barplot(y='feature_list', x='mean_cluster_list', data=mean_df_to_plot, palette=sns.cubehelix_palette(20, start=.5, rot=-.75, reverse=True),                 alpha=0.75, dodge=True, ax=ax)

    for i,p in enumerate(ax.patches):
      ax.annotate("{:.02f}".format((p.get_width())), 
                  (1, p.get_y() + p.get_height() / 2.), xycoords=('axes fraction', 'data'),
                  ha='right', va='top', fontsize=10, color='black', rotation=0, 
                  xytext=(0, 0),
                  textcoords='offset pixels')
  
  ax.set_title('Unique Characteristics of Cluster ' + str(label))
  ax.set_xlabel('Standardized Mean')
  ax.axvline(x=1, color='k')

def plot_features_all_cluster(df, label_col, n_clusters, sensitivity):
  n_plot = n_clusters
  fig, ax = plt.subplots(n_plot, 1, figsize=(12, n_plot*6), sharex='col')
  ax= ax.ravel()
  
  label = np.arange(n_clusters)
  for i in label:
    plot_barchart_all_unique_features(df, label_col, label=i, ax=ax[i], sensitivity=sensitivity)
    ax[i].xaxis.set_tick_params(labelbottom=True, rotation=45)
    
  plt.tight_layout()
  display(fig)


plot_features_all_cluster(df=clusterdata, label_col='cluster', n_clusters=2, sensitivity=0.2)


# Get the centroid values for each cluster
centroids = pca.inverse_transform(kmeans.cluster_centers_)

# Convert centroid values back into original feature space
cluster_features = pd.DataFrame(
    data=centroids,
    columns=clusterdata.columns[:-1]
)

# Print the feature values for each cluster
for i, row in cluster_features.iterrows():
    print(f"Cluster {i}:")
    print(row.sort_values(ascending=False)[:5])
    print()

    


clusterdata.head


#### Random forest ####
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score


# Define the pipeline
pipeline = Pipeline([
    ('over', SMOTE(sampling_strategy=0.4)),
    ('under', RandomUnderSampler(sampling_strategy=0.6)),
    ('rf', RandomForestClassifier())
])


# Define the parameter grid for grid search
#param_grid = {
 #   'rf__n_estimators': [50, 100, 200],
  #  'rf__max_depth': [2, 5, 10, 15]}

param_grid = {
    'rf__n_estimators': [10, 50, 100],
    'rf__max_depth': [2, 5, 8],
    'rf__min_samples_split': [2, 5, 10],
    'rf__min_samples_leaf': [1, 2, 4]
}

# Define the scoring metric for the grid search
scoring = {'accuracy': 'accuracy', 'recall':'recall', 'F1':'f1'}

# Create the grid search object
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring=scoring,n_jobs=4, cv=5, return_train_score=True,refit='accuracy')


# Fit the grid search to the training data
grid_search.fit(X_train, y_train)


# Check the dimensions of the dataset before and after sampling
print(f"Original dataset shape: {X_train.shape}")
X_train_resampled, y_train_resampled = pipeline['over'].fit_resample(X_train, y_train)
X_train_resampled, y_train_resampled = pipeline['under'].fit_resample(X_train_resampled, y_train_resampled)
print('Resampled data shape:', X_train_resampled.shape)


# save the data so i can use it in R
X_train_resampled.to_csv('X_train_resampled.csv', index=False)
y_train_resampled.to_csv('y_train_resampled.csv', index=False)


print('\nBalance of positive and negative classes (%):')
y_train_resampled.value_counts(normalize=True) * 100


# Get the best estimator from the grid search
best_estimator = grid_search.best_estimator_
print(best_estimator)


print("Best parameters: ", grid_search.best_params_)


# Predict on the test data using the best estimator
y_pred = best_estimator.predict(X_test)

# METRICS
# Calculate the accuracy score on the test data
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)

# Accuracy on the test set
print("\033[1m Accuracy of Random forest on test set:","{:.2%}".format(accuracy_score(y_test, y_pred)))
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# we should use average accuracy not true accuracy
# Access the cv_results_ dictionary
cv_results = grid_search.cv_results_

# Get the mean test accuracy for each parameter combination
mean_test_accuracy = cv_results['mean_test_accuracy']

# Find the index of the best parameter combination
best_index = np.argmax(mean_test_accuracy)

# Get the best mean test accuracy
best_accuracy = mean_test_accuracy[best_index]

# Print the best accuracy
print("Best accuracy:", best_accuracy)


# balanced accuracy (for imbalanced test set)
from sklearn.metrics import balanced_accuracy_score
balanced_accuracy_score(y_test, y_pred)


# Calculate accuracy on the training data
y_train_pred = best_estimator.predict(X_train_resampled)
accuracy_train = accuracy_score(y_train_resampled, y_train_pred)
print("Train Accuracy:", accuracy_train)

print(classification_report(y_train_resampled, y_train_pred))


# Calculate the AUC score on the test data
y_prob = best_estimator.predict_proba(X_test)[:, 1]
auc_rf = roc_auc_score(y_test, y_prob)
print("AUC:", auc)

# confusion matrix
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve
# Print the confusion matrix for the testing set
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix, test:")
print(cm)

# Print the confusion matrix for the training set
cm_train = confusion_matrix(y_train_resampled, y_train_pred)
print("Confusion Matrix, train:")
print(cm_train)


# roc curve
fpr_rf, tpr_rf, thresholds = roc_curve(y_test, y_prob)
plt.plot(fpr_rf, tpr_rf, label='RF AUC = {:.2f}%'.format(auc*100))
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc='lower right')
plt.title("ROC Curve")
plt.show()


print(tpr_rf)
print(fpr_rf)

# confusion matrix for the test set
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
sns.heatmap(cm, annot=True, fmt='g',
                       xticklabels=['Control', 'Dementia case'],
            yticklabels=['Control', 'Dementia case'])
plt.xlabel('Predicted')
plt.ylabel('True')

import shap

# Create a SHAP explainer for the best_estimator
explainer = shap.TreeExplainer(best_estimator.named_steps['rf'])
shap_values = explainer.shap_values(X_test)

# Plot SHAP summary plots
plt.title('Random Forest SHAP plot for Dementia')
shap.summary_plot(shap_values, features=X_test, class_inds=[1])
plt.show()

plt.title('Random Forest SHAP Density plot for Dementia')
shap.summary_plot(shap_values[1], features=X_test)
plt.show()

# Get the trained Random Forest classifier from the pipeline
rf = grid_search.best_estimator_.named_steps['rf']

# Get the feature importances
importances = rf.feature_importances_
indices = importances.argsort()
features = X_train_resampled.columns[indices]

# Plot the feature importances as a horizontal bar chart
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), features)
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importances')
plt.show()



#### XGBoost  ####
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE 
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split

X2 = sampled_data.drop(columns=['incident_case'])
Y2 = sampled_data['incident_case']


X_train, X_test, y_train, y_test = train_test_split(
    X2, Y2, test_size=0.2, random_state=42
)


# Define the pipeline
pipeline = Pipeline([
    ('over', SMOTE(sampling_strategy=0.4)),
    ('under', RandomUnderSampler(sampling_strategy=0.6))
])

# Check the dimensions of the dataset before and after sampling
print(f"Original dataset shape: {X_train.shape}")
X_resampled, y_resampled = pipeline['over'].fit_resample(X_train, y_train)
X_resampled, y_resampled = pipeline['under'].fit_resample(X_resampled, y_resampled)
print(f"Resampled dataset shape: {X_resampled.shape}")
print(f"Resampled dataset shape, Y: {y_resampled.shape}")

#print('\nBalance of positive and negative classes (%):')
y_resampled.value_counts(normalize=True) * 100


# Define the pipeline
pipeline = Pipeline([
    ('over', SMOTE(sampling_strategy=0.4)),
    ('under', RandomUnderSampler(sampling_strategy=0.6)),
    ('xgb', XGBClassifier(seed=42))
])


# Define the parameter grid for grid search
param_grid = {
    'xgb__n_estimators': [50, 100, 200],
    'xgb__max_depth': [5, 10, 15],
    'xgb__min_child_weight': [1, 3, 5, 7],
    'xgb__gamma':[i/10.0 for i in range(0,5)]
}


# Define the scoring metric for the grid search
scoring = {'accuracy': 'accuracy', 'recall':'recall', 'F1':'f1'}

# Create the grid search object
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring=scoring,n_jobs=4, cv=5, return_train_score=True,refit='accuracy')

# Fit the grid search to the training data
grid_search.fit(X_train, y_train)


# Check the dimensions of the dataset before and after sampling
print(f"Original dataset shape: {X_train.shape}")
X_train_resampled, y_train_resampled = pipeline['over'].fit_resample(X_train, y_train)
X_train_resampled, y_train_resampled = pipeline['under'].fit_resample(X_train_resampled, y_train_resampled)
print('Resampled data shape:', X_train_resampled.shape)


# Get the best estimator from the grid search
best_estimator = grid_search.best_estimator_
print(best_estimator)

print("Best parameters: ", grid_search.best_params_)


# Predict on the test data using the best estimator
y_pred = best_estimator.predict(X_test)


# METRICS
# Accuracy on the test set
print("\033[1m Accuracy of XGBoost on test set:","{:.2%}".format(accuracy_score(y_test, y_pred)))
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# we should use average accuracy not true accuracy
# Access the cv_results_ dictionary
cv_results = grid_search.cv_results_

# Get the mean test accuracy for each parameter combination
mean_test_accuracy = cv_results['mean_test_accuracy']

# Find the index of the best parameter combination
best_index = np.argmax(mean_test_accuracy)

# Get the best mean test accuracy
best_accuracy = mean_test_accuracy[best_index]

# Print the best accuracy
print("Best accuracy:", best_accuracy)


# balanced accuracy (for imbalanced test set)
from sklearn.metrics import balanced_accuracy_score
balanced_accuracy_score(y_test, y_pred)


# Calculate metrics on the training data
y_train_pred = best_estimator.predict(X_train_resampled)
accuracy_train = accuracy_score(y_train_resampled, y_train_pred)
print("Train Accuracy:", accuracy_train)

print(classification_report(y_train_resampled, y_train_pred))


# Calculate the AUC score on the test data
y_prob = best_estimator.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)
print("AUC:", auc)


from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve
# roc curve
fpr_xgb, tpr_xgb, thresholds = roc_curve(y_test, y_prob)
plt.plot(fpr_xgb, tpr_xgb, label='XGBoost AUC = {:.2f}%'.format(auc*100))
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc='lower right')
plt.title("ROC Curve")
plt.show()


print(tpr_xgb)
print(fpr_xgb)

# confusion matrix for the test set
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
sns.heatmap(cm, annot=True, fmt='g',
                       xticklabels=['Control', 'Dementia case'],
            yticklabels=['Control', 'Dementia case'])
plt.xlabel('Predicted')
plt.ylabel('True')


import shap
# shap plot
xgb_model = grid_search.best_estimator_['xgb']
explainer = shap.Explainer(xgb_model)
shap_values = explainer.shap_values(X_test)



# Plot the shap summary plot
plt.title('XGBoost SHAP plot for Dementia')
shap.summary_plot(shap_values, X_test, plot_type='bar', class_names=['No Dementia', 'Dementia'])
plt.show()

#plt.title('XGBoost SHAP Density plot for Dementia')
#shap.summary_plot(shap_values[1], features = X_test)
#plt.show()

# Plot the shap dependence plots for each feature
#for i in range(X_test.shape[1]):
 #   plt.title('XGBoost SHAP dependence plot for feature {}'.format(i))
  #  shap.dependence_plot(i, shap_values, X_test)
   # plt.show()

shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test, plot_type = "violin")


#### SVM #####

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Define the pipeline
pipeline = Pipeline([
    ('over', SMOTE(sampling_strategy=0.4)),
    ('under', RandomUnderSampler(sampling_strategy=0.6)),
    ('standardscaler', StandardScaler()),
    ('svm', SVC())
])

# Define the parameter grid for grid search
# Define the parameter grid for grid search
param_grid = {
    'svm__C': [0.1, 1, 10],
    'svm__kernel': ['linear', 'rbf', 'sigmoid', 'poly'],
    'svm__gamma': [1,0.1,0.01,0.001]
}

# Define the scoring metric for the grid search
scoring = {'accuracy': 'accuracy', 'recall':'recall', 'F1':'f1'}

# Create the grid search object
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring=scoring,n_jobs=4, cv=5, return_train_score=True,refit='accuracy')


# Fit the grid search to the training data
grid_search.fit(X_train, y_train)


# Check the dimensions of the dataset before and after sampling
print(f"Original dataset shape: {X_train.shape}")
X_train_resampled, y_train_resampled = pipeline['over'].fit_resample(X_train, y_train)
X_train_resampled, y_train_resampled = pipeline['under'].fit_resample(X_train_resampled, y_train_resampled)
X_train_resampled = pipeline['standardscaler'].fit_transform(X_train_resampled)

print('Resampled data shape:', X_train_resampled.shape)


# Get the best estimator from the grid search
best_estimator = grid_search.best_estimator_
print(best_estimator)


print("Best parameters: ", grid_search.best_params_)


# Predict on the test data using the best estimator
y_pred = best_estimator.predict(X_test)


# METRICS
# Calculate the accuracy score on the test data
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)

# Accuracy on the test set
print("\033[1m Accuracy of Random forest on test set:","{:.2%}".format(accuracy_score(y_test, y_pred)))
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# we should use average accuracy not true accuracy
# Access the cv_results_ dictionary
cv_results = grid_search.cv_results_

# Get the mean test accuracy for each parameter combination
mean_test_accuracy = cv_results['mean_test_accuracy']

# Find the index of the best parameter combination
best_index = np.argmax(mean_test_accuracy)

# Get the best mean test accuracy
best_accuracy = mean_test_accuracy[best_index]

# Print the best accuracy
print("Best accuracy:", best_accuracy)


# Calculate accuracy on the training data
y_train_pred = best_estimator.predict(X_train_resampled)
accuracy_train = accuracy_score(y_train_resampled, y_train_pred)
print("Train Accuracy:", accuracy_train)

print(classification_report(y_train_resampled, y_train_pred))


# Calculate the AUC score on the test data
y_prob = best_estimator.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)
print("AUC:", auc)


# Calculate the predicted scores on the test data
y_scores = best_estimator.decision_function(X_test)

# Calculate the AUC score on the test data
auc = roc_auc_score(y_test, y_scores)
print("AUC:", auc)


from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

# Calculate the false positive rate and true positive rate
fpr_svm, tpr_svm, thresholds= roc_curve(y_test, y_scores)

# Plot the ROC curve
plt.plot(fpr_svm, tpr_svm, lw=2, label='SVM AUC = {:.2f}%'.format(auc*100))
plt.plot([0, 1], [0, 1],lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

# confusion matrix for the test set
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
sns.heatmap(cm, annot=True, fmt='g',
                       xticklabels=['Control', 'Dementia case'],
            yticklabels=['Control', 'Dementia case'])
plt.xlabel('Predicted')
plt.ylabel('True')


# shap plot
svm_model = grid_search.best_estimator_['svm']
explainer = shap.Explainer(svm_model, X_train_resampled)
shap_values = explainer.shap_values(X_test)

plt.title('SVM SHAP plot for Dementia')
shap.summary_plot(shap_values, features = X_test, class_inds = [1])
plt.show()

plt.title('SVM SHAP Density plot for Dementia')
shap.summary_plot(shap_values[1], features = X_test)
plt.show()



# Generate the summary plot
shap.summary_plot(shap_values, X_test, plot_type="bar", feature_names=X_train.columns)


#### Plot all AUCs together ####

import matplotlib.pyplot as plt

# Plot SVM ROC curve
plt.plot(fpr_svm, tpr_svm, label='SVM')

# Plot XGBoost ROC curve
plt.plot(fpr_xgb, tpr_xgb, label='XGBoost', linewidth=5)

# Plot Random Forest ROC curve
plt.plot(fpr_rf, tpr_rf, label='Random Forest')

# Set axis labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

# Add legend
plt.legend()
plt.plot([0, 1], [0, 1],lw=2, linestyle='--')
# Show plot
plt.show()




