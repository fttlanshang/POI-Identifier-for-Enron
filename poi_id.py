#!/usr/bin/python
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from tester import test_classifier

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
financial_features = [
	'salary', 'deferral_payments', 'total_payments', 'loan_advances',
	'bonus', 'restricted_stock_deferred', 'deferred_income', 
	'total_stock_value', 'expenses', 'exercised_stock_options', 
	'other', 'long_term_incentive', 'restricted_stock', 
	'director_fees'
]
## Since I added three new email features(which would be discussed below)
# and they overlapped the previous email features. So I ran a default 
# GaussianNB classfier to determine which email features set to use. 
##=====================result for original email features===================
# - features: All financial features and email features
# - model: GaussianNB
# - Precision: 0.22604      Recall: 0.39500
##=====================result for new added email features===================
# - features: All financial features and three new email features
# - model: GaussianNB
# - Precision: 0.25385      Recall: 0.42850
# We can see that using three relative email features preformed better.
# So I used all financial features and three relative email features as input features.

# email_features = [
# 	'to_messages', 'from_poi_to_this_person', 'from_messages', 
# 	'from_this_person_to_poi', 'shared_receipt_with_poi'
# ]
email_features = [
	"from_this_person_to_poi_ratio", 
	"from_poi_to_this_person_ratio", "shared_ratio"
]

features_list = ['poi'] + financial_features + email_features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# turn dict into pandas dataframe
df = pd.DataFrame.from_records(list(data_dict.values()))
employees = pd.Series(list(data_dict.keys()))
# set the index of df to be the employees series:
df.set_index(employees, inplace=True)
df.replace('NaN', np.nan, inplace = True)

### Task 2: Remove outliers
# It was easy to find the "TOTAL" outlier when drawing a scatterplot.
def draw_scatterplot(variable1, variable2):
    feature1_poi = [df[variable1][ii] for ii in range(0, len(df[variable1])) if df["poi"][ii] == True]
    feature1_non_poi = [df[variable1][ii] for ii in range(0, len(df[variable1])) if df["poi"][ii] == False]
    feature2_poi = [df[variable2][ii] for ii in range(0, len(df[variable2])) if df["poi"][ii] == True]
    feature2_non_poi = [df[variable2][ii] for ii in range(0, len(df[variable2])) if df["poi"][ii] == False]
    plt.scatter(x = feature1_poi, y = feature2_poi, color="r", label="poi")
    plt.scatter(x = feature1_non_poi, y = feature2_non_poi, color = "b", label = "non_poi")
    plt.legend()
    plt.xlabel(variable1)
    plt.ylabel(variable2)
    plt.show()

def record_with_many_nan():
	for index in df.index:
	    record = df.loc[index, :]
	    count = 0
	    for variable in record.index:
	        if (type(record[variable]) == float or \
	        		type(record[variable]) == np.float64) and \
	        	np.isnan(record[variable]):
	            count += 1
	    if count >= 18:
	        print index, " : ", record

# draw_scatterplot("salary", "bonus")
# df[df["bonus"] > 0.8e8] #TOTAL
# TOTAL has pretty large bonus, so it's easy to recognize
df = df[df.index != "TOTAL"]
# But when I looked through the forum, a post showed another outlier! 
# reference: https://discussions.udacity.com/t/looking-for-assistance-on-the-final-project/240282
df = df[df.index != "THE TRAVEL AGENCY IN THE PARK"]
# record_with_many_nan()

#Through the hint provided by the reviewer, I found one record contained all NaN values
# except POI, "LOCKHART EUGENE E". It was also an outlier, since it could not 
# provide any information.
df = df[df.index != "LOCKHART EUGENE E"]

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

## turn email features into ratio
df["from_this_person_to_poi_ratio"] = df["from_this_person_to_poi"] / df["from_messages"]
df["from_poi_to_this_person_ratio"] = df["from_poi_to_this_person"] / df["to_messages"]
df["shared_ratio"] = df["shared_receipt_with_poi"] / (df["from_messages"] + df["to_messages"])

#turn dataframe back to dict
df.replace(np.nan, 'NaN', inplace = True)
data_dict = df.to_dict('index')
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2, f_classif,mutual_info_classif
from sklearn.metrics import recall_score, precision_score
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# I integrated selectKBest/PCA/feature scaling and classifier into a pipeline.
# And different algorithms have different pipelines.

N_FEATURES_OPTIONS = [2, 3, 5, 8, 10, 12, 15]

##==========================Naive Bayes==========================
# Actually, I think naive bayes don't need feature scaling. 
# But I think PCA need feature scaling, but it did't perform well when doing
# feature scaling before PCA. Why?
##---------------option 1: using PCA to reduce dimensionality----------
# pipe = Pipeline([
# 	("scale", MinMaxScaler()),
# 	("reduce_dim", PCA(random_state=42)),
# 	("classify", GaussianNB())
# ])
# param_grid = [
# 	{
# 		"reduce_dim__n_components":  N_FEATURES_OPTIONS,
# 	}
# ]
#without scale: 
 # {'reduce_dim__n_components': 8} - Precision: 0.42568      Recall: 0.34650

#with scale:
#{'reduce_dim__n_components': 15} - Precision: 0.32231      Recall: 0.34600

## -----------option 2: using selectKBest--(final algorithm used)---------------------
# pipe = Pipeline([
# 	("reduce_dim", SelectKBest(f_classif)),
# 	("classify", GaussianNB())
# ])
# param_grid = [
# 	{
# 		"reduce_dim__k":  N_FEATURES_OPTIONS,
# 	}
# ]
# {'reduce_dim__k': 5} 
# Precision: 0.41646      Recall: 0.33900


##==========================Decision Tree==========================
MIN_SAMPLES_SPLIT_OPTIONS = [2, 4, 6, 8, 10, 12, 15, 20, 30]
MAX_DEPTH_OPTIONS = [2, 3, 4, 5, 6, 7, 8]
# ----option 1: PCA---------------- 
pipe3 = Pipeline([
	("scale", MinMaxScaler()),
	("pca", PCA(random_state = 42)),
	("classify", tree.DecisionTreeClassifier())
])
param_grid3 = [
	{
		"pca__n_components": N_FEATURES_OPTIONS,
		"classify__min_samples_split": MIN_SAMPLES_SPLIT_OPTIONS,
		"classify__max_depth": MAX_DEPTH_OPTIONS
	}
]
# {'classify__min_samples_split': 15, 'pca__n_components': 2, 'classify__max_depth': 3}
# Precision: 0.45178      Recall: 0.26000


# ----option 2: selectKBest----------------
pipe4 = Pipeline([
	("reduce_dim", SelectKBest(f_classif)),
	("classify", tree.DecisionTreeClassifier())
])
param_grid4 = [
	{
		"reduce_dim__k":  N_FEATURES_OPTIONS,
		"classify__min_samples_split": MIN_SAMPLES_SPLIT_OPTIONS,
		"classify__max_depth": MAX_DEPTH_OPTIONS
	}
]
# {'classify__min_samples_split': 2, 'reduce_dim__k': 2, 'classify__max_depth': 3}
# Precision: 0.37475      Recall: 0.18400


##==========================K Nearest Neighbors==========================
N_NEIGHBORS_OPTIONS = [3, 5, 6, 7, 8, 10, 12, 15]
# ----------option 1: PCA--------------------
pipe5 = Pipeline([
	("scale", MinMaxScaler()),
	("pca", PCA(random_state = 42)),
	("classify", KNeighborsClassifier())
])
param_grid5 = [
	{
		"pca__n_components": N_FEATURES_OPTIONS,
		"classify__n_neighbors": N_NEIGHBORS_OPTIONS,
	}
]
# {'pca__n_components': 2, 'classify__n_neighbors': 3}
# Precision: 0.37860      Recall: 0.23000

# -------------option 2: selectKBest---------------
pipe6 = Pipeline([
	("scale", MinMaxScaler()),
	("reduce_dim", SelectKBest(f_classif)),
	("classify", KNeighborsClassifier())
])
param_grid6 = [
	{
		"reduce_dim__k":  N_FEATURES_OPTIONS,
		"classify__n_neighbors": N_NEIGHBORS_OPTIONS,
	}
]
# {'reduce_dim__k': 2, 'classify__n_neighbors': 5}
# Precision: 0.51042      Recall: 0.14700


##==========================SVC==========================
C_OPTIONS = [0.1, 1, 10, 100, 1000, 2500, 5000, 7500, 10000, 1e5]
kernels = ["rbf"]
GAMMA_OPTIONS = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 0.2, 0.5, 0.75, 1]

# -----------option 1: PCA --------------------------------
pipe7 = Pipeline([
	("scale", MinMaxScaler()),
	("pca", PCA()),
	("classify", SVC())
])
param_grid7 = [
	{
		"pca__n_components": N_FEATURES_OPTIONS,
		"classify__C": C_OPTIONS,
		"classify__kernel": kernels,
		"classify__gamma": GAMMA_OPTIONS
	}
]
# {'classify__C': 10000, 'pca__n_components': 2, 'classify__gamma': 0.75, 'classify__kernel': 'rbf'}
 # Precision: 0.22973      Recall: 0.08500


#----------------option 2: selectKBest-----------------
pipe8 = Pipeline([
	("scale", MinMaxScaler()),
	("reduce_dim", SelectKBest(f_classif)),
	("classify", SVC())
])
param_grid8 = [
	{
		"reduce_dim__k": N_FEATURES_OPTIONS,
		"classify__C": C_OPTIONS,
		"classify__kernel": kernels,
		"classify__gamma": GAMMA_OPTIONS
	}
]
# {'classify__C': 2500, 'reduce_dim__k': 15, 'classify__gamma': 0.005, 'classify__kernel': 'rbf'}
# Precision: 0.56355      Recall: 0.16850

#actually, SelectKBest(chi2) raise error:
# https://stackoverflow.com/questions/25792012/feature-selection-using-scikit-learn
pipes = [pipe3, pipe4, pipe5, pipe6, pipe7, pipe8]
param_grids = [param_grid3, param_grid4, param_grid5, param_grid6, param_grid7, param_grid8 ]
for i in range(len(pipes)):
	pipe = pipes[i]
	param_grid = param_grids[i]

	cv = StratifiedShuffleSplit(n_splits = 100, random_state = 42)
	grid = GridSearchCV(pipe, cv=cv, param_grid=param_grid, scoring='f1')

	grid.fit(features, labels)

	print grid.best_params_
	test_classifier(grid.best_estimator_, my_dataset, features_list)

	clf = grid.best_estimator_

	try:
		features_final = clf.named_steps['reduce_dim'].get_support()
		feature_scores_final = clf.named_steps['reduce_dim'].scores_
		for i in range(len(financial_features + email_features)):
			feature = (financial_features + email_features)[i]
			score = round(feature_scores_final[i],2)
			print feature, " | ", features_final[i], ' | ', score, " |"
	except:
		print "PCA"

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
dump_classifier_and_data(clf, my_dataset, features_list)