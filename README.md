# Identify POI from Enron Dataset

### Questions and Answers
> Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]

In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, a significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for top executives. The goal of this project is to identify persons of interest in the famous Enron fraud case using the email and financial dataset given. We could use machine learning algorithms, in this case, supervised learning algorithms, to classify people into binary groups, poi or non-poi.  

There were two obvious outliers in the data. The first one was "TOTAL" and I found it through scatterplot. The second one was called "THE TRAVEL AGENCY IN THE PARK". I didn't notice it at first and got it from [a post](https://discussions.udacity.com/t/looking-for-assistance-on-the-final-project/240282). It couldn't be a person's name and there were only one valid value for expense. I removed these two outliers in the dataset, since they were not information describing a person.


> What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “intelligently select features”, “properly scale features”]

I added three new features into the dataset, `"from_this_person_to_poi_ratio"`/`"from_poi_to_this_person_ratio"` and `"shared_ratio"`. It was introduced in the previous mini-projects and I thought they were good features. We were interested to distinguish poi and non-poi. The exact numbers of email received or sent weren't as important as ratio.

```python
df["from_this_person_to_poi_ratio"] = df["from_this_person_to_poi"] / df["from_messages"]
df["from_poi_to_this_person_ratio"] = df["from_poi_to_this_person"] / df["to_messages"]
df["shared_ratio"] = df["shared_receipt_with_poi"] / (df["from_messages"] + df["to_messages"])
```

At first, I wanted to use scatterplot to select features manually. But the scatterplot could only show two features at one time and it was pretty hard to determine through the scatterplots. So later, I used `selectKBest` and `PCA` function to reduce dimensionality. I integrated them into pipeline. Since both can reduce dimensionality, so I preferred not to use them in the same pipeline.

Since K Nearest Neighbors, SVC and PCA are sensitive to feature scaling, so I performed feature scaling when using above algorithms or preprocessing. And Naive Bayes and Decision Trees are invariant to feature scaling, so I didn't do feature scaling unless I used PCA ahead for dimeansion reduce.

I ended up using 5 features in my POI identifier: `total_stock_value`, `exercised_stock_options`, `salary`,`bonus` and `from_this_person_to_poi_ratio`. The table below summarised all features and their relative scores. 

Features      					| used or not 	| score | 
------------------- 			| -------- 		|-----  |
total_stock_value  				|  True  		|  14.69|
exercised_stock_options  		|  True  		|  13.71|
salary  						|  True  		|  11.2 |
bonus  							|  True 		|  11.13|
from_this_person_to_poi_ratio  	|  True  		|  8.24 |
shared_ratio 					|  False  		|  6.6  |
restricted_stock  				|  False  		|  6.58 |
expenses  						|  False  		|  5.91 |
deferred_income  				|  False  		|  5.3  |
total_payments  				|  False  		|  2.77 |
long_term_incentive  			|  False  		|  2.61 | 
from_poi_to_this_person_ratio  	|  False  		|  2.35 |
director_fees  					|  False  		|  1.8  |
restricted_stock_deferred  		|  False 		|  0.76 |
deferral_payments  				|  False  		|  0.26 |
loan_advances  					|  False  		|  0.2  |
other  							|  False  		|  0.01 |


> What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]

I ended up using GaussianNB algorithm. I also tried SVC, K Nearest Neighbors and Decision Trees. The table below summarized their performance on precision and recall. I used the parameters found by pipeline and GridSearchCV for each algorithm to evaluate their performance. We could see that GaussianNB performed better than any other algorithms, achieving the goal that the recall and precision score should both be larger than 0.3. Roughly speaking, the other three algorithms had good score in precision, but low score in recall. 

Algorithm      		| method to reduce dimensionality | precision | recall  |
------------------- | ------------------------------- | --------- | --------|
GaussianNB     		| PCA 							  | 0.32214   | 0.31650 |
GaussianNB(final)   | selectKBest 					  | 0.41646   | 0.33900 |
Decision Trees 		| PCA 							  | 0.39211   | 0.24350 |
Decision Trees 		| selectKBest 					  | 0.37475   | 0.18400 |
K Nearest Neighbors | PCA 							  | 0.37860   | 0.23000 |
K Nearest Neighbors | selectKBest 					  | 0.51042   | 0.14700 |
SVC 				| PCA 							  | 0.22973   | 0.08500 |
SVC 				| selectKBest 					  | 0.56355   | 0.16850 |

> What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? What parameters did you tune? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric items: “discuss parameter tuning”, “tune the algorithm”]

Parameters would cause huge difference in performance. We tuned the parameters of an algorithms to achieve better performance and sometimes to avoid overfitting. If we didn't do this well, then we could probably get worse performance than the default one.

I used `GridSearchCV` during the parameters tuning period. Actually, the final algorithm that I perferred to use didn't have parameters that I need to tune. But in my whole exploration process, I tuned parameters for other three algorithms. Below is a summary of the parameters I tuned. 

- For `selectKBest`: `k`
- For `PCA`: `n_components`
- For `SVC`: `C` and `gamma`
- For `DecisionTreeClassifier`: `max_depth` and `min_samples_split`
- For `KNeighborsClassifier`: `k`

> What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric items: “discuss validation”, “validation strategy”]

In my opinion, validation is separating testing data and training data. Only use training data for model training, and use test data to estimate the model performance and to check whether overfitting exist in the model. To maximize the use of our data, we could do K-Fold cross validation and others. And in this case, `StratifiedShuffleSplit` were used. And I think using all features and labels to train the model would probably be a classic mistake, in this way, you would get a rather high score because your model had already seen the data.

During `GridSearchCV`, I used `StratifiedShuffleSplit` as the cross validation method. And the best estimator produced by `GridSearchCV` were estimated using `tester.py`.

> Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]

I used `tester.py` to evaluate the average performance for my final algorithm.
- Precision: 0.41646 (final)
- Recall: 0.33900 (final)

For the precision, suppose I labeled 1000 POIs at last, and 416 would truly belong to the POI class, while the other were not POIs.

For the recall, suppose there were 1000 POIs actually, the algorithm that I used would correctly classify 339 POIs.


### References
* [the idea on how to do this project](https://discussions.udacity.com/t/project-fear-strugging-with-machine-learning-project/198529/6)
* [sklearn feature selection](http://scikit-learn.org/stable/modules/feature_selection.html)
* https://stats.stackexchange.com/questions/27750/feature-selection-and-cross-validation、
* [how to turn dict into pandas dataframe](https://discussions.udacity.com/t/pickling-pandas-df/174753/2)
* [pipeline & gridSearchCV work together](http://nbviewer.jupyter.org/gist/swwelch/64a71c4e67f829728e27/GridSearchCV%20and%20Pipelines.ipynb)
* [a post inspiration on outlier](https://discussions.udacity.com/t/looking-for-assistance-on-the-final-project/240282)
* [about feature scaling](http://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html)

