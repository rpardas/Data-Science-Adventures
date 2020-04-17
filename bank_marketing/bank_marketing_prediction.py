# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# #### March 2020
# %% [markdown]
# We will use a marketing/banking dataset obtained from the UCI Machine Learning repository - https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
# %% [markdown]
# The dataset is related to phone call marketing campaigns of Portugese banking institutions.
#
# The goal is to find the most accurate model that predicts whether the client will subsribe to a term deposit or not.
#
# The target variable, y is a yes/no.
#
# We will use sklearn for
# - pre-processing,
# - splitting data for train/test
# - comparing 4 models (DummyClassifier, LogisticRegression, DecisionTreeClassifier, RandomForestClassifier)
# - comparing metrics (confusion matrix, accuracy, recall, f1, precision, auc)

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_curve,
    roc_auc_score,
    auc,
)
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

get_ipython().run_line_magic("matplotlib", "inline")

# %% [markdown]
# ## Data Collection
# %% [markdown]
# **The attribute information from UCI**<br>
#
# **Input variables:**<br>
# - **bank client data:**<br>
# 1 - age (numeric)<br>
# 2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')<br>
# 3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)<br>
# 4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')<br>
# 5 - default: has credit in default? (categorical: 'no','yes','unknown')<br>
# 6 - housing: has housing loan? (categorical: 'no','yes','unknown')<br>
# 7 - loan: has personal loan? (categorical: 'no','yes','unknown')<br>
# - **related with the last contact of the current campaign:**<br>
# 8 - contact: contact communication type (categorical: 'cellular','telephone')<br>
# 9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')<br>
# 10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')<br>
# 11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.<br>
# - **other attributes:**<br>
# 12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)<br>
# 13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)<br>
# 14 - previous: number of contacts performed before this campaign and for this client (numeric)<br>
# 15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')<br>
# - **social and economic context attributes:**<br>
# 16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)<br>
# 17 - cons.price.idx: consumer price index - monthly indicator (numeric)<br>
# 18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)<br>
# 19 - euribor3m: euribor 3 month rate - daily indicator (numeric)<br>
# 20 - nr.employed: number of employees - quarterly indicator (numeric)<br>
#
# **Output variable (desired target):**<br>
# 21 - y - has the client subscribed a term deposit? (binary: 'yes','no')
# %% [markdown]
# **Generate a Pandas dataframe from the .csv file.**<br>
# It uses a semicolon delimiter.

# %%
bank_df = pd.read_csv("bank-additional-full.csv", delimiter=";")

# %% [markdown]
# ## Preliminary Data Exploration

# %%
bank_df.head(5)


# %%
bank_df.shape

# %% [markdown]
# 41888 rows and 21 columns
# %% [markdown]
# **View the dtypes.**<br>
# A mix of object, int, and float.

# %%
bank_df.dtypes

# %% [markdown]
# Glance at some initial stats and descriptions

# %%
bank_df.describe()

# %% [markdown]
# **Number of unique values in each column**

# %%
for c in bank_df.columns:
    print(c, bank_df[c].nunique())

# %% [markdown]
# Peek at unique values of some questionable columns

# %%
for u in ["job", "previous", "emp.var.rate"]:
    print(u + ": ", bank_df[u].unique())

# %% [markdown]
# 2 types of contact<br>
# landline or cell

# %%
bank_df["contact"].unique()

# %% [markdown]
# 'default' is yes/no/unknown

# %%
bank_df["default"].unique()

# %% [markdown]
# 'pdays' is the number of days that passed by after the client was last contacted from a previous campaign -
# 999 means the client has not been contacted

# %%
bank_df["pdays"].unique()

# %% [markdown]
# There don't appear to be any null values.

# %%
bank_df.isnull().sum(axis=0)

# %% [markdown]
# 12 duplicated values

# %%
bank_df.duplicated().sum()


# %%
# view the duplicates
bank_df[bank_df.duplicated()]

# %% [markdown]
# **Drop the duplicates since there are only 12 out of the 41k**

# %%
bank_df.drop_duplicates(inplace=True)

# %% [markdown]
# Turn the 3 'basic' education types into one.

# %%
bank_df.replace(["basic.6y", "basic.4y", "basic.9y"], "basic", inplace=True)

# %% [markdown]
# Create a 'contacted' column from pdays to a 0/1 (no/yes)

# %%
bank_df["contacted"] = [0 if x == 999 else 1 for x in bank_df["pdays"]]

# %% [markdown]
# Create numeric columns out of 'y', 'day_of_week', and 'month'.

# %%
bank_df["y_num"] = bank_df[["y"]].replace({"no": 0, "yes": 1})
bank_df["dow_num"] = bank_df[["day_of_week"]].replace(
    {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4}
)
bank_df["month_num"] = bank_df[["month"]].replace(
    {
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }
)

# %% [markdown]
# Drop the original ones

# %%
bank_df.drop(["y", "day_of_week", "month", "pdays"], axis=1, inplace=True)

# %% [markdown]
# **The target variable is 'y_num'.**<br>
# Has the client subscribed to a term deposit or not?<br>
# Create a dataframe of the yes (1) records.

# %%
yes_df = bank_df[bank_df["y_num"] == 1]

# %% [markdown]
# Percentage of clients who have subscribed = 11%<br>
# Imbalanced dataset

# %%
((yes_df.shape[0]) / (bank_df.shape[0])) * 100

# %% [markdown]
# **Descriptive and exploratory analytics**
# %% [markdown]
# Clients with university degrees, admin jobs, and non-existent previous campaign have the highest proportion of clients subscribing to a term deposit

# %%
# university degree highest
yes_df["education"].value_counts().plot(kind="barh")
plt.show()

# %% [markdown]
# Clients with admin jobs

# %%
# admin highest
yes_df["job"].value_counts().plot(kind="barh")
plt.show()

# %% [markdown]
# Outcome of the previous marketing campaign - Nonexistent is the highest

# %%
yes_df["poutcome"].value_counts().plot(kind="barh")
plt.show()

# %% [markdown]
# **Category columns**

# %%
cat_cols = list(bank_df.columns[bank_df.dtypes == object])
cat_cols

# %% [markdown]
# **Numeric columns**

# %%
num_cols = list(bank_df.columns[bank_df.dtypes != object])
num_cols

# %% [markdown]
# Look at correlation between variables
# %% [markdown]
# Some multicollinearity from looking at the corralation matrix

# %%
plt.figure(figsize=(12, 10))
cor = bank_df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

# %% [markdown]
# A note from the UCI data source site: The 'duration' attribute highly affects the target variable. It should be removed if we would like to have a realistic model.
# %% [markdown]
# 'contacted' (whether they were contacted previously or not) seems to have the next highest correlation with 'y_num' (0.32)

# %%
bank_df.drop(["duration"], axis=1, inplace=True)

# %% [markdown]
# ## Data Preparation
# %% [markdown]
# ##### Note: decided to go with LabelEncoder but also explored pd.get_dummies

# %%
# bank_df = (pd.get_dummies(bank_df[cat_cols])).merge(bank_df[num_cols], left_index=True, right_index=True)
# bank_df.head(10)

# %% [markdown]
# **Use LabelEncoder for the categorical columns to get 0/1 values**

# %%
le = preprocessing.LabelEncoder()
for c in cat_cols:
    bank_df[c] = le.fit_transform(bank_df[c])

# %% [markdown]
# **Independent and dependent variables in to X and y**

# %%
X = bank_df.loc[:, bank_df.columns != "y_num"]
y = bank_df.loc[:, "y_num"]

# %% [markdown]
# **Split for train and test using 80/20 ratio**
# %% [markdown]
# using train_test_split

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# %%
print("Dimensions of X_train: ", X_train.shape)
print("Dimensions of X_test: ", X_test.shape)
print("Dimensions of y_train: ", y_train.shape)
print("Dimensions of y_test: ", y_test.shape)

# %% [markdown]
# **We will compare with scaled data using sklearn's preprocessing.StandardScaler**<br>
# See if accuracy improves

# %%
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scale = scaler.transform(X_train)
X_test_scale = scaler.transform(X_test)

# %% [markdown]
# Function to print the accuracy, recall, precision, and F1 scores

# %%
def print_scores(pred, type_string):

    accuracy = accuracy_score(y_test, pred)
    recall = recall_score(y_test, pred)
    precision = precision_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    print(type_string + " Scores:")
    print("   Accuracy: ", accuracy)
    print("   Recall: ", recall)
    print("   Precision: ", precision)
    print("   F1 score: ", f1)


# %% [markdown]
# Function to plot the False and True Positive rates and show AUC scores

# %%
def auc_plot(pred, type_string):

    fpr, tpr, _ = roc_curve(y_test, pred)
    auc_score = auc(fpr, tpr)

    fig = plt.figure(figsize=(8, 5))

    plt.plot(fpr, tpr, label="AUC score is : " + str(auc_score))
    plt.xlabel("F.P.R.", fontsize=10)
    plt.ylabel("T.P.R", fontsize=10)
    plt.xlim([-0.01, 1])
    plt.ylim([0, 1.01])
    plt.legend()

    plt.plot([0, 1], [0, 1], "r--")
    plt.show()

    print("AUC Score for " + type_string + " is", roc_auc_score(y_test, pred))


# %% [markdown]
# ## Compare ML Models
# %% [markdown]
# **We will compare the results of scaled and un-scaled data for<br>
# Dummy Classifer, Logistic Regression, Decision Tree, and Random Forest**
# %% [markdown]
# We will look at confusion matrices of the different predictions
# %% [markdown]
# |                   |Predicted Negative|Predicted Postive|
# |-------------------|------------------|-----------------|
# |**Actual Negative**|True Negative     |False Positive   |
# |**Actual Positive**|False Negative    |True Positive    |
# %% [markdown]
# ### Dummy Classifier
# %% [markdown]
# **Fit and predict**

# %%
dummy = DummyClassifier()

## fit on the training data
dummy.fit(X_train, y_train)

## make predictions on test data
dummy_test_pred = dummy.predict(X_test)

## fit on the scaled training data
dummy.fit(X_train_scale, y_train)

## make predictions on scaled test data
dummy_test_pred_scale = dummy.predict(X_test_scale)

# %% [markdown]
# **Confusion matrix**

# %%
dummy_matrix = confusion_matrix(y_test, dummy_test_pred)
print("Dummy, not scaled: " + "\n\n" + str(dummy_matrix) + "\n")

dummy_matrix_scale = confusion_matrix(y_test, dummy_test_pred_scale)
print("Dummy, scaled: " + "\n\n" + str(dummy_matrix_scale))

# %% [markdown]
# The Overall Error Rate is fairly high here for both scaled and un-scaled
# %% [markdown]
# **Scores**

# %%
print_scores(dummy_test_pred, "Dummy Classifier")


# %%
print_scores(dummy_test_pred_scale, "Scaled Dummy Classifier")

# %% [markdown]
# AUC ranges in value from 0 to 1. A model whose predictions are 100% wrong has an AUC of 0.0; one whose predictions are 100% correct has an AUC of 1.0.

# %%
auc_plot(dummy_test_pred, "Dummy Classifier")


# %%
auc_plot(dummy_test_pred_scale, "Scaled Dummy Classifier")

# %% [markdown]
# AUC scores are low for Dummy Classifier and scaling did not improve the results
# %% [markdown]
# ### Logistic Regression
# %% [markdown]
# **Fit and predict**

# %%
## max_iter = 1000
logr = LogisticRegression(random_state=0, max_iter=1000)
logr.fit(X_train, y_train)
logr_test_pred = logr.predict(X_test)

## scale
logr.fit(X_train_scale, y_train)
logr_test_pred_scale = logr.predict(X_test_scale)

# %% [markdown]
# **Confusion matrix**

# %%
logr_matrix = confusion_matrix(y_test, logr_test_pred)
print("LGR, not scaled: " + "\n\n" + str(logr_matrix) + "\n")


# %%
logr_matrix_scale = confusion_matrix(y_test, logr_test_pred_scale)
print("LGR, scaled: " + "\n\n" + str(logr_matrix_scale) + "\n")

# %% [markdown]
# **Scores**

# %%
print_scores(logr_test_pred, "Logistic Regression")


# %%
print_scores(logr_test_pred_scale, "Scaled Logistic Regression")

# %% [markdown]
# Accuracy is higher but scores did not improve with scaling

# %%
auc_plot(logr_test_pred, "Logistic Regression")


# %%
auc_plot(logr_test_pred_scale, "Scaled Logistic Regression")

# %% [markdown]
# Although AUC is slightly higher
# %% [markdown]
# ### Decision Tree Classifier
# %% [markdown]
# **Fit and predict**

# %%
## limit the depth to 5
dt = DecisionTreeClassifier(max_depth=5, random_state=15)
dt.fit(X_train, y_train)
dt_test_pred = dt.predict(X_test)

## scale
dt.fit(X_train_scale, y_train)
dt_test_pred_scale = dt.predict(X_test_scale)

# %% [markdown]
# **Confusion matrix**

# %%
dt_matrix = confusion_matrix(y_test, dt_test_pred)
print("DT, not scaled: " + "\n\n" + str(dt_matrix) + "\n")


# %%
dt_matrix_scale = confusion_matrix(y_test, dt_test_pred_scale)
print("DT, not scaled: " + "\n\n" + str(dt_matrix_scale) + "\n")

# %% [markdown]
# **Scores**

# %%
print_scores(dt_test_pred, "Decision Tree")


# %%
print_scores(dt_test_pred_scale, "Scaled Decision Tree")


# %%
auc_plot(dt_test_pred, "Decision Tree")


# %%
auc_plot(dt_test_pred_scale, "Scaled Decision Tree")

# %% [markdown]
# No difference with scaling for Decision Tree classifier
# %% [markdown]
# ### RandomForest
# %% [markdown]
# **Fit and predict**

# %%
rfc = RandomForestClassifier()

rfc.fit(X_train, y_train)
rfc_test_pred = rfc.predict(X_test)

## scale
rfc.fit(X_train_scale, y_train)
rfc_test_pred_scale = rfc.predict(X_test_scale)

# %% [markdown]
# **Confusion matrix**

# %%
rfc_matrix = confusion_matrix(y_test, rfc_test_pred)
print("RF, not scaled: " + "\n\n" + str(rfc_matrix) + "\n")


# %%
rfc_matrix_scale = confusion_matrix(y_test, rfc_test_pred_scale)
print("RF, scaled: " + "\n\n" + str(rfc_matrix_scale) + "\n")

# %% [markdown]
# **Scores**

# %%
print_scores(rfc_test_pred, "Random Forest")


# %%
print_scores(rfc_test_pred_scale, "Scaled Random Forest")


# %%
auc_plot(rfc_test_pred, "Random Forest")


# %%
auc_plot(rfc_test_pred_scale, "Scaled Random Forest")

# %% [markdown]
# Accuracy decreased slightly with scaling for Random Forest Classifier.
# %% [markdown]
# ### Compare the Models
# %% [markdown]
# **AUC Scores**

# %%
scores_not_scaled = pd.Series(
    {
        "Dummy": roc_auc_score(y_test, dummy_test_pred),
        "Logistic Regression": roc_auc_score(y_test, logr_test_pred),
        "Decision Tree": roc_auc_score(y_test, dt_test_pred),
        "Random Forest": roc_auc_score(y_test, rfc_test_pred),
    }
)

scores_scaled = pd.Series(
    {
        "Dummy": roc_auc_score(y_test, dummy_test_pred_scale),
        "Logistic Regression": roc_auc_score(y_test, logr_test_pred_scale),
        "Decision Tree": roc_auc_score(y_test, dt_test_pred_scale),
        "Random Forest": roc_auc_score(y_test, rfc_test_pred_scale),
    }
)

scores_df = pd.DataFrame({"Not Scaled": scores_not_scaled, "Scaled": scores_scaled})

# %% [markdown]
# Dataframe to compare AUC scores

# %%
scores_df

# %% [markdown]
# Highest AUC score is un-scaled Random Forest. Decision Tree is 2nd

# %%
accuracy_not_scaled = pd.Series(
    {
        "Dummy": accuracy_score(y_test, dummy_test_pred),
        "Logistic Regression": accuracy_score(y_test, logr_test_pred),
        "Decision Tree": accuracy_score(y_test, dt_test_pred),
        "Random Forest": accuracy_score(y_test, rfc_test_pred),
    }
)

accuracy_scaled = pd.Series(
    {
        "Dummy": accuracy_score(y_test, dummy_test_pred_scale),
        "Logistic Regression": accuracy_score(y_test, logr_test_pred_scale),
        "Decision Tree": accuracy_score(y_test, dt_test_pred_scale),
        "Random Forest": accuracy_score(y_test, rfc_test_pred_scale),
    }
)

accuracy_df = pd.DataFrame(
    {"Not Scaled": accuracy_not_scaled, "Scaled": accuracy_scaled}
)

# %% [markdown]
# Dataframe to compare accuracy

# %%
accuracy_df

# %% [markdown]
# The highest accuracy of predicting whether a client subsribes to a term deposit or not comes from using a **Decision Tree** Classifer regardless of scaling pre-processing. There may be a bias since the dataset is imbalanaced.
#
# **89.8%**
# %% [markdown]
# Below is a snippet of a sampling method to possibly improve imbalance.

# %%
# Try random sampling method
# because of the imbalance
# Attempt to minimize the bias

## shuffled_df = bank_df.sample(frac=1,random_state=1)
## sample of 5000 from 'no' values
## no_df = shuffled_df.loc[shuffled_df['y_num'] == 0].sample(n=5000,random_state=1)
## bank_df = pd.concat([yes_df, no_df])


# %%
