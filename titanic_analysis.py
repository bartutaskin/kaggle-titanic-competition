import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_validate,
    learning_curve,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import re


pd.set_option("display.max_columns", None)
pd.set_option("display.width", 800)

train_data = pd.read_csv("./data/train.csv")
test_data = pd.read_csv("./data/test.csv")

#! Exploratory Data Analysis


def check_def(dataframe, head=5):
    print("####### Shape #######")
    print(dataframe.shape)
    print("####### Types #######")
    print(dataframe.dtypes)
    print("####### Head #######")
    print(dataframe.head(head))
    print("####### Tail #######")
    print(dataframe.tail(head))
    print("####### NA Count #######")
    print(dataframe.isnull().sum())


check_def(train_data)


def grab_col_names(dataframe, cat_th=5, car_th=20):
    # Grabbing categorical columns.
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtype == "O"]
    # Survived is numeric but categoric.
    num_but_cat = [
        col
        for col in dataframe.columns
        if dataframe[col].dtype != "O" and dataframe[col].nunique() < cat_th
    ]
    cat_but_car = [
        col
        for col in dataframe.columns
        if dataframe[col].dtype == "O" and dataframe[col].nunique() > car_th
    ]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # Grabbing numeric columns
    num_cols = [col for col in dataframe.columns if dataframe[col].dtype != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")
    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(train_data)
print(f"Categorical Columns:  {cat_cols}")
print(f"Numerical Columns:  {num_cols}")
print(f"Cardinal Columns:  {cat_but_car}")

num_cols = [col for col in num_cols if col != "PassengerId"]
cat_cols = [col for col in cat_cols if col != "Survived"]


def cat_summary(dataframe, col_name, plot=False):
    print(
        pd.DataFrame(
            {
                col_name: dataframe[col_name].value_counts(),
                "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe),
            }
        )
    )

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


for col in cat_cols:
    cat_summary(train_data, col)


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


for col in num_cols:
    num_summary(train_data, col, True)

train_data[train_data["Fare"] > 200]


def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


def target_summary_with_cat(dataframe, target, categorical_col):
    print(
        pd.DataFrame(
            {"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}
        ),
        end="\n\n\n",
    )


for col in num_cols:
    target_summary_with_num(train_data, "Survived", col)

for col in cat_cols:
    target_summary_with_cat(train_data, "Survived", col)

# Visualize survival rates by sex.
survival_rates_by_sex = train_data.groupby("Sex")["Survived"].mean()
sns.barplot(
    x=survival_rates_by_sex.index, y=survival_rates_by_sex.values, palette="coolwarm"
)
plt.title("Survival Rates by Gender")
plt.xlabel("Gender")
plt.ylabel("Survival Rate")
plt.ylim(0, 1)  # Survival rates range from 0 to 1
plt.show()

# Visualize survival rates by class.
survival_rates_by_class = train_data.groupby("Pclass")["Survived"].mean()
sns.barplot(
    x=survival_rates_by_class.index,
    y=survival_rates_by_class.values,
    palette="coolwarm",
)
plt.title("Survival Rates by Class")
plt.xlabel("Class")
plt.ylabel("Survival Rate")
plt.ylim(0, 1)
plt.show()


#! Outliers Analysis
def outlier_thresholds(dataframe, col_name, q1=0.01, q3=0.99):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquartile = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquartile
    low_limit = quartile1 - 1.5 * interquartile

    return low_limit, up_limit


def check_outlier(dataframe, col_name, q1=0.01, q3=0.99):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[
        (dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)
    ].any(axis=None):
        return True
    else:
        return False


for col in num_cols:
    print(check_outlier(train_data, col))

#! Missing Values Analysis
train_data.isnull().sum()


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (
        dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100
    ).sort_values(ascending=False)

    missing_df = pd.concat(
        [n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss", "ratio"]
    )
    print(missing_df, end="\n")
    if na_name:
        return na_columns


missing_values_table(train_data)

# Cabin column has a lot of null values. I don't think it effects the prediction. That's why I'll drop it.
train_data = train_data.drop(columns=["Cabin"])
train_data["Age"] = train_data.groupby(["Pclass", "Sex"])["Age"].transform(
    lambda x: x.fillna(x.mean())
)
train_data["Embarked"] = train_data["Embarked"].fillna(train_data["Embarked"].mode()[0])

train_data.isnull().sum()

#! Correlation
# Correlation between numerical columns.
f, ax = plt.subplots(figsize=[12, 8])
sns.heatmap(train_data[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

# Correlation between numerical columns and Target. Since I dropped Survived column in the beggining, I've created new one.
# numeric_cols_df = train_data.select_dtypes(include=["number"])
# numeric_cols_df = [col for col in numeric_cols_df if col != "PassengerId"]
# corr_with_survived = numeric_cols_df.corr()["Survived"].drop("Survived")

# plt.figure(figsize=(8, 6))
# corr_with_survived.sort_values(ascending=False).plot(kind="bar", color="teal")
# plt.title("Correlation with 'Survived'")
# plt.ylabel("Correlation Coefficient")
# plt.xlabel("Features")
# plt.ylim(-1, 1)
# plt.show()

#! Base Model Before Feature Engineering
train_data_copy = train_data.drop(columns=["PassengerId", "Name", "Ticket"])


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(
        dataframe, columns=categorical_cols, drop_first=drop_first
    )
    return dataframe


cat_cols = ["Sex", "Embarked"]
train_data_copy = one_hot_encoder(train_data_copy, cat_cols, True)
train_data_copy.head()

y = train_data_copy["Survived"]
X = train_data_copy.drop(columns=["Survived"])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(algorithm="SAMME", random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Classifier": SVC(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
}

model_scores = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)
    model_scores[model_name] = accuracy_score(y_val, predictions)

for model_name, accuracy in model_scores.items():
    print(f"{model_name} Accuracy: {accuracy:.4f}")

# Logistic Regression Accuracy: 0.8212
# Random Forest Accuracy: 0.8268
# Gradient Boosting Accuracy: 0.8212
# AdaBoost Accuracy: 0.7989
# K-Nearest Neighbors Accuracy: 0.7151
# Support Vector Classifier Accuracy: 0.6536
# Decision Tree Accuracy: 0.7989

#! Feature Engineering
train_df = train_data.copy()

# Categorizing Age.
train_df["NEW_AGE_CATEGORY"] = pd.cut(
    train_df["Age"],
    bins=[0, 18, 35, 55, 90],
    labels=["0-18", "19-35", "36-55", "56+"],
    right=True,
)

train_df.drop(columns=["PassengerId"], inplace=True)

train_df["NEW_FAMILY_SIZE"] = train_df["SibSp"] + train_df["Parch"] + 1
train_df["NEW_IS_ALONE"] = 0
train_df.loc[train_df["NEW_FAMILY_SIZE"] < 2, "NEW_IS_ALONE"] = 1

train_df["Age"] = train_df["Age"].round().astype(int)

train_df.head(10)


# Extracting Titles.
# def extract_title(name):
#     title = re.search('([A-Za-z]+)\.', name)  # Matches any word followed by a period
#     if title:
#         return title.group(1)
#     return None

# train_df['NEW_TITLE'] = train_df['Name'].apply(extract_title)

# cat_cols, num_cols, cat_but_car = grab_col_names(train_df)
# print(f"Categorical Columns:  {cat_cols}")
# print(f"Numerical Columns:  {num_cols}")
# print(f"Cardinal Columns:  {cat_but_car}")
# num_cols = [col for col in num_cols if col != 'PassengerId']


#! Encoding
train_df.drop(columns=["Name", "Ticket"], inplace=True)

cat_cols, num_cols, cat_but_car = grab_col_names(train_df)
print(f"Categorical Columns:  {cat_cols}")
print(f"Numerical Columns:  {num_cols}")
print(f"Cardinal Columns:  {cat_but_car}")

cat_cols = [col for col in cat_cols if col != "Survived"]


train_df = one_hot_encoder(train_df, cat_cols, True)

#! Scaling
scaler = StandardScaler()
train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
train_df[num_cols].head()

#! Model
y = train_df["Survived"]
X = train_df.drop(columns=["Survived"])

rf_model = RandomForestClassifier(random_state=42)

# Before hyper parameter optimization.
cv_results = cross_validate(rf_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results["test_accuracy"].mean()  # 0.81

# After hyper parameter optimization.
rf_params = {
    "max_depth": [5, 8, None],
    "max_features": [3, 5, 7, "auto"],
    "min_samples_split": [2, 5, 8, 10],
    "n_estimators": [100, 200, 500],
}

rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(
    X, y
)

rf_best_grid.best_params_

rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=42).fit(X, y)

cv_results = cross_validate(
    rf_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"]
)
cv_results["test_accuracy"].mean()  # 0.84
cv_results["test_f1"].mean()
cv_results["test_roc_auc"].mean()

#! test.csv
test_df = test_data.copy()

test_df["Age"] = test_df.groupby(["Pclass", "Sex"])["Age"].transform(
    lambda x: x.fillna(x.mean())
)
test_df["Embarked"] = test_df["Embarked"].fillna(test_df["Embarked"].mode()[0])
test_df["Fare"] = test_df["Fare"].fillna(test_df["Fare"].mean())

test_df["NEW_AGE_CATEGORY"] = pd.cut(
    test_df["Age"],
    bins=[0, 18, 35, 55, 90],
    labels=["0-18", "19-35", "36-55", "56+"],
    right=True,
)
test_df["NEW_FAMILY_SIZE"] = test_df["SibSp"] + test_df["Parch"] + 1
test_df["NEW_IS_ALONE"] = 0
test_df.loc[test_df["NEW_FAMILY_SIZE"] < 2, "NEW_IS_ALONE"] = 1

test_df["Age"] = test_df["Age"].round().astype(int)

test_df.drop(columns=["Name", "Ticket"], inplace=True)

cat_cols, num_cols, cat_but_car = grab_col_names(test_df)
cat_cols = [col for col in cat_cols if col != "Survived"]
test_df = one_hot_encoder(test_df, cat_cols, drop_first=True)

test_df.drop(columns=["Cabin"], inplace=True)
test_df.drop(columns=["PassengerId"], inplace=True)
# Scale numeric features
scaler = StandardScaler()
test_df[num_cols] = scaler.fit_transform(test_df[num_cols])

# Final Test Data
X_test = test_df

predictions = rf_final.predict(X_test)

submission = pd.DataFrame(
    {"PassengerId": test_data["PassengerId"], "Survived": predictions}
)

submission.to_csv("submission.csv", index=False)
print("Submission file has been saved as 'submission.csv'")
