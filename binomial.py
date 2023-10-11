import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


csv_path = r'C:\Users\Martín\Documents\DATA SCIENCE\Platzi\ML_projects\logistic_regression\binomial\WA_Fn-UseC_-Telco-Customer-Churn.csv'
df_data = pd.read_csv(csv_path)

df_data.head(5)
df_data.info()

df_data.TotalCharges = pd.to_numeric(df_data.TotalCharges, errors='coerce')

df_data.isnull().sum()
df_data.dropna(inplace=True)

df_data.drop('customerID', axis=1, inplace=True)

# One-hot encoding in label
df_data['Churn'].replace(to_replace='Yes', value=1, inplace=True)
df_data['Churn'].replace(to_replace='No', value=0, inplace=True)

df_data_preprocessing = df_data.copy()

# One-hot encoding in categorical features
df_data_preprocessing = pd.get_dummies(df_data_preprocessing)
df_data_preprocessing.head(5)

# How is it interpreted the corr() function?
fig = plt.figure(figsize=(15, 9))
df_data_preprocessing.corr()['Churn'].sort_values(ascending=True).plot(kind='bar')
plt.show()

# What does the MinMaxScaler()?
scaler = MinMaxScaler()
df_data_preprocessing_scaled = scaler.fit_transform(df_data_preprocessing)
df_data_preprocessing_scaled = pd.DataFrame(df_data_preprocessing_scaled)
df_data_preprocessing_scaled.columns = df_data_preprocessing.columns
df_data_preprocessing_scaled.head(5)

sns.countplot(data=df_data, x='gender', hue='Churn')
plt.show()


def plot_categorical(column):
    plt.figure(figsize=(5, 5))
    sns.countplot(data=df_data, x=column, hue='Churn')
    plt.show()


# ???
column_cat = df_data.select_dtypes(include='object').columns

for _ in column_cat:
    plot_categorical(_)

fig = plt.figure(figsize=(10, 10))
sns.pairplot(data=df_data, hue='Churn')  # hue?
plt.show()

X = df_data_preprocessing_scaled.drop('Churn', axis=1)
y = df_data_preprocessing_scaled['Churn'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression()
result = model.fit(X_train, y_train)

prediction_test = model.predict(X_test)
print(metrics.accuracy_score(y_test, prediction_test))

# 0.795734597156398

model.predict_proba(X_test)

model.coef_ # what is this doing?
model.feature_names_in_ #?????

weights = pd.Series(model.coef_[0],
                    index=X.columns.values)
print(weights.sort_values(ascending=False)[:10].plot(kind='bar'))

plt.figure(figsize=(11, 11))
cm = confusion_matrix(y_test, prediction_test, labels=model.classes_) #?????
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_) #????
disp.plot(cmap='gray')
plt.show()
