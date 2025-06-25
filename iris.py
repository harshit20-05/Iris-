import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Data
data = pd.read_csv("C:/Users/attri/Downloads/Iris.csv")

# Data Overview
data.info()
data.describe()
data.head()
data.tail()
data.columns
for col in data.columns:
    print(col)
data.shape
data.count()
data.nunique()
data['Species'].unique()

# Data Cleaning and Preprocessing
data.rename(columns={
    'SepalLengthCm': 'sepal_length',
    'SepalWidthCm': 'sepal_width',
    'PetalLengthCm': 'petal_length',
    'PetalWidthCm': 'petal_width'
}, inplace=True)
data.drop('Id', axis=1, inplace=True)
data.isnull().sum()

# Handle Duplicates
data.duplicated().sum()
data = data.drop_duplicates(keep='first', ignore_index=True)

# Value Counts and Grouping
data.value_counts('Species')
data.groupby("Species").min()

# Univariate Visualizations
sns.countplot(x='Species', data=data, palette="viridis")
plt.title("Species Distribution")
plt.figure(figsize=(10, 5))
sns.boxplot(x='Species', y='sepal_length', data=data, palette="viridis")
plt.title("Boxplot of Sepal length by Species")
sns.boxplot(x='Species', y='sepal_width', data=data, palette="viridis")
sns.boxplot(x='Species', y='petal_length', data=data, palette='viridis')
sns.boxplot(x='Species', y='petal_width', data=data, palette='viridis')
sns.violinplot(x='Species', y='petal_length', data=data, palette="colorblind")
sns.displot(data['sepal_length'], bins=30, kde=True)

# Feature Distributions by Class
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes[0, 0].set_title("Sepal Length")
axes[0, 0].hist(data['sepal_length'], bins=20)
axes[0, 1].set_title("Sepal Width")
axes[0, 1].hist(data['sepal_width'], bins=5)
axes[1, 0].set_title("Petal Length")
axes[1, 0].hist(data['petal_length'], bins=6)
axes[1, 1].set_title("Petal Width")
axes[1, 1].hist(data['petal_width'], bins=6)

# Multivariate Analysis
sns.scatterplot(x='sepal_length', y='petal_length', hue='Species', data=data)
plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
plt.title("Scatter Plot of Sepal Length vs. Petal Length")
plt.show()
sns.scatterplot(x='sepal_length', y='sepal_width', hue='Species', data=data)
plt.title("Scatter Plot of Sepal Length vs. Sepal Width")
plt.show()
sns.pairplot(data, hue='Species', height=2)

# Correlation Analysis
numerical_data = data.select_dtypes(include=['number'])
correlation_matrix = numerical_data.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix of Numerical Features")
plt.show()

# Outlier Detection and Handling
sns.boxplot(data=data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']])
plt.title("Boxplot to Detect Outliers")
plt.show()
Q1 = data['sepal_width'].quantile(0.25)
Q3 = data['sepal_width'].quantile(0.75)
IQR = Q3 - Q1
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR
upper = np.where(data['sepal_width'] >= upper_limit)
lower = np.where(data['sepal_width'] <= lower_limit)
data.drop(upper[0], inplace=True)
data.drop(lower[0], inplace=True)

# Feature Engineering
data['sepal_ratio'] = data['sepal_length'] / data['sepal_width']
data['petal_ratio'] = data['petal_length'] / data['petal_width']
sns.scatterplot(x='sepal_ratio', y='petal_ratio', hue='Species', data=data)
data['petal_size'] = pd.cut(data['petal_length'], bins=[0, 2, 4, 7], labels=['Small', 'Medium', 'Large'])
sns.countplot(x='petal_size', hue='Species', data=data)

# Encoding Categorical Variables
label_encoder = LabelEncoder()
data['species'] = label_encoder.fit_transform(data['Species'])
data['petal_size'] = label_encoder.fit_transform(data['petal_size'])

# Model Preparation
X = data.drop(labels=["Species", "species"], axis=1)
y = data["species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN Model Training and Evaluation
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Compare Different k Values
accuracy_scores = []
for k in range(1, 20):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    accuracy_scores.append(acc)

plt.plot(range(1, 20), accuracy_scores, marker='o')
plt.title("KNN Accuracy for Different k Values")
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Accuracy")
plt.show()
