import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,ConfusionMatrixDisplay,classification_report)

#set up and preprocess data
data = pd.read_csv("Customer-Churn-Records.csv")
print(data.info())
print(data.head())
data = data.drop(["RowNumber", "CustomerId", "Surname", "Complain"],axis=1)
X = data.drop("Exited", axis=1)
y = data["Exited"]
categorical_columns = X.select_dtypes(include=["object"]).columns
numerical_columns = X.select_dtypes(exclude=["object"]).columns
print(list(categorical_columns))
print(list(numerical_columns))
preprocessor = ColumnTransformer(
    transformers=[("cat",OneHotEncoder(handle_unknown="ignore"),categorical_columns),
    ("num",StandardScaler(),numerical_columns)])


#create kNN model also train and test it
knn = KNeighborsClassifier(n_neighbors=5)
model = Pipeline(steps=[("preprocessor", preprocessor),("classifier", knn)])
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("Results:")
print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1-score:", round(f1, 4))
print("Classification Report:")
print(classification_report(y_test, y_pred))

#Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
display = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=["Stayed", "Exited"])
display.plot()
plt.title("kNN - Bank Customer Churn")
plt.show()