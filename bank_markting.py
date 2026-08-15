import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,ConfusionMatrixDisplay,classification_report)

#set up data and split data into features and target variable and then split into training and testing sets
data = pd.read_csv("bank-additional-full.csv", sep=";")
print(data.info())
print(data.head())
print(data["y"].value_counts())

X = data.drop("y", axis=1)
y = data["y"].map({"yes": 1,"no": 0})
print(y.value_counts())

categorical_columns = X.select_dtypes(include=["object"]).columns
numerical_columns = X.select_dtypes(exclude=["object"]).columns
print(list(categorical_columns))
print(list(numerical_columns))

#Convert categorical variables into numbers
preprocessor = ColumnTransformer(transformers=[("cat",OneHotEncoder(handle_unknown="ignore"),categorical_columns)],remainder="passthrough")

#create decision tree model and run it
decision_tree = DecisionTreeClassifier(max_depth=7,random_state=42)
model = Pipeline(steps=[("preprocessor", preprocessor),("classifier", decision_tree)])
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
print("results for Decision Tree:")
print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1-score:", round(f1, 4))
print("Classification Report:")
print(classification_report(y_test, y_pred))


#feature importance and visualization
encoded_features = (model.named_steps["preprocessor"].named_transformers_["cat"].get_feature_names_out(categorical_columns))
feature_names = list(encoded_features) + list(numerical_columns)
importances = model.named_steps["classifier"].feature_importances_
feature_importance = pd.DataFrame({"Feature": feature_names,"Importance": importances})
feature_importance = feature_importance.sort_values(by="Importance",ascending=False)
print("Top 10 important features:")
print(feature_importance.head(10))

cm = confusion_matrix(y_test, y_pred)
display = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=["No", "Yes"])
display.plot()
plt.title("Decision Tree - Bank Marketing")
plt.show()
top_features = feature_importance.head(10)
plt.figure(figsize=(10, 6))
plt.barh(top_features["Feature"][::-1],top_features["Importance"][::-1])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 10 Decision Tree Feature Importances")
plt.tight_layout()
plt.show()