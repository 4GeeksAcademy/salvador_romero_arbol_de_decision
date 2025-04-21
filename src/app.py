
def eliminar_atipicos(datos, columnas):
    new_data = datos
    for i in columnas:
        q1=new_data[i].quantile(0.25)
        q3=new_data[i].quantile(0.75)
        iqr = q3-q1
        low_lim = q1 - 1.5*iqr
        hi_lim = q3 + 1.5*iqr
        rem = new_data[(new_data[i]>=hi_lim) | (new_data[i]< low_lim)]
        new_data = new_data.drop(index=rem.index)
    return new_data.copy()

# your code here
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import GridSearchCV

data = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/decision-tree-project-tutorial/main/diabetes.csv")
data_uni = data.drop_duplicates()


data_clean = eliminar_atipicos(data_uni,data_uni.columns.drop("Outcome"))

scaler = StandardScaler()
norm_features = scaler.fit_transform(data_clean)
data_scal = pd.DataFrame(norm_features, index = data_clean.index,columns=data_clean.columns)
data_scal["Outcome"] = data_clean["Outcome"]

col = ["Pregnancies", "Glucose", "BMI", "Age"]
X = data_scal[col]
Y = data_scal["Outcome"]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 8)

tree_model = DecisionTreeClassifier(max_depth=5)
tree_model.fit(x_train,y_train)
y_pred = tree_model.predict(x_test)
score = accuracy_score(y_test,y_pred)
print(f"score 1: {score}")


hyperparams = {
    "criterion": ["gini", "entropy"],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

grid = GridSearchCV(tree_model, hyperparams, scoring = "accuracy", cv = 10)

grid.fit(x_train, y_train)

print(f"Best hyperparameters: {grid.best_params_}")

tree_model2 = DecisionTreeClassifier(criterion="gini",max_depth=5, min_samples_leaf=4, min_samples_split=5)
tree_model2.fit(x_train,y_train)
y_pred2 = tree_model2.predict(x_test)
score2 = accuracy_score(y_test,y_pred2)
print(f"score 1: {score2}")