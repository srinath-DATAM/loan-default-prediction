import pandas as pd
df=pd.read_csv(r"C:\Loan-Default-Prediction\data\cleaned_loan_data.csv")
print(df.shape)
print("data loaded successfully")
#seperate target features (y) and input features (X)
target_column="Loan_Status"
if target_column in df.columns:
    # Map 'Y'/'N' to 1 and 0
    y=df[target_column].map({"Y":1,"N":0})
    # Drop target and any ID columns from X
    x=df.drop(columns=[target_column,"Customer_ID"])
    print("Target (y) and Features (X) separated successfully!")
else:
    print("Target column{target_column} not found in the dataset.")

x=pd.get_dummies(x,drop_first=True)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)
print("x train shape:",x_train.shape)
print("y train shape :",y_train.shape)
print("y train shape counts:",y_train.value_counts())
print("using smote synthetice minoirty oversampling technique to balance the dataset")
from imblearn.over_sampling import SMOTE
smote=SMOTE(random_state=42)
x_train_resampled,y_train_resampled=smote.fit_resample(x_train,y_train)
print("x train resampled shape:",x_train_resampled.shape)
print("y train resampled shape:",y_train_resampled.shape)
print(y_train_resampled.value_counts())
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train_resampled)
x_test_scaled=scaler.transform(x_test)
from sklearn.linear_model import LogisticRegression#BASE MODEL 
lr=LogisticRegression(random_state=42)
lr.fit(x_train_scaled,y_train_resampled)
y_pred_lr=lr.predict(x_test_scaled)
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print("Logistic Regression Model Evaluation:")
print("accuracy_score:",accuracy_score(y_test,y_pred_lr))
print("classification_report:\n",classification_report(y_test,y_pred_lr))
import matplotlib.pyplot as plt 
import seaborn as sns 
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test,y_pred_lr),annot=True,fmt="d",cmap="viridis")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(random_state=42)
dt.fit(x_train_scaled,y_train_resampled)
y_pred_dt=dt.predict(x_test_scaled)
print("Decision Tree Model Evaluation:")
print("accuracy_score:",accuracy_score(y_test,y_pred_dt))
print("classification_report:\n",classification_report(y_test,y_pred_dt))
sns.heatmap(confusion_matrix(y_test,y_pred_dt),annot=True,fmt="d",cmap="viridis",
            xticklabels=['Rejected (0)', 'Approved (1)'],
            yticklabels=['Rejected (0)', 'Approved (1)'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(random_state=42)
rf.fit(x_train_scaled,y_train_resampled)
y_pred_rf=rf.predict(x_test_scaled)
print("Random Forest Model Evaluation:")
print("accuracy_score:",accuracy_score(y_test,y_pred_rf))
print("classification_report:\n",classification_report(y_test,y_pred_rf))
sns.heatmap(confusion_matrix(y_test,y_pred_rf),annot=True,fmt="d",cmap="viridis",
            xticklabels=['Rejected (0)', 'Approved (1)'],
            yticklabels=['Rejected (0)', 'Approved (1)'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
from sklearn.svm import SVC
svm=SVC(random_state=42)
svm.fit(x_train_scaled,y_train_resampled)
y_pred_svm=svm.predict(x_test_scaled)
print("Support Vector Machine Model Evaluation:")
print("accuracy_score:",accuracy_score(y_test,y_pred_svm))
print("classification_report:\n",classification_report(y_test,y_pred_svm))
sns.heatmap(confusion_matrix(y_test,y_pred_svm),annot=True,fmt="d",cmap="viridis",
            xticklabels=['Rejected (0)', 'Approved (1)'],   
            yticklabels=['Rejected (0)', 'Approved (1)'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
from sklearn.ensemble import GradientBoostingClassifier
gb=GradientBoostingClassifier(random_state=42)
gb.fit(x_train_scaled,y_train_resampled)
y_pred_gb=gb.predict(x_test_scaled)
print("Gradient Boosting Model Evaluation:")
print("accuracy_score:",accuracy_score(y_test,y_pred_gb))
print("classification_report:\n",classification_report(y_test,y_pred_gb))
sns.heatmap(confusion_matrix(y_test,y_pred_gb),annot=True,fmt="d",cmap="viridis",
            xticklabels=['Rejected (0)', 'Approved (1)'],   
            yticklabels=['Rejected (0)', 'Approved (1)'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
print("next step is to perform hyperparamater tunninng for all 5 models")
from scipy.stats import randint,loguniform
model_to_tune={"Logistic Regression":{"model":LogisticRegression(random_state=42),
"params":{"C":loguniform(1e-3,1e3),"solver":["saga","liblinear"]}},
"Decision Tree":{"model":DecisionTreeClassifier(random_state=42),
"params":{"max_depth":randint(3,20),"min_samples_split":randint(2,20)}},
"RandomForest":{"model":RandomForestClassifier(random_state=42),
"params":{"n_estimators":randint(50,200),"min_samples_split":randint(2,20)}},
"Svm":{"model":SVC(random_state=42),"params":{"C":loguniform(1e-3,1e3),"kernel":["linear","rbf"]}},
"gradient boosting":{"model":GradientBoostingClassifier(random_state=42),"params":{"n_estimators":randint(50,200),"learning_rate":loguniform(1e-3,1e-1)}}}
# 2. Loop through each model and run RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV
best_models={}
for name,data in model_to_tune.items():
    print(f"tuning hyperparameters for {name}...")
    rc=RandomizedSearchCV(estimator=data["model"],param_distributions=data["params"],
    cv=5,n_iter=10,scoring="accuracy",n_jobs=-1,random_state=42)
    rc.fit(x_train_scaled,y_train_resampled)
    best_models[name]=rc.best_estimator_
    print(f"best parameters for {name}:{rc.best_params_}")
    print(f"best score for {name}:{rc.best_score_}")
    print(f"best estimator for {name}:{rc.best_estimator_}")
best_model=best_models["RandomForest"]
import seaborn as sns 
importance=best_model.feature_importances_
feature_importance_columns=pd.DataFrame({"feature":x_train.columns,"importance":importance})
sort_importance=feature_importance_columns.sort_values(by="importance",ascending=False)
plt.figure(figsize=(10,6))
sns.barplot(data=sort_importance,y="feature",x="importance",palette="viridis")
plt.subplots_adjust(left=0.3)
plt.ylabel("Features")
plt.xlabel("Importance")
plt.title("Feature Importance")
plt.show()
print("saving the best model to disk")
import joblib 
joblib.dump(best_model,"load_model.pkl")
joblib.dump(scaler,"scaler.pkl")
joblib.dump(list(x_train.columns), "features.pkl")
print("model and scaler saved successfully to disk")