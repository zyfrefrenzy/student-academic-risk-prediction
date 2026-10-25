import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



raw_data ={
    'Student_ID':[1001,1002,1003,1004,1005,1001,1006,1007,1008,1009],
    'County':['Nakuru','Narok','Kakamega','Nakuru','Nairobi',None,'Narok','Nakuru','Kakamega',None],
    'Study_Hours_Per_Week':[15,8,20,None,12,15,30,10,25,18],
    'Attendance_Percent':[85,60,92,55,78,85,95,65,88,None],
    'Fee_Balance_Ksh':[0,15000,0,25000,5000,0,0,18000,2000,12000],
    'Previous_grade':[72,48,85,45,65,72,90,50,82,58],
    'Student_Type':['Reqular','Regular','scholarship','regular','scholarship','regular','regular','scholarship','regular','regular'],
    'Academic_Risk':[ 0,1,0,1,0,0,0,1,0,1]

}
df=pd.DataFrame(raw_data)


print(df)

#inspect dataset
print()
print("First five rows:")
print(df.head())



print("\n dataset dimensions:")
print(df.shape)#returns number of rows and collumns

print()
print("data types:")
print(df.dtypes)#give the data type of each collumn

print()
print("Collumn name:")#dispays number of collumns
print(df.columns)

print()
print("Dataset information:")
print(df.info())#display info of the data data types collumns number of non missing values in the collumn


print()
print("summary statistics:")
print(df.describe())#summary statistics for numerical data


print()
print("Number of duplicate rows:")
print(df.duplicated().sum())#counts the number of rows with the same values in every collumn

df=df.drop_duplicates()#removes duplicate values to avoid counting the same record twice

print()
print("shape after removing duplicates:")
print(df.shape)



print()
print("missing values before cleaning:")
print(df.isnull().sum())#used to count the number of missing values

#use median to fill the numerical values of eacg collumn with numerical values
numerical_columns=[
    'Study_Hours_Per_Week',
    'Attendance_Percent'
]

for column in numerical_columns:
    df[column]=df[column].fillna(df[column].median())

#filling the county column with the most repeated county
df['County']=df['County'].fillna(df['County'].mode()[0])

print()
print("missing values after cleaning")
print(df.isnull().sum())#used to check for any other missing values

print()
print("Cleaned dataset")
print(df)


sns.set_theme(style="whitegrid")#for apperance of the chart

print()
print("Number of students in each academic risk category:")
print(df['Academic_Risk'].value_counts().sort_index())#displays number of students in each risk  category 0 means no risk while 1 means at risk 

#1.accademmic risk distribution count no or students with risk and no risk 
sns.countplot(data=df, x='Academic_Risk')
plt.title("Distirbution of Academic risk")
plt.xlabel("Academic risk (0=NO 1= YES)")
plt.ylabel("Number of students")
plt.tight_layout()
plt.show()

#study hours and academic risk
sns.boxplot(data=df, x='Academic_Risk' , y='Study_Hours_Per_Week')
plt.title("Study hours vs academic risk")
plt.xlabel("Academic risk (0=No 1=Yes)")
plt.ylabel("Study Hours Per Week")
plt.tight_layout()
plt.show()

#attendance and academic risk
sns.boxplot(data=df, x='Academic_Risk',y='Attendance_Percent')
plt.title("Attendance vs Academic risk")
plt.xlabel("Academic risk (0=No 1=Yes)")
plt.ylabel("attendace percentage")
plt.tight_layout()
plt.show()

#previous grade and academic risk 
sns.boxplot(data=df, x='Academic_Risk',y='Previous_grade')
plt.title("previous grade vs academic risk ")
plt.xlabel("Academic risk (0=No 1=Yes)")
plt.ylabel("Previous Grade")
plt.tight_layout()
plt.show()

#attendance lower 60% is low 60% to 80% is medium and abpve that it is high
df["Attendance_Category"]=pd.cut(
    df["Attendance_Percent"],
    bins=[-1,60,80,100],
    labels=["low","medium","high"]
)
#colummn to check outstanding fee balance 1 has 0 has no
df["Has_Fee_Balance"]=(df["Fee_Balance_Ksh"]>0).astype(int)

print()
print("dataset new features:")
print(df[["Attendance_Percent",
          "Attendance_Category",
          "Fee_Balance_Ksh",
          "Has_Fee_Balance"]])

numerical_df=df.select_dtypes(include="number")#selects only numerical columns

correlation_matrix=numerical_df.corr()#this finds correlation and values range between -1 and +1

print()
print("correlation matrix:")
print(correlation_matrix)

plt.figure(figsize=(10,7))#figure for the correlation  heatmap


#heatmap
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)
plt.title("Correlation heatmap for the students data")
plt.tight_layout()
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report
from sklearn.tree import plot_tree
from sklearn.tree import export_text

df_encoded=pd.get_dummies(
    df,
    columns=["County","Student_Type","Attendance_Category"],
    drop_first=True
)
X=df_encoded.drop(columns=["Student_ID","Academic_Risk"])
y=df_encoded["Academic_Risk"]

#split the datset into training(70%) and testing(30%) sets helps preserve the propotion of each risk class
X_train,X_test, y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

#decsion tree
model =DecisionTreeClassifier(
    max_depth=3,#limit tree depth to reduce complexity
    random_state=42#make results reproducible
)
#training model
model.fit(X_train,y_train)
y_pred=model.predict(X_test)#used trained model to predict rosk for the testing data
accuracy=accuracy_score(y_test,y_pred)#calculate accuracy

print()
print("Decsion tree accuracy:")
print(accuracy)

print()
print("classification report:")#presicion recall and f1 score for each class
print(classification_report(
    y_test,
    y_pred,zero_division=0
))
plt.figure(figsize=(20,10))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Not at risk","At risk"],
    filled=True,
    rounded=True,
    fontsize=9
)
plt.title("Decision tree for predicting academic risk")
plt.tight_layout()
plt.show()

#converting to tree decision into readable text rules
tree_rules=export_text(
    model,
    feature_names=list(X.columns)
)
print()
print("Decision tree rules:")
print(tree_rules)

results=pd.DataFrame({#creating table with the below
    "Actual_Academic_Risk":y_test.values,
    "Predicted_Academic_Risk":y_pred
})
results.to_csv("student_risk_prediction.csv",index=False)#save results in a csv file 
print()
print("Actual vs predicted academic risk")
print(results)

with open("Decision_tree_rules.txt","w") as file:
    file.write(tree_rules)


#confirming
print()
print("results saved to student_risk_prediction.csv")
print("decision tree rules save to decision_tree_rules.txt")
