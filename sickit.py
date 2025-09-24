1)
a=list(input("Enter the list values:").split())
b=list(input("Enter the Sublist values:").split())
if all(element in a for element in b):
 print("The sublist is present in the list")
else:
 print("The sublist is not present in the list")


2)
marks=[{"m1":90,"m2":50},{"m1":50,"m2":43},
{"m1":95,"m2":100}]
for mark in marks:
 mark["average"]=(mark["m1"]+mark["m2"])/2
 print(mark) 


3)
a=(10,20,30)
print("Tuple ",a)
print("Type:",type(a))
print("Accessing Index -> 1 value of tuple",a[1])
b=('a','b','c')
print("Concatenation",a+b)
c=('Python')*3
print("Repetition",c)
print("Slicing",b[1:])
d=[1,2,3,4]
d=tuple(d)
print("Type Conversion",d)

4)
import numpy as np
import pandas as pd
arr1=np.array([[1,2,3],[4,5,6]])
arr2=np.array([[1,2,3],[4,5,6]])
result_numpy=np.power(arr1,arr2)
print("Numpy Power Result")
print(result_numpy)
print("Numpy ** Operator Result")
print(arr1**arr2)
Series1=pd.Series(arr1.flatten())
Series2=pd.Series(arr2.flatten())
result_pandas=Series1.pow(Series2)
print("Pandas Power Result")
print(result_pandas) 


5)
import string
sentence = 'The five boxing wizards jump quickly'
def ispanagram(s):
 alphabet = set(string.ascii_lowercase)
 return alphabet <= set(s.lower())
if ispanagram(sentence):
 print("Yes, It is a Pangram")
else:
 print("No, It is not a Pangram")


6)
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold,
cross_val_score
X, y = datasets.load_iris(return_X_y=True)
clf = DecisionTreeClassifier(random_state=42)
k_folds = KFold(n_splits=5)
scores = cross_val_score(clf, X, y, cv=k_folds)
print("Cross Validation Scores:", scores)
print("Average CV Score:", scores.mean())
print("Number of CV Scores used in Average:",
len(scores)) 


7)
import csv
file=open('data.csv','r')
reader=csv.reader(file)
for row in reader:
 print(row) 


Program 8: (excel columns sno,customer name,age,annual score,spending score)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
X = pd.read_csv("d:\\ML\\Mall_Customers.csv").iloc[:, [3,4]].values
plt.plot(range(1,11), [KMeans(n_clusters=i, init='k-means++', random_state=42).fit(X).inertia_ for i in range(1,11)], 'o-')
plt.title("Elbow Method"); plt.xlabel("Clusters"); plt.ylabel("WCSS"); plt.show()
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42).fit(X)
y = kmeans.labels_
colors = ['blue','green','red','cyan','magenta']
for i,c in enumerate(colors): plt.scatter(X[y==i,0], X[y==i,1], s=100, c=c, label=f"Cluster {i+1}")
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=300, c='yellow', marker='*', label="Centroids")
plt.title("Clusters of Customers"); plt.xlabel("Annual Income (k$)"); plt.ylabel("Spending Score (1-100)")
plt.legend(); plt.show()



PROGRAM-9 :
import numpy as np from sklearn.model_selection import train_test_split from sklearn.naive_bayes
import GaussianNB from sklearn.metrics import accuracy_score
x=np.array([[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],[9,10],[10,11]])
y=np.array([0,0,0,0,0,1,1,1,1,1])
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
model=GaussianNB() model.fit(x_train,y_train) y_pred=model.predict(x_test)
accuracy=accuracy_score(y_test,y_pred) print("Accuracy: {:.2f}%".format(accuracy*100))


PROGRAM-10 :
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
data=load_iris()
X=data.data
y=data.target
binary_indices=np.where(y!=2)
X=X[binary_indices]
y=y[binary_indices]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
model=LogisticRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy : {accuracy:.2f}")
print('Classification report:')
print(classification_report(y_test,y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test,y_pred))
X_train_2D=X_train[:,:2]
X_test_2D=X_test[:,:2]
model_2D=LogisticRegression()
model_2D.fit(X_train_2D,y_train)
h=0.2
x_min,x_max=X_train_2D[:,0].min()-1,X_train_2D[:,0].max()+1
y_min,y_max=X_train_2D[:,0].min()-1,X_train_2D[:,0].max()+1
xx,yy=np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
Z=model_2D.predict(np.c_[xx.ravel(),yy.ravel()])
Z=Z.reshape(xx.shape)
plt.contourf(xx,yy,Z,alpha=0.8)
plt.scatter(X_train_2D[:,0],X_train_2D[:,1],c=y_train,edgecolor='k',marker='o')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision boundary')
plt.show()


PROGRAM-11 :
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
data = load_iris()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
base_models = [
 ('decision_tree', DecisionTreeClassifier()),
 ('knn', KNeighborsClassifier()),
 ('svc', SVC(probability=True))
]
meta_model = LogisticRegression()
stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)
stacking_model.fit(X_train, y_train)
y_pred = stacking_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))


PROGRAM-12 :
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import numpy as np
import pandas as pd
X, Y = make_blobs(n_samples=500, centers=2, random_state=0, cluster_std=0.40)
plt.scatter(X[:, 0], X[:, 1], c=Y, s=50, cmap='spring')
plt.title('Synthetic Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
clf = SVC(kernel='linear')
x = pd.read_csv("path/to/cancer.csv")
if 'malignant' in x.columns and 'benign' in x.columns:
 y = x.iloc[:, 30].
 x_features = np.column_stack((x['malignant'], x['benign']))
 clf.fit(x_features, y)
 prediction1 = clf.predict([[120, 990]])
 prediction2 = clf.predict([[85, 550]])
 print(f"Prediction for [120, 990]: {prediction1}")
 print(f"Prediction for [85, 550]: {prediction2}")
else:
 print("Columns 'malignant' and 'benign' not found in the CSV file")
xfit = np.linspace(-1, 3.5)
plt.scatter(X[:, 0], X[:, 1], c=Y, s=50, cmap='spring')
for m, b, d in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.9, 0.2)]:
 yfit = m * xfit + b
 plt.plot(xfit, yfit, '-k')
 plt.fill_between(xfit, yfit - d, yfit + d, edgecolor='none', color='#AAAAAA', alpha=0.4)
plt.xlim(-1, 3.5)
plt.title('Decision Boundaries')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
