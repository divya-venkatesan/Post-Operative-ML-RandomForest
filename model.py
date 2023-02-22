import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle

data = pd.read_csv("post-operative.csv",sep = ",", names = ["L-Core","L-Surf","L-O2","L-BP","Surf-STBL","Core-STBL","BP-STBL","Comfort","Decision"])

data.head()

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
le=preprocessing.LabelEncoder()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

L_Core_vals = ["high","mid","low"]
le=preprocessing.LabelEncoder()
le.fit(L_Core_vals)
L_Core_vals_dict={index: value for index, value in enumerate(le.classes_)}
print(L_Core_vals_dict)

L_Surf_vals = ["high","mid","low"]
le=preprocessing.LabelEncoder()
le.fit(L_Core_vals)
L_Surf_vals_dict={index: value for index, value in enumerate(le.classes_)}
print(L_Surf_vals_dict)

L_O2_vals = ["excellent","good"]
le=preprocessing.LabelEncoder()
le.fit(L_O2_vals)
L_O2_vals_dict={index: value for index, value in enumerate(le.classes_)}
print(L_O2_vals_dict)

L_BP_vals = ["high","mid","low"]
le=preprocessing.LabelEncoder()
le.fit(L_BP_vals)
L_BP_vals_dict={index: value for index, value in enumerate(le.classes_)}
print(L_BP_vals_dict)

surf_STBL_vals = ["stable","mod-stable","unstable"]
le=preprocessing.LabelEncoder()
le.fit(surf_STBL_vals)
surf_STBL_vals_dict={index: value for index, value in enumerate(le.classes_)}
print(surf_STBL_vals_dict)

core_STBL_vals = ["stable","mod-stable","unstable"]
le=preprocessing.LabelEncoder()
le.fit(core_STBL_vals)
core_STBL_vals_dict={index: value for index, value in enumerate(le.classes_)}
print(core_STBL_vals_dict)

BP_STBL_vals = ["stable","mod-stable","unstable"]
le=preprocessing.LabelEncoder()
le.fit(BP_STBL_vals)
BP_STBL_vals_dict={index: value for index, value in enumerate(le.classes_)}
print(BP_STBL_vals_dict)

number = LabelEncoder()
data["L-Core"] = number.fit_transform(data['L-Core'])
data["L-Surf"] = number.fit_transform(data['L-Surf'])
data["L-O2"] = number.fit_transform(data['L-O2'])
data["L-BP"] = number.fit_transform(data['L-BP'])
data["Surf-STBL"] = number.fit_transform(data['Surf-STBL'])
data["Core-STBL"] = number.fit_transform(data['Core-STBL'])
data["BP-STBL"] = number.fit_transform(data['BP-STBL'])
data["Comfort"] = number.fit_transform(data['Comfort'])

X = data[["L-Core","L-Surf","L-O2","L-BP","Surf-STBL","Core-STBL","BP-STBL","Comfort"]]
y = data["Decision"]

X_train, X_test, y_train, y_test = train_test_split(X,y)
test_size = 0.20,
random_state = 0

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test= sc.transform(X_test)

classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

pickle.dump(classifier, open("model.pkl", "wb"))

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

y_pred = classifier.predict(X_test)
ac = accuracy_score(y_test,y_pred, normalize = True)
print(ac)

cm = confusion_matrix(y_test, y_pred)
print(cm)

plt.figure(figsize=(9,9))
sns.heatmap(cm,annot=True,  linewidth = 5.0, square = True, cbar = False, cmap = "Spectral")
plt.ylabel("Actual Values")
plt.xlabel("Predicted Values")
plt.show()

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

print(classifier.predict([[0,0,0,0,0,0,0,12]]))
