#%%
import pandas as pd
from sklearn.datasets import load_wine
import yaml
from sklearn import datasets

import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score,plot_confusion_matrix
from numpy import linalg as LA
from sklearn.preprocessing import MinMaxScaler

#%%
raw_data = datasets.load_wine()
print(raw_data)
features = pd.DataFrame(data=raw_data['data'],columns=raw_data['feature_names'])
data = features
data['target']=raw_data['target']
data['class']=data['target'].map(lambda ind: raw_data['target_names'][ind])
print(data.head())
#%%
data=data.loc[data['target']!=2]
data_copy=data.copy()
data.drop(['class'],inplace=True,axis=1)
params = yaml.safe_load(open("params.yaml"))
#%%
data1=data[['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
       'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
       'proanthocyanins', 'color_intensity', 'hue',
       'od280/od315_of_diluted_wines', 'proline', 'target']]
#%%
def generate_model_report(y_actual, y_predicted):
    print("Accuracy : " ,"{:.4f}".format(accuracy_score(y_actual, y_predicted)))     
    auc = roc_auc_score(y_actual, y_predicted)
    print("AUC : ", "{:.4f}".format(auc))
#%%    
data1=data[['alcohol','proline', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
       'flavanoids', 
       'proanthocyanins', 'color_intensity', 'hue',
       'target']]
X = data1.iloc[:, :-1]
y = data1.iloc[:, -1]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = params['prepare']['split'], random_state = 12)    
#%%
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
print("Training Report................ ")
generate_model_report(y_train,model.predict(X_train))
print("Testing Report................ ")

y_pred = model.predict(X_test)
generate_model_report(y_test,y_pred)
feat_importances_rf = pd.Series(model.feature_importances_, index=X_train.columns)
important_features_rf=feat_importances_rf.nlargest(50)
print('Feature Importance........') 
print(important_features_rf) 