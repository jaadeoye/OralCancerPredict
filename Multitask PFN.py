# imports necessary modules
import os
import torch
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression as LR
from tabpfn import TabPFNClassifier
import shap
shap.initjs()

# Load training and test data
xls = pd.ExcelFile('[DATA]')
df_train = pd.read_excel(xls, '[TRAIN]')
df_test = pd.read_excel(xls, '[EXTERNAL TESTING]')
features = ['Age','Sex','Category','Cancer', 'Lesion', 'Tongue', 'Buccal','Retromolar',
            'Gingiva','Palate','Number','Homogeneity','Induration', 'Size', 'Excision_status',
            'Surgery_type', 'WHO_Dysplasia']
x = df_train[features]
y1 = df_train.REC
y2 = df_train.MT_status
a = df_test[features]
b1 = df_test.REC
b2 = df_test.MT_status
c = df_test.Binary_Dysplasia # comparator

#Call Prior-data Fitted Network
clf1 = TabPFNClassifier()
clf2 = TabPFNClassifier()
clf1.fit(x, y1)
clf2.fit(x, y2)
scores_1_auc = cross_val_score(clf1, x, y1, cv=10, scoring='roc_auc')
scores_1_brier = cross_val_score(clf1, x, y1, cv=10, scoring='neg_brier_score')
scores_2_auc = cross_val_score(clf2, x, y2, cv=10, scoring='roc_auc')
scores_2_brier = cross_val_score(clf2, x, y2, cv=10, scoring='neg_brier_score')

# Predict probabilities_task 1
pred_rec=clf1.predict(a)
pred1_rec=clf1.predict_proba(a)[:, 1]
pred2_rec=clf1.predict_proba

# Predict probabilities_task 2
pred_MT=clf2.predict(a)
pred1_MT=clf2.predict_proba(a)[:, 1]
pred2_MT=clf2.predict_proba

#AUC for task 1
roc_rec=roc_auc_score(b1,pred1_rec, average=None)
roc1_rec=roc_auc_score(b1,c, average=None)
brier_rec=brier_score_loss(b1, pred1_rec)
brier1_rec=brier_score_loss(b1, c) #comparator

#AUC for task 2
roc_MT=roc_auc_score(b2,pred1_MT, average=None)
roc1_MT=roc_auc_score(b2,c, average=None)
brier_MT=brier_score_loss(b2, pred1_MT)
brier1_MT=brier_score_loss(b2, c) #comparator

print(scores_1_auc)
print(scores_1_brier)
print(scores_2_auc)
print(scores_2_brier)
print(brier_rec)
print(roc_rec)
print(brier_MT)
print(roc_MT)


#Shapley values for task 1
explainer = shap.Explainer(pred2_rec, x)
shap_test = explainer(a)
print(f"Shap values length: {len(shap_test)}\n")
print(f"Sample shap value:\n{shap_test[0]}")
shap_df = pd.DataFrame(shap_test.values[:,:,1], 
                       columns=shap_test.feature_names, 
                       index=a.index)
shap_df
shap.plots.bar(shap_test[:,:,1])
shap.plots.beeswarm(shap_test[:,:,1])
shap.summary_plot(shap_test[:,:,1], plot_type='violin')
shap.summary_plot(shap_test[:,:,1], max_display=28, color_bar=False, show=False)
plt.colorbar(label='Feature value')
shap.plots.waterfall(shaptest[0,:,0])

#Shapley values for task 2
explainer = shap.Explainer(pred2_MT, x)
shap_test = explainer(a)
print(f"Shap values length: {len(shap_test)}\n")
print(f"Sample shap value:\n{shap_test[0]}")
shap_df = pd.DataFrame(shap_test.values[:,:,1], 
                       columns=shap_test.feature_names, 
                       index=a.index)
shap_df
shap.plots.bar(shap_test[:,:,1])
shap.plots.beeswarm(shap_test[:,:,1])
shap.summary_plot(shap_test[:,:,1], plot_type='violin')
shap.summary_plot(shap_test[:,:,1], max_display=28, color_bar=False, show=False)
plt.colorbar(label='Feature value')
shap.plots.waterfall(shaptest[0,:,0])

plt.show()

