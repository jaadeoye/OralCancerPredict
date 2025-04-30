import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.metrics import RocCurveDisplay
import shap
from sklearn.linear_model import LogisticRegression as LR
shap.initjs()
from xgboost import XGBClassifier
import pickle

#import data
xls = pd.ExcelFile('[DATA_PATH]')
df_train = pd.read_excel(xls, '[Training DATA]')
df_test = pd.read_excel(xls, '[Testing DATA]')
cal=pd.read_excel(xls, 'PLATT SCALING')
features = ['KRT13','p53','Age','Sex','smoking',
            'Previous','lesion','Tongue/FOM','Buccal/Labial','Retromolar',
            'Gingiva','Palate','number','homogeneity','Induration', 'dysplasia']
x = df_train[features]
y = df_train.MT
a = df_test[features]
b = df_test.MT

#Base model - XGBoost
xgb = XGBClassifier(n_estimators=1000, max_depth = 1, learning_rate=0.01, subsample=0.7,
                   scale_pos_weight=7.416666666667, random_state=123)
xgb.fit(x,y)

#10-fold CV performance
scores = cross_val_score(xgb, x, y, cv=10, scoring='roc_auc')
scores
print((scores.mean(), scores.std()))

#Save model
pickle.dump(xgb, open("OralCancerPredict.pkl", "wb"))

#Prediction
pred=xgb.predict(a)
pred1=xgb.predict_proba(a)[:, 1]
pred2=xgb.predict_proba

#confusion matrix
cnf_matrix = metrics.confusion_matrix(b, pred)
cnf_matrix

#plot confusion matric
class_names=['No MT','MT'] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

#AUROC and Brier score (uncalibrated)
roc_test=roc_auc_score(b,pred1, average=None)
brier_test=brier_score_loss(b, pred1)

#Platt scaling
y_cal = cal.MT
xgb = cal.XGBoost
lr = LR()                                                       
lr.fit( xgb.values.reshape( -1, 1 ), y_cal)     # LR needs X to be 2-dimensional
p_calibrated = lr.predict_proba(pred1.reshape( -1, 1 ))[:,1]

#Calibrated Brier score
brier_cal=brier_score_loss(b, p_calibrated)

#Print Metrics
print(cnf_matrix)
print(brier_test)
print(roc_test)
print("Accuracy:",metrics.accuracy_score(b,pred))
print("Precision:",metrics.precision_score(b,pred))
print("Recall:",metrics.recall_score(b,pred))
print("F1 score:",metrics.f1_score(b,pred))

#Shapley values and Explainability
explainer = shap.Explainer(pred2, x)
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
plt.show()

