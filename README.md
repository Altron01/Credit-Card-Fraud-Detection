# Credit Card Fraud Prediction Model

### Premise
It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase.

### Summary
This is a very interesting exercise, specialy because we have no idea what the features may be so there is no real business knowledge we can apply on them. To make things worse, there is an extreme difference between the number of "Fraud" cases and "No Fraud" as we will see later.

To begin the problem, we know that the dataset we are dealing with is unbalance. Just for fun we will show how different our correlation heatmap can be if we are using the whole dataset and not an undersampled version. After properly understanding the relation between the varaibles we proceed to delete those features that share a high correlation with each others since having them both will not bring value to our model. From the resulting group of features we select those who have a decent correlation with our target class. As our final step a logistic regression model was created and its performance reviewed.

Link to Kaggle: https://www.kaggle.com/mlg-ulb/creditcardfraud

# Table of Contents

1. [Heatmap Analysis](#ha)
2. [Undersampling Data](#ud)
3. [Heatmap Analysis with Undersampled Data](#haud)
4. [Removing Redundant Features](#rrf)
5. [Boxplot Analysis](#ba)
6. [Feature Selection](#fs)
7. [Distribution Analysis](#da)
8. [Oversamping Data with SMOTE](#od)
9. [Model creation](#mc)
10. [Model scoring](#ms)


```python
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
```


```python
df = pd.read_csv("creditcard.csv")
print(df.shape)
df.head()
```

    (284807, 31)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>...</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>-1.359807</td>
      <td>-0.072781</td>
      <td>2.536347</td>
      <td>1.378155</td>
      <td>-0.338321</td>
      <td>0.462388</td>
      <td>0.239599</td>
      <td>0.098698</td>
      <td>0.363787</td>
      <td>...</td>
      <td>-0.018307</td>
      <td>0.277838</td>
      <td>-0.110474</td>
      <td>0.066928</td>
      <td>0.128539</td>
      <td>-0.189115</td>
      <td>0.133558</td>
      <td>-0.021053</td>
      <td>149.62</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>1.191857</td>
      <td>0.266151</td>
      <td>0.166480</td>
      <td>0.448154</td>
      <td>0.060018</td>
      <td>-0.082361</td>
      <td>-0.078803</td>
      <td>0.085102</td>
      <td>-0.255425</td>
      <td>...</td>
      <td>-0.225775</td>
      <td>-0.638672</td>
      <td>0.101288</td>
      <td>-0.339846</td>
      <td>0.167170</td>
      <td>0.125895</td>
      <td>-0.008983</td>
      <td>0.014724</td>
      <td>2.69</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>-1.358354</td>
      <td>-1.340163</td>
      <td>1.773209</td>
      <td>0.379780</td>
      <td>-0.503198</td>
      <td>1.800499</td>
      <td>0.791461</td>
      <td>0.247676</td>
      <td>-1.514654</td>
      <td>...</td>
      <td>0.247998</td>
      <td>0.771679</td>
      <td>0.909412</td>
      <td>-0.689281</td>
      <td>-0.327642</td>
      <td>-0.139097</td>
      <td>-0.055353</td>
      <td>-0.059752</td>
      <td>378.66</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>-0.966272</td>
      <td>-0.185226</td>
      <td>1.792993</td>
      <td>-0.863291</td>
      <td>-0.010309</td>
      <td>1.247203</td>
      <td>0.237609</td>
      <td>0.377436</td>
      <td>-1.387024</td>
      <td>...</td>
      <td>-0.108300</td>
      <td>0.005274</td>
      <td>-0.190321</td>
      <td>-1.175575</td>
      <td>0.647376</td>
      <td>-0.221929</td>
      <td>0.062723</td>
      <td>0.061458</td>
      <td>123.50</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.0</td>
      <td>-1.158233</td>
      <td>0.877737</td>
      <td>1.548718</td>
      <td>0.403034</td>
      <td>-0.407193</td>
      <td>0.095921</td>
      <td>0.592941</td>
      <td>-0.270533</td>
      <td>0.817739</td>
      <td>...</td>
      <td>-0.009431</td>
      <td>0.798278</td>
      <td>-0.137458</td>
      <td>0.141267</td>
      <td>-0.206010</td>
      <td>0.502292</td>
      <td>0.219422</td>
      <td>0.215153</td>
      <td>69.99</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>




```python
df_X = df.drop(['Class'], axis=1)
df_y = df['Class']
```


```python
f, ax = plt.subplots(figsize=(7, 5))
sns.barplot(x=['Fraud', 'No Fraud'], y=df_y.value_counts().tolist(), palette="muted",ax = ax)
ax.axhline(0, color="k", clip_on=False)
ax.set_title("Fraud vs No Fraud")
ax.set_ylabel("Number of cases")
```




    Text(0, 0.5, 'Number of cases')




![png](output_4_1.png)


<a name="ha"></a>
## Heatmap Analysis


```python
f, ax = plt.subplots(figsize=(20, 20))
corrs = df.corr()
corrs = corrs.applymap(lambda x : round(x, 2))
sns.heatmap(corrs, annot=True, linewidths=.5, ax=ax, cmap="YlGnBu")
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7ff6de7195b0>




![png](output_6_1.png)


<a name="ud"></a>
## Undersampling Data


```python
from sklearn.datasets import make_classification
from imblearn.under_sampling import ClusterCentroids
cc = ClusterCentroids(random_state=42)
X_und, y_und = cc.fit_resample(df_X, df_y)
print('Resampled dataset shape %s' % X_und.shape[0])
```

    Resampled dataset shape 984



```python
f, ax = plt.subplots(figsize=(7, 5))
sns.barplot(x=['Fraud', 'No Fraud'], y=y_und.value_counts().tolist(), palette="muted",ax = ax)
ax.axhline(0, color="k", clip_on=False)
ax.set_title("Fraud vs No Fraud")
ax.set_ylabel("Number of cases")
```




    Text(0, 0.5, 'Number of cases')




![png](output_9_1.png)



```python

```

<a name="haud"></a>
## Heatmap Analysis with Undersampled Data


```python
f, ax = plt.subplots(figsize=(20, 20))
corrs = (X_und.join(y_und)).corr()
corrs = corrs.applymap(lambda x : abs(round(x, 2)))
sns.heatmap(corrs, annot=True, linewidths=.5, ax=ax, cmap="YlGnBu")
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7ff6d6eba040>




![png](output_12_1.png)


<a name="rrf"></a>
## Removing Redundant Features

As seen in the heatmap, there are certain variable that overlap at the point that they share almost 95% correlation. It was decided that from those features that hold 85%+ correlation would be removed since they do not provide much value.


```python
features = X_und.columns.tolist()
to_remove = []

for i in range(len(features)-1):
    for j in range(i, len(features)):
        if i != j and corrs[features[i]][features[j]] >= 0.85:
            print(cols[i], " -> ", features[j], " = ", corrs[features[i]][j])
            to_remove.append(features[i])
            break
for i in to_remove:
    features.remove(i)
```

    V2  ->  V3  =  0.9
    V5  ->  V10  =  0.86
    V15  ->  V10  =  0.89
    V18  ->  V12  =  0.89
    V19  ->  V12  =  0.94
    V20  ->  V14  =  0.88
    V24  ->  V17  =  0.95
    V25  ->  V18  =  0.97



```python
f, ax = plt.subplots(figsize=(15, 15))
corrs = (X_und[cols].join(y_und)).corr()
corrs = corrs.applymap(lambda x : round(x, 2))
sns.heatmap(corrs, annot=True, linewidths=.5, ax=ax, cmap="YlGnBu")
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7ff6cc0827c0>




![png](output_15_1.png)


<a name="ba"></a>
## Boxplot Analysis

We are using boxplots to further understand the statistic values of our features. As seen below, when using the raw data there is a huge number of outliers that might affect the training of our model. After removing outliers using the IQR our data is much more usable. We have to remember to be careful to not removed to many instances of the minority class in order to safeguard enough information for our model.

### Boxplots with raw data


```python
sns.set(style="whitegrid")

for fea in features:
    
    f, ax = plt.subplots(figsize=(6,6))
    sns.boxplot(x="Class", y=fea, palette=["r", "g"],
                data=df, ax=ax)
    plt.show()
```


![png](output_17_0.png)



![png](output_17_1.png)



![png](output_17_2.png)



![png](output_17_3.png)



![png](output_17_4.png)



![png](output_17_5.png)



![png](output_17_6.png)



![png](output_17_7.png)



![png](output_17_8.png)



![png](output_17_9.png)



![png](output_17_10.png)



![png](output_17_11.png)



![png](output_17_12.png)



![png](output_17_13.png)



![png](output_17_14.png)



![png](output_17_15.png)



![png](output_17_16.png)



![png](output_17_17.png)



![png](output_17_18.png)



![png](output_17_19.png)



![png](output_17_20.png)



![png](output_17_21.png)


### Boxplots after outlier removal


```python
new_df = df

outlier_limit = 1.5

for fea in features:
    plotData = df[[fea, 'Class']]
    
    nofraud = plotData[plotData['Class'] == 0]
    nofraud_q1 = nofraud[fea].quantile(0.25)
    nofraud_q3 = nofraud[fea].quantile(0.75)
    nofraud_iqr = nofraud_q3 - nofraud_q1
    nofraud_il = nofraud_q1 - outlier_limit * nofraud_iqr
    nofraud_ml = nofraud_q3 + outlier_limit * nofraud_iqr
    
    fraud = plotData[plotData['Class'] == 1]
    fraud_q1 = fraud[fea].quantile(0.25)
    fraud_q3 = fraud[fea].quantile(0.75)
    fraud_iqr = fraud_q3 - fraud_q1
    fraud_il = fraud_q1 - 1.5 * fraud_iqr
    fraud_ml = fraud_q3 + 1.5 * fraud_iqr
    plotData = plotData.loc[((plotData[fea] >= nofraud_il) & (plotData[fea] <= nofraud_ml) & (plotData['Class'] == 0)) |
                           ((plotData[fea] >= fraud_il) & (plotData[fea] <= fraud_ml) & (plotData['Class'] == 1))]
    
    bad_df = df.index.isin(plotData.index.tolist())
    new_df = new_df.drop(df[~bad_df].index.tolist(), errors="ignore")

for fea in features:
    f, ax = plt.subplots(figsize=(6,6))
    sns.boxplot(x="Class", y=fea, palette=["r", "g"],
                data=new_df, ax=ax)
    plt.show()
    
    
```


![png](output_19_0.png)



![png](output_19_1.png)



![png](output_19_2.png)



![png](output_19_3.png)



![png](output_19_4.png)



![png](output_19_5.png)



![png](output_19_6.png)



![png](output_19_7.png)



![png](output_19_8.png)



![png](output_19_9.png)



![png](output_19_10.png)



![png](output_19_11.png)



![png](output_19_12.png)



![png](output_19_13.png)



![png](output_19_14.png)



![png](output_19_15.png)



![png](output_19_16.png)



![png](output_19_17.png)



![png](output_19_18.png)



![png](output_19_19.png)



![png](output_19_20.png)



![png](output_19_21.png)



```python
df_X = new_df.drop(['Class'], axis=1)
df_y = new_df['Class']
```

<a name="fs"></a>
## Feature Selection

For the feature selection we based the desicion based on both the boxplots and the correlation matrix. Both sources provide similar insight and show that there is a conection between our features and our class and thos chosen have no extreme overlap on values. Our selected features are V2, V4, V6, V7, V14, V18

<a name="da"></a>
## Distribution Analysis


```python
final_features = ['V2', 'V4', 'V6', 'V7', 'V14', 'V18']
for i in final_features:
    sns.distplot(df_X[i], hist=False, color="g", kde_kws={"shade": True})
    plt.show()
```


![png](output_23_0.png)



![png](output_23_1.png)



![png](output_23_2.png)



![png](output_23_3.png)



![png](output_23_4.png)



![png](output_23_5.png)


<a name="od"></a>
## Oversamping Data with SMOTE


```python
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)
X_smote, y_smote = sm.fit_resample(df_X[final_features], df_y)
f, ax = plt.subplots(figsize=(7, 5))
sns.barplot(x=['Fraud', 'No Fraud'], y=y_smote.value_counts().tolist(), palette="muted",ax = ax)
ax.axhline(0, color="k", clip_on=False)
ax.set_title("Fraud vs No Fraud")
ax.set_ylabel("Number of cases")

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_smote, y_smote, test_size=0.30, random_state=42)
```


![png](output_25_0.png)


<a name="mc"></a>
## Model creation


```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=0).fit(X_train, y_train)

y_result = clf.predict(X_test)
```

<a name="ms"></a>
## Model scoring

As seen below in the confusion matrix and the PR-Curve, our model hold a good performance overall so we can be satisfied with our model. Just a reminder PR-Curve can be use in cases like this where our dataset is not balance.


```python
from sklearn.metrics import explained_variance_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve

print(clf.score(X_smote, y_smote))
print(explained_variance_score(y_test, y_result))

print('Confusion matrix')
print(confusion_matrix(y_test, y_result))

print(classification_report(y_test, y_result))

y_score = clf.decision_function(X_test)

from sklearn.metrics import average_precision_score
average_precision = average_precision_score(y_test, y_score)

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt

disp = plot_precision_recall_curve(clf, X_test, y_test)
disp.ax_.set_title('2-class Precision-Recall curve: '
                   'AP={0:0.2f}'.format(average_precision))
```

    0.9425490333829362
    0.7729668661737412
    Confusion matrix
    [[45414  1521]
     [ 3870 43142]]
                  precision    recall  f1-score   support
    
               0       0.92      0.97      0.94     46935
               1       0.97      0.92      0.94     47012
    
        accuracy                           0.94     93947
       macro avg       0.94      0.94      0.94     93947
    weighted avg       0.94      0.94      0.94     93947
    
    Average precision-recall score: 0.99





    Text(0.5, 1.0, '2-class Precision-Recall curve: AP=0.99')




![png](output_29_2.png)



```python

```
