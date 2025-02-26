# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 11:03:37 2025

@author: User
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('./data_fintech.csv')
summary = df.describe()
types = df.dtypes


# merevisi kolom numscreens
df['screen_list'] = df.screen_list.astype(str) + ','
df['nums_screens'] = df.screen_list.str.count(',')
df.drop(columns=['numscreens'], inplace=True)

# cek kolom hour
df.hour[1]
df.hour = df.hour.str.slice(1,3).astype(int)

# mendeteksi variable khusus numerik
df_numeric = df.drop(columns=['user', 'first_open',
                              'screen_list', 'enrolled_date'], inplace=False)

#  membuat histogram
sns.set()
plt.suptitle('hist numeric data')
for i in range(0, df_numeric.shape[1]):
    plt.subplot(3,3,i+1)
    fig = plt.gca()
    fig.set_title(df_numeric.columns.values[i])
    bin_count = np.size(df_numeric.iloc[:,i].unique())
    plt.hist(df_numeric.iloc[:,i], bins=bin_count)
    
    
#  membuat correlation matrix
corr = df_numeric.drop(columns=['enrolled'], inplace=False).corrwith(df_numeric.enrolled)
corr.plot.bar(title='correlation variable of enrolled decision')
corr_matrix = df_numeric.drop(columns=['enrolled'], inplace=False).corr()
print(corr_matrix.isnull().sum().sum())  # Jika hasilnya > 0, berarti ada NaN
sns.heatmap(corr_matrix, cmap='Blues')

mask = np.zeros_like(corr_matrix, dtype=bool)
mask[np.tril_indices_from(mask)] = False

# membuat correlation matrix dengan heatmap custom
ax=plt.axes()
my_cmap = sns.diverging_palette(200, 0, as_cmap=True)
sns.heatmap(corr_matrix, cmap=my_cmap, mask=mask, linewidths=0.5, 
            center=0, square=True)
plt.suptitle('correlation matrix custom')
plt.show()


# features engineering
from dateutil import parser
df.first_open = [parser.parse(data) for data in df.first_open]
df.enrolled_date = [parser.parse(data) if isinstance(data, str)
                    else data for data in df.enrolled_date]
df['difference'] = ((df.enrolled_date - df.first_open).dt.total_seconds() / 3600).dropna().astype(int)
print(df['difference'].dtype)  # Harusnya float64

# membuat plot histogram difference
plt.hist(df.difference.dropna(), range=[0,500])
plt.suptitle('selisih waktu antara enrolled date dengan first open')
plt.show()


df.loc[df.difference>48, 'enrolled'] = 0

# top screen dataset
df_ts = pd.read_csv('top_screens.csv')
df_ts_col = np.array(df_ts.loc[:, 'top_screens'])

df2 = df.copy()

# membuat kolom untuk top_screens
for screen in df_ts_col:
    df[screen] = df.screen_list.str.contains(screen).astype(int)
    
for screen in df_ts_col:
    df['screen_list'] = df.screen_list.str.replace(screen+'','')
    
# menghitung item non-top screens
df['lainnya'] = df.screen_list.str.count(',')


# urutkan data top screen
df_ts_col.sort()

# proses penggabungan beberapa kolom screen yang sama (funneling)
credit_screens = ['Credit1', 'Credit2', 'Credit3',
                  'Credit3Container', 'Credit3Dashboard']
df['credit_total'] = df[credit_screens].sum(axis=1)
df.drop(columns=credit_screens, inplace=True)

loan_screens = ['Loan', 'Loan2', 'Loan3', 'Loan4']
df['loan_total'] = df[loan_screens].sum(axis=1)
df.drop(columns=loan_screens, inplace=True)

saving_screens = ['Saving1', 'Saving2', 'Saving2Amount', 'Saving4',
                'Saving5', 'Saving6', 'Saving7', 'Saving8', 
                'Saving9', 'Saving10']
df['saving_total'] = df[saving_screens].sum(axis=1)
df.drop(columns=saving_screens, inplace=True)

cc_screens = ['CC1', 'CC1Category', 'CC3']
df['cc_total'] = df[cc_screens].sum(axis=1)
df.drop(columns=cc_screens, inplace=True)

print('total cc_screens', df.cc_total.sum())
print('total credit_screens', df.credit_total.sum())
print('total saving_screens', df.saving_total.sum())
print('total loan_screens', df.loan_total.sum())


# mendefinisikan variable dependen
var_enrolled = np.array(df['enrolled'])

# menghapus kolom yang tidak diperlukan
df.drop(columns=['user', 'first_open', 'screen_list', 
                 'enrolled','enrolled_date'], inplace=True)
df.drop(columns=['difference'], inplace=True)

# membagi data menjadi training dan test set (80:20)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    np.array(df), var_enrolled, test_size=0.2, random_state=111)

# agar skala datanya sama bisa lakukan standarisasi.. dan ada yang namanya normalisasi
# standarisasi = ketika kita mengasumsikan data yang kita miliki berdistribusi normal. maka nilai rata-ratanya akan berpusat dari minus ke titik 0
# normalisasi = ketika kita mengasumsikan bahwa data yang dimiliki tidak normal. maka range nilainya akan berpusat dari 0-1

# Pre-processing standarization (feature scaling)
from sklearn.preprocessing import StandardScaler
ss_X = StandardScaler()
X_train=ss_X.fit_transform(X_train)
X_test=ss_X.transform(X_test)

# delete kolom yang nilainya 0 semua dalam bentuk numpy
X_train = np.delete(X_train, 27, 1)
X_test = np.delete(X_test, 27, 1)

# membuat model
# logistic regression : algoritma supervised learning yang paling sederhana dan popular di gunakan. dia menggunakan probalilitas antara 0 dan 1
# untuk setiap baris kalau probabilitasnya < 1/2 maka masuk ke dalam golongan 0. kalau minimal lebih dari1/2 maka masuk ke dalam golongan 1
from sklearn.linear_model import LogisticRegression

# buat object modelnya. machine learning konvensional/classifier
classifier = LogisticRegression(random_state=0, 
                                solver='liblinear',
                                penalty='l1')

classifier.fit(X_train, y_train)

# memprediksi test set
y_pred=classifier.predict(X_test)

# evaluasi model dengan confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))

# evaluasi model dengan accuracy_score
evaluate = accuracy_score(y_test, y_pred)
print('accuracy : {:.2f}'.format(evaluate*100))

# menggunakan seaborn untuk confusion matrix/proses evaluasi
cm_label = pd.DataFrame(cm, columns=np.unique(y_test), 
                        index=np.unique(y_test))
cm_label.index.name='Actual'
cm_label.columns.name='Prediction'
sns.heatmap(cm_label, annot=True, cmap='Reds', fmt='g')


# melakukan cross-validation sebanyak 10x
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, 
                             X=X_train, y=y_train,
                             cv=10)
acc_mean = accuracies.mean()*100
acc_std = accuracies.std()*100
print('accuracy logistic regresion = {:.2f} +/- {:.2f}'.format(acc_mean, acc_std))



# mendefinisikan variable dependen
var_enrolled = df2['enrolled']

# membagi data menjadi training dan test set (80:20)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    df2, var_enrolled, test_size=0.2, random_state=111)

# get user column
train_user = X_train['user']
test_user = X_test['user']

# merging all the columns
y_pred_series = y_test.rename('real', inplace=True)
end_result = pd.concat([y_pred_series, test_user], axis=1).dropna()
end_result['pred'] = y_pred
end_result = end_result[['user','real', 'pred']].reset_index(drop=True)