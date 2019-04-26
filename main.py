#Import libs
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
%matplotlib inline

df = pd.read_excel('Dados.xlsx')

df.drop('Cpf,' axis=1, inplace=True)

x = df.drop('Pagamento', axis=1)
y = df['Pagamento']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=101)

from sklearn.linear_model import LogisticRegression

#Logistic Regression
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X_test)

#Metrics
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
print(classification_report(y_test, predictions))
print('\n')
print(confusion_matrix(y_test, predictions))
print('\n')
print('MSE: {}'.format(mean_squared_error(y_test, predictions)))

#log

model_log = y_test
predito_log = predictions

file = open('Log_logistic.txt', 'w')

for i in range(len(model_log)):
    file.write('O valor real foi: {} e o valor predito foi: {} \n'.format(model_log.iloc[i],predito_log[i]))

file.close()

#Gradient boosting Regressor

from sklearn.ensemble import GradientBoostingRegressor

gradient = GradientBoostingRegressor()
gradient.fit(X_train, y_train)
gb_pred = gradient.predict(X_test)

#Metrics
print('MSE: {}'.format(mean_squared_error(y_test, gb_pred)))
print('\n')
len(y_test)

#log de gradient regressor

model_gradient = y_test
predito_gradient = gb_pred

file = open('Log_gb.txt', 'w')

for i in range(len(model_gradient)):
    file.write('O valor real foi: {} e o valor predito foi: {} \n'.format(model_gradient.iloc[i],predito_gradient[i]))

file.close()