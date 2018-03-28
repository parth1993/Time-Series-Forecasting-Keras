# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler, Imputer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import copy
import pickle


# In[4]:


class PreProcessing:
    """Preprocessing class for
       data analysis and exploration"""

    def __init__(self):
        self.data = pd.read_csv("Funchal_Madeira_Forecast_1.csv", encoding="latin1")

    def fill_na(self):
        imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
        self.data.iloc[:, 2:] = imputer.fit_transform(self.data.iloc[:, 2:])
        return self.data

    def drop_reduntant_cols(self):
        reduntant_cols = ['iUniZigZagPrice_V3 B#0',
                          'BinWave(13) B#1',
                          'FDI(5) B#1']
        self.data.drop(reduntant_cols, axis=1, inplace=True)

        return self.data

    def date_time_group(self, data):
        data.Date = pd.to_datetime(data.Date, errors='coerce')
        train = data.loc[data['Date'] <= '2016-12-31']
        test = data.loc[data['Date'] > '2016-12-31']

        return train, test

    def normalization(self, data):
        scaler = MinMaxScaler()
        scaler.fit(data)
        data = scaler.transform(data)

        return data

    def covariance_mat(self, data):
        cov_mat = np.corrcoef(data.T)

        return cov_mat

    def plot_graph(self):
        pass

    def eigen_decomp(self, cov_mat):
        eig_val, eig_vec = np.linalg.eig(cov_mat)
        for ev in eig_vec:
            np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
        print("OK")
        return eig_val, eig_vec


# # Preprocessing :
# ## 1. Cleaning of the data.
# ## 2. Filling empty fields.
# ## 3. Discard redundant variables i.e. iUniZigZagPrice_V3 B#0,BinWave(13) B#1,FDI(5) B#1

# In[5]:


preprocess = PreProcessing()

# In[6]:


"""the redundant columns i.e. iUniZigZagPrice_V3 B#0,BinWave(13) B#1,FDI(5) B#1"""
data = preprocess.drop_reduntant_cols()
data = preprocess.fill_na()
data.head()

# In[7]:


# relevant data i.e. columns except date time columns
rel_data = data.iloc[:, 2:]

# In[8]:


rel_data.head()

# In[9]:


rel_data_1 = rel_data.iloc[:, 5:8]
rel_data.drop(['iUniZigZagPrice_V3 B#6', 'iUniZigZagPrice_V3 B#7', 'iUniZigZagPrice_V3 B#8'], axis=1, inplace=True)
rel_data.head()

# In[10]:


# print(rel_data.head())


# ## X--Preprocessing Finished--X

# # Data Normalization, Building Covariance Matrix and Calulation of Eigen values and vectors :

# In[11]:


# ---Normailzation---#


# In[13]:


## Co-variance Matrix and eigen decomposition
covariance = preprocess.covariance_mat(rel_data)
# print(covariance)
# ## Plotting of eigen decomposition and getting k numbers of components/features which contains most of the data information.

# In[14]:
rel_data = rel_data.iloc[:,:4]
rel_data = preprocess.normalization(rel_data)
# X = rel_data[:,:-1]
# y = rel_data[:, -1]

print(rel_data.shape)
# x=['PC %s' %i for i in range(1, len(eig_val)+1)]
# trace1 = plt.bar(
#         x=['PC %s' %i for i in range(1, len(eig_val)+1)],
#         height=var_exp)
#
# trace1.xlabel='Explained variance in percent'
# trace1.ylabel='Explained variance by different principal components'
# rects = trace1.patches
#
# for rect, eig in zip(rects, var_exp):
#     height = rect.get_height()
#     plt.text(rect.get_x() + rect.get_width()/2, height + 5, "{0:.2f}".format(eig), ha='center', va='bottom')
# plt.show()
## PC 1 to PC 9 are the number of columns in rel_data


# # Dimension Reduction of data

# In[15]:



# In[16]:


a, b = preprocess.date_time_group(data)
# print(len(a),len(b))


# In[17]:

import time
import math
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
import numpy as np
import pandas as pd
import sklearn.preprocessing as prep




def standard_scaler(X_train, X_test):
    train_samples, train_nx, train_ny = X_train.shape
    test_samples, test_nx, test_ny = X_test.shape

    X_train = X_train.reshape((train_samples, train_nx * train_ny))
    X_test = X_test.reshape((test_samples, test_nx * test_ny))

    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)

    X_train = X_train.reshape((train_samples, train_nx, train_ny))
    X_test = X_test.reshape((test_samples, test_nx, test_ny))

    return X_train, X_test


def preprocess_data(stock, seq_len):
    amount_of_features = stock.shape[1]
    data = stock

    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])

    result = np.array(result)
    row = round(0.9 * result.shape[0])
    train = result[: int(row), :]

    train, result = standard_scaler(train, result)

    X_train = train[:, : -1]
    y_train = train[:, -1][:, -1]
    X_test = result[int(row):, : -1]
    y_test = result[int(row):, -1][:, -1]

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], amount_of_features))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], amount_of_features))

    return [X_train, y_train, X_test, y_test]

def build_model(layers):
    model = Sequential()

    # By setting return_sequences to True we are able to stack another LSTM layer
    model.add(LSTM(
        input_dim=layers[0],
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.4))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.3))

    model.add(Dense(
        output_dim=layers[3]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop", metrics=['accuracy'])
    print("Compilation Time : ", time.time() - start)
    return model

window = 8
X_train, y_train, X_test, y_test = preprocess_data(rel_data[:: -1], window)
print("X_train", X_train.shape)
print("y_train", y_train.shape)
print("X_test", X_test.shape)
print("y_test", y_test.shape)

model = build_model([X_train.shape[2], window, 100, 1])

model.fit(
    X_train,
    y_train,
    batch_size=768,
    nb_epoch=300,
    validation_split=0.1,
    verbose=0)


trainScore = model.evaluate(X_train, y_train, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))

testScore = model.evaluate(X_test, y_test, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))


diff = []
ratio = []
pred = model.predict(X_test)
for u in range(len(y_test)):
    pr = pred[u][0]
    ratio.append((y_test[u] / pr) - 1)
    diff.append(abs(y_test[u] - pr))

import matplotlib.pyplot as plt2

plt2.plot(pred, color='red', label='Prediction')
plt2.plot(y_test, color='blue', label='Ground Truth')
plt2.legend(loc='upper left')
plt2.show()


