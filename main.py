# Nayeem Rafsan 24-01-2022
from turtle import width
from wsgiref import validate
from sklearn.model_selection import train_test_split
from sklearn import datasets, preprocessing, linear_model
import pytz
from pytz import timezone
from datetime import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional
from sklearn.preprocessing import MinMaxScaler
from keras.layers import BatchNormalization
from tqdm.keras import TqdmCallback
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

st.set_page_config(
    page_title="Solar Radiation prediction",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

st.title('Solar radiation prediction')


@st.cache
def load_data():
    data = pd.read_csv('SolarPrediction.csv')
    data['Date'] = [x.split()[0] for x in data['Data']]
    data = data.drop('Data', axis=1)
    return data


data = load_data()

df = data.sort_values(['UNIXTime'], ascending=[True])
col1, col2 = st.columns(2)
with col1:
    st.subheader('Raw data')
    st.write(df.head(n=5))
# adding new features from unixtime
with col2:
    st.subheader('After feature engineering')
    dhaka = timezone('Asia/Dhaka')
    df.index = pd.to_datetime(df['UNIXTime'], unit='s')
    df.index = df.index.tz_localize(pytz.utc).tz_convert(dhaka)
    df['MonthOfYear'] = df.index.strftime('%m').astype(int)
    df['DayOfYear'] = df.index.strftime('%j').astype(int)
    df['WeekOfYear'] = df.index.strftime('%U').astype(int)
    df['Hour'] = df.index.hour
    df['Month'] = df.index.month
    df['Date'] = df.index.date

    df['TimeSunRise'] = pd.to_datetime(
        df['TimeSunRise'], format='%H:%M:%S').dt.time
    df['TimeSunSet'] = pd.to_datetime(
        df['TimeSunSet'], format='%H:%M:%S').dt.time
    df['Total_time'] = pd.to_datetime(df['TimeSunSet'], format='%H:%M:%S').dt.hour - \
        pd.to_datetime(df['TimeSunRise'], format='%H:%M:%S').dt.hour
    df = df[['Temperature', 'Pressure', 'Humidity', 'MonthOfYear',
             'DayOfYear', 'WeekOfYear', 'Total_time', 'Radiation']]
    st.write(df.head(n=5))
col1, col2 = st.columns(2)


with col1:
    X = df[['Temperature', 'Pressure', 'Humidity', 'MonthOfYear',
            'DayOfYear', 'WeekOfYear', 'Total_time']]
    y = df['Radiation']
    st.subheader('Before removing outliers')
    f, ax = plt.subplots(1, 4, figsize=(20, 4))
    sns.histplot(y, stat="density", bins=25, ax=ax[0])
    sns.histplot(X['Pressure'], stat="density", bins=25, ax=ax[1])
    sns.histplot(X['Humidity'], stat="density", bins=25, ax=ax[2])
    sns.histplot(X['Temperature'], stat="density", bins=25, ax=ax[3])
    st.pyplot(f)


with col2:
    # removing outliers
    X = df[(np.abs(stats.zscore(df)) < 2.5).all(axis=1)]
    y = X['Radiation']
    X = X.drop(columns='Radiation', axis=1)
    st.subheader('Outliers removed')
    f, ax = plt.subplots(1, 4, figsize=(20, 4))
    sns.histplot(y, stat="density", bins=25, ax=ax[0])
    sns.histplot(X['Pressure'], stat="density", bins=25, ax=ax[1])
    sns.histplot(X['Humidity'], stat="density", bins=25, ax=ax[2])
    sns.histplot(X['Temperature'], stat="density", bins=25, ax=ax[3])
    st.pyplot(f)
# normalizing the values
X = (X-X.min())/(X.max()-X.min())
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=5)
# classifier


def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'KNN':
        K = st.slider('Neighbors', 1, 15)
        L = st.slider('Leaf size', 1, 15)
        params['K'] = K
        params['leaf_size'] = L
    elif clf_name == 'Linear Regression':
        pass
    elif clf_name == 'Decision Tree':
        R = st.slider('Random state', 0, 20)
        params['R'] = R
    elif clf_name == 'Random Forest':
        max_depth = st.slider('Max depth', 1, 40)
        params['max_d'] = max_depth
        n_estimators = st.slider('N estimators', 1, 100)
        params['n_esti'] = n_estimators
    elif clf_name == 'Extra Trees':
        max_depth = st.slider('Max depth', 1, 40)
        params['max_depth'] = max_depth
        n_estimators = st.slider('N estimators', 1, 100)
        params['n_estimators'] = n_estimators
    elif clf_name == 'Neural Network':
        params['epochs'] = st.slider('Epochs', 10, 50)
        params['layers'] = st.slider('Hidden layers', 2, 15)
    return params


def get_regressor(clf_name, params):
    clf = None
    history = None
    if clf_name == 'KNN':
        clf = KNeighborsRegressor(
            n_neighbors=params['K'], leaf_size=params['leaf_size'])
    elif clf_name == 'Linear Regression':
        clf = LinearRegression()
    elif clf_name == 'Decision Tree':
        clf = DecisionTreeRegressor(random_state=params['R'])
    elif clf_name == 'Random Forest':
        clf = RandomForestRegressor(n_estimators=params['n_esti'],
                                    max_depth=params['max_d'], random_state=100)
    elif clf_name == 'Extra Trees':
        clf = ExtraTreesRegressor(n_estimators=params['n_estimators'],
                                  max_depth=params['max_depth'], random_state=100)
    elif clf_name == 'Neural Network':
        model = Sequential()
        model.add(BatchNormalization(input_shape=[7]))
        for layer in range(params['layers']):
            model.add(Dense(256,
                            kernel_initializer='normal', activation='relu'))
            model.add(Dropout(0.2))
            model.add(BatchNormalization())
        model.add(Dense(1, kernel_initializer='normal'))
        # dataset = df.copy()
        # sequence_len = 100
        # sequences = []
        # labels = []
        # start_idx = 0
        # for stop_idx in range(sequence_len,len(dataset)):
        #     sequences.append(dataset.iloc[start_idx:stop_idx])
        #     labels.append(dataset.iloc[stop_idx])
        #     start_idx += 1
        # X_train, y_train = (np.array(sequences),np.array(labels))
        # model.add(LSTM(units=64))
        # model.add(Dropout(0.1))
        # model.add(LSTM(units=32))
        # model.add(Dense(1, kernel_initializer='normal'))
        # # Compile model
        model.compile(loss='mean_squared_error',
                      optimizer='adam', metrics=['mse'])
        history = model.fit(X_train, y_train, epochs=params['epochs'],
                            verbose=1, validation_split=0.2, callbacks=[TqdmCallback(verbose=2)])
        clf = model.predict(X_test)
    return clf, history


col1, col2, col3 = st.columns([2, 1, 3])
with col1:
    classifier_name = st.selectbox(
        'Select regressor',
        ('Decision Tree', 'Extra Trees', 'Neural Network',
         'KNN', 'Linear Regression', 'Random Forest')
    )
    params = add_parameter_ui(classifier_name)

    clf, history = get_regressor(classifier_name, params)
if history == None:
    with col2:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.write('MSE score:')
        st.info(round(mse, 3))
        st.write('R2 score')
        st.info(round(r2, 3))
    with col3:
        y_pd = pd.DataFrame({'Y predict': y_pred, 'Y truth': y_test})
        st.area_chart(y_pd)
        # fig, ax = plt.subplots(figsize=(10, 4))
        # plt.scatter(X_test['MonthOfYear'], y_test)
        # plt.plot(y_pred)
        # plt.ylabel('y pred')
        # plt.xlabel('y truth')
        # st.pyplot(fig)
else:
    with col2:
        st.write('MSE score:')
        mse = mean_squared_error(y_test, clf)
        st.info(round(mse, 3))
        r2 = r2_score(y_test, clf)
        st.write('R2 score:')
        st.info(round(r2, 3))
    with col3:
        fig, ax = plt.subplots(figsize=(10, 4))
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.ylabel('loss')
        plt.xlabel('epoch')
        st.pyplot(fig)


# Neural network
# normalizer = tf.keras.layers.Normalization(axis=-1)
# normalizer.adapt(np.array(X_train))


# def build_and_compile_model(norm):
#     model = keras.Sequential([
#         norm,
#         layers.Dense(64, activation='relu'),
#         layers.Dense(64, activation='relu'),
#         layers.Dense(1)
#     ])

#     model.compile(loss='mean_absolute_error',
#                   optimizer=tf.keras.optimizers.Adam(0.001))
#     return model


# dnn = build_and_compile_model(normalizer)
# history = dnn.fit(
#     x=X_train,
#     y=y_train,
#     validation_split=0.2,
#     verbose=0, epochs=10)
# y_pred = dnn.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# st.write(mse)
# model = Sequential()
# model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
# model.add(Dense(1, kernel_initializer='normal'))
# # Compile model
# model.compile(loss='mean_squared_error', optimizer='adam')
# history = model.fit(X_train, y_train, epochs=15,
#                     verbose=1, validation_data=(X_test, y_test))
# # y_pred = model.predict(X_test, y_test)
# # mse = mean_squared_error(y_test, y_pred)
# col1, col2 = st.columns([1, 2])
# with col1:
#     fig = plt.figure()
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.ylabel('loss')
#     plt.xlabel('epoch')
#     st.pyplot(fig)
