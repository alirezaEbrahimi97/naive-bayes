# load the iris dataset 
import pandas as pd
from sklearn.datasets import load_iris 
import datetime
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def add_time(df, quntize_time = 60):
    df['time'] = np.array(list(map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').time(), df['click_time'].to_numpy())))
    df['time'] = np.array(list(map(lambda x: x.hour * 3600 + x.minute * 60 + x.second, df['time'])))
    df['time'] = np.array(list(map(lambda x: x // quntize_time, df['time'])))
    return df

def set_train_and_test(X, y, size=0.2):
    return train_test_split(X, y, test_size=size)

def false_mean(pred, src):
    return len(pred[np.logical_and(pred != src, pred == 0)]), len(pred[np.logical_and(pred != src, pred == 1)])

def draw_plot(y_test, y_pred):
    plot_confusion_matrix(y_test, y_pred, classes=range(10),
                      title='Confusion matrix, without normalization')
    plot_confusion_matrix(y_test, y_pred, classes=range(10), normalize=True,
                      title='Normalized confusion matrix')
    plt.show()

def train_and_test(X, y):
    # splitting X and y into training and testing sets 
    X_train, X_test, y_train, y_test = set_train_and_test(X, y)
    #X_train_1, X_train_2, y_train_1, y_train_2 = set_train_and_test(X_train, y_train)
    # training the model on training set 
    gnb = MultinomialNB() 
    gnb.fit(X_train, y_train) 
    #gnb.partial_fit(X_train_1, y_train_1, classes=np.unique(y_train, return_counts=False))
    #gnb.partial_fit(X_train_2, y_train_2)
    # making predictions on the testing set 
    y_pred = gnb.predict(X_test) 
    # comparing actual response values (y_test) with predicted response values (y_pred)  
    acc = metrics.accuracy_score(y_test, y_pred)*100
    f_n, f_p = (false_mean(y_pred, y_test))
    return 1 - f_n / (y_test == 1).sum(), 1 - f_p /(y_test == 0).sum(), acc

def add_date_options(df):
    df['date'] = np.array(list(map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').date(), df['click_time'].to_numpy())))
    df['month'] = np.array(list(map(lambda x: x.month, df['date'])))
    df['year'] = np.array(list(map(lambda x: x.year, df['date'])))
    df['day'] = np.array(list(map(lambda x: x.day, df['date'])))
    df['day_of_week'] = np.array(list(map(lambda x: x % 7, df['day'])))
    return df

def merge_name(name1, name2): 
  return name1 + '&'+name2

def mix_columns(src, tar, data):
    data[src + "&" + tar] = data[src] * (max(data[tar].values) - min(data[tar].values)) + data[tar]
    return data

# store the feature matrix (X) and response vector (y)
df = pd.read_csv("train_sample.csv")
df = add_time(df)
df = add_date_options(df)
data = mix_columns('os','device',df)
data = mix_columns('os','app',df)
data = mix_columns('app','device',df)
df['ip'] = df['ip'] // 1000
X = df[['ip', 'app','device', 'os', 'channel', 'time', 'day_of_week', merge_name('os', 'device'), merge_name('os', 'app'), merge_name('app', 'device'), 'channel', 'app']].values
y = df['is_attributed'].values


