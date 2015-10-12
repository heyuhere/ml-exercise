# /usr/bin/env python

"""
A Fund Raising Revenue Prediction Model

Why SVM (support vector machine)?
- good trade-off of runtime speed and prediction accuracy
- easy conversion between linear and non-linear models (kernel trick for non-linearity)
- effectiveness in high dimensional spaces
"""

from sklearn import preprocessing, decomposition, svm, metrics, linear_model
import numpy as np
import pandas as pd

###################
# data I/O
###################

def load(filename, columns=None):
  """
  load data from file

  return:
    pandas.DataFrame object that holds data
  """
  df = pd.read_csv(filename, usecols=columns)
  df.index = df['CONTROLN']
  return df


def store(data, filename='output.csv'):
  data.to_csv(filename)
  print 'data written out to %s' % filename



###################
# pre-processing
###################

class Preprocess:
  """
  Performs data cleaning, dimensionality reduction, and data scaling
  """

  def select_feature(self, data, label, threshold=0.7):
    """
    Perform feature selection by maximum information coefficient that can capture both linear and non-linear relationships.
    """
    selected = []

    from minepy import MINE
    mine = MINE()

    for i, col in enumerate(data):
      print 'feature selection: %d/%d %s' % (i, data.shape[1], col)
      mine.compute_score(data[col], label)
      if mine.mic() > threshold:
        selected.append(col)

    print '%d out of %d features were selected' % (len(selected), data.shape[1])

    return selected



  def reduce_dim(self, data):
    """
    Perform dimensionality reduction. i.e., drop variables that overlap with others. Here, PCA is used as it can be applied to either linear or non-linear relationships.

    parameter:
      data: pandas.DataFrame that contains variables

    return:
      selected/transformed variables
    """
    if not hasattr(self, 'reducer'):
      self.reducer = decomposition.KernelPCA(n_components=10, kernel='rbf')
      return self.reducer.fit_transform(data)
    else:
      return self.reducer.transform(data)

  def normalize(self, data):
    """
    Standardize training data set. 
    """
    if not hasattr(self, 'normalizer'):
      self.normalizers = {}

      for col in data:
        dtype = data[col].dtype

        # if numerical, scale
        if dtype == np.float64:
          data[col] = data[col].astype(np.float32)
          processor = preprocessing.StandardScaler()
        elif dtype == np.int64:
          data[col] = data[col].astype(np.int32)
          processor = preprocessing.StandardScaler()
        # if categorical, label
        elif dtype == np.object or dtype == 'string':
          processor = preprocessing.LabelEncoder()
        else:
          print 'unsupported data type: %s for column %s\nskipping this variable...' % (dtype, col)
          continue

        data[col] = processor.fit_transform(data[col])
        self.normalizers[col] = processor
    else:
      for col in data:
        if col not in self.normalizers:
          continue

        data[col] = self.normalizers[col].transform(data[col])

    return data



###################
# modeling
###################

class DonationProbModel:
  """
  A binary classification model that classifies whether a mailed person will make a donation or not.
  """
  def train(self, data, label):
    """
    Train model(SGDClassifier).
    Why SGDClassifier?
    - Fast
    - Low memory consumption
    - Capable of online learning

    parameter:
      data: pre-processed(cleansed, normalized) training data set
      label: dependent (Y) variable

    return:
      trained model (sklearn.linear_model.SGDClassifier instance)
    """
    self.model = linear_model.SGDClassifier(penalty='elasticnet', loss='log', n_jobs=-1)
    # self.model = svm.SVC()
    self.model = self.model.fit(data, label)
    print 'donation classification model training completed'
    return self.model


  def predict(self, data):
    """
    Predict labels with trained model

    parameter:
      model:  trained model (sklearn.linear_model.SGDClassifier)
      data:   cleansed, normalized test data set

    return:
      prediction labels in pandas.Series
    """
    prediction = self.model.predict(data)
    pred = pd.Series(prediction, index=data.index)
    print 'model prediction completed'
    return pred


class DonationAmountModel(DonationProbModel):
  """
  A regression model that predicts the donation amount.
  """
  def train(self, data, label):
    self.model = linear_model.SGDRegressor(penalty='elasticnet', loss='squared_loss')
    # self.model = svm.SVR()
    print 'donation amount model training completed'
    return self.model.fit(data, label)


###################
# evaluation
###################

def evaluate(targets, pred):
  """
  Evaluate model by 2 metrics: donation flag accuracy, and donation amount difference standard deviation

  parameter:
    targets:  pandas.DataFrame that holds actual values of TARGET_B and TARGET_D
    pred: pandas.DataFrame that holds predicted values of TARGET_B and TARGET_D

  return:
    accuracy of donation flag prediction and standard deviation of difference of donation amount between actual and predicted
  """
  accuracy = metrics.accuracy_score(targets['TARGET_B'], pred['TARGET_B'])
  std = (targets['TARGET_D'] - pred['TARGET_D']).std()
  return accuracy, std


###################
# main
###################


def get_pred(train, targets):

  # set up models
  probModel = DonationProbModel()
  amtModel = DonationAmountModel()

  # run models to predict TARGET_B and TARGET_D
  probModel = probModel.train(train, targets['TARGET_B'])
  target_b = probModel.predict(train)
  amtModel = amtModel.train(train, targets['TARGET_D'])
  target_d = probModel.predict(train)
  target_d[target_b == 0] = 0

  # tabulate prediction results
  pred = pd.DataFrame({'TARGET_B':target_b, 'TARGET_D':target_d})
  return pred, probModel, amtModel


def main(train_file='train.csv', test_file='test.csv'):
  preprocess = Preprocess()
  train = load(train_file)
  test = load(test_file)

  targets = pd.DataFrame({'TARGET_B': train.pop('TARGET_B'), 'TARGET_D': train.pop('TARGET_D')})
  print 'normalizing train data'
  train = preprocess.normalize(train.fillna(0))
  test = preprocess.normalize(test.fillna(0))

  # print 'reducing train data dimension'
  # train = preprocess.reduce_dim(train)
  # test = preprocess.reduce_dim(test)
  print 'data ready. start training...'

  pred, probModel, amtModel = get_pred(train, targets)

  acc, std = evaluate(targets, pred)
  print 'training data prediction accuracy: %.4f\tdonation amount difference standard deviation: %.4f' % (acc, std)

  
  # test = test[selected_cols]
  # test = preprocess.reduce_dim(test)
  target_b = probModel.predict(test)
  target_d = amtModel.predict(test)
  target_d[target_b == 0] = 0
  pred = pd.DataFrame({'TARGET_B':target_b, 'TARGET_D':target_d})
  store(pred)


if __name__ == '__main__':
  main()