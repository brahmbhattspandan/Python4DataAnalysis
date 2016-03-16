## Getting the data and creating the RDD

```python
import urllib
f = urllib.urlretrieve ("http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz", "kddcup.data.gz")
```

```python
data_file = "./kddcup.data.gz"
raw_data = sc.textFile(data_file)

print "Train data size is {}".format(raw_data.count())
```

## Test Data

```python
ft = urllib.urlretrieve("http://kdd.ics.uci.edu/databases/kddcup99/corrected.gz", "corrected.gz")
```

```python
test_data_file = "./corrected.gz"
test_raw_data = sc.textFile(test_data_file)

print "Test data size is {}".format(test_raw_data.count())
```

# Labeled Points


## Preparing the training data

```python
from pyspark.mllib.regression import LabeledPoint
from numpy import array

def parse_interaction(line):
    line_split = line.split(",")
    # leave_out = [1,2,3,41]
    clean_line_split = line_split[0:1]+line_split[4:41]
    attack = 1.0
    if line_split[41]=='normal.':
        attack = 0.0
    return LabeledPoint(attack, array([float(x) for x in clean_line_split]))


training_data = raw_data.map(parse_interaction)
```

## Preparing the test data

```python
test_data = test_raw_data.map(parse_interaction)
```

## Detecting network attacks using Logistic Regression

```python
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from time import time

# Build the model
t0 = time()
logit_model = LogisticRegressionWithLBFGS.train(training_data)
tt = time() - t0

print "Classifier trained in {} seconds".format(round(tt,3))
```

## Evaluating the model on new data

```python
labels_and_preds = test_data.map(lambda p: (p.label, logit_model.predict(p.features)))

t0 = time()
test_accuracy = labels_and_preds.filter(lambda (v, p): v == p).count() / float(test_data.count())
tt = time() - t0

print "Prediction made in {} seconds. Test accuracy is {}".format(round(tt,3), round(test_accuracy,4))
```

# Model selection

## Using a correlation matrix

- dst_host_same_src_port_rate, (column 35). 
- srv_serror_rate (column 25). 
- srv_rerror_rate (column 27). 
- dst_host_srv_serror_rate (column 38). 
- dst_host_srv_rerror_rate (column 40).

## Evaluating the new model

```python
def parse_interaction_corr(line):
    line_split = line.split(",")
    # leave_out = [1,2,3,25,27,35,38,40,41]
    clean_line_split = line_split[0:1]+line_split[4:25]+line_split[26:27]+line_split[28:35]+line_split[36:38]+line_split[39:40]
    attack = 1.0
    if line_split[41]=='normal.':
        attack = 0.0
    return LabeledPoint(attack, array([float(x) for x in clean_line_split]))

corr_reduced_training_data = raw_data.map(parse_interaction_corr)
corr_reduced_test_data = test_raw_data.map(parse_interaction_corr)
```

## Train the model

```python
t0 = time()
logit_model_2 = LogisticRegressionWithLBFGS.train(corr_reduced_training_data)
tt = time() - t0

print "Classifier trained in {} seconds".format(round(tt,3))
```

## Evaluate its accuracy

```python
labels_and_preds = corr_reduced_test_data.map(lambda p: (p.label, logit_model_2.predict(p.features)))
t0 = time()
test_accuracy = labels_and_preds.filter(lambda (v, p): v == p).count() / float(corr_reduced_test_data.count())
tt = time() - t0

print "Prediction made in {} seconds. Test accuracy is {}".format(round(tt,3), round(test_accuracy,4))
```

## Using hypothesis testing

## Pearson's chi-squared ( χ2) tests

```python
feature_names = ["land","wrong_fragment",
             "urgent","hot","num_failed_logins",
             "logged_in","num_compromised",
             "root_shell","su_attempted",
             "num_root","num_file_creations",
             "num_shells","num_access_files",
            "num_outbound_cmds",
             "is_hot_login","is_guest_login",
             "count","srv_count","serror_rate",
             "srv_serror_rate","rerror_rate",
             "srv_rerror_rate","same_srv_rate",
             "diff_srv_rate","srv_diff_host_rate",
             "dst_host_count","dst_host_srv_count",
             "dst_host_same_srv_rate","dst_host_diff_srv_rate",
             "dst_host_same_src_port_rate",
             "dst_host_srv_diff_host_rate","dst_host_serror_rate",
             "dst_host_srv_serror_rate",
             "dst_host_rerror_rate","dst_host_srv_rerror_rate"]


def parse_interaction_categorical(line):
    line_split = line.split(",")
    clean_line_split = line_split[6:41]
    attack = 1.0
    if line_split[41]=='normal.':
        attack = 0.0
    return LabeledPoint(attack, 
        array([float(x) for x in clean_line_split]))

training_data_categorical = raw_data.map(parse_interaction_categorical)


from pyspark.mllib.stat import Statistics

chi = Statistics.chiSqTest(training_data_categorical)


import pandas as pd
pd.set_option('display.max_colwidth', 30)

records = [(result.statistic, result.pValue) for result in chi]

chi_df = pd.DataFrame(data=records, index= feature_names, columns=["Statistic","p-value"])

chi_df 
```

## Evaluating the new model


- land and num_outbound_cmds could be removed from our model


## Evaluating the new model

```python
def parse_interaction_chi(line):
    line_split = line.split(",")
    # leave_out = [1,2,3,6,19,41]
    clean_line_split = line_split[0:1] + line_split[4:6] + line_split[7:19] + line_split[20:41]
    attack = 1.0
    if line_split[41]=='normal.':
        attack = 0.0
    return LabeledPoint(attack, array([float(x) for x in clean_line_split]))

training_data_chi = raw_data.map(parse_interaction_chi)
test_data_chi = test_raw_data.map(parse_interaction_chi)
```

## Train the Model 

```python
t0 = time()
logit_model_chi = LogisticRegressionWithLBFGS.train(training_data_chi)
tt = time() - t0

print "Classifier trained in {} seconds".format(round(tt,3))
```

## Evaluate the model

```python
labels_and_preds = test_data_chi.map(lambda p: (p.label, logit_model_chi.predict(p.features)))
t0 = time()
test_accuracy = labels_and_preds.filter(lambda (v, p): v == p).count() / float(test_data_chi.count())
tt = time() - t0

print "Prediction made in {} seconds. Test accuracy is {}".format(round(tt,3), round(test_accuracy,4))
```




> The reference book is Learning Spark by Holden Karau, Andy Konwinski, Patrick Wendell, and Matei Zaharia.





























