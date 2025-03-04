# titanic_data
neuronal network titanic 
Titanic challenge part 3
In this notebook, we will be covering all of the steps required to train, tune and assess a neural network.

Part 1 of this series dealt with the pre-processing and manipulation of the data. This notebook will make use of the datasets that were created in the first part.

We will do each of the following:

train and test a neural network model
use grid search to optimize the hyperparameters
submit predictions for the test set
Part 2 covered the use of a random forest for tackling this challenge. Now let's see if we can beat that model with a neural network!

NOTE: make sure to use a GPU for this notebook, as it will be significantly faster to train

Table of Contents:
1. Load packages and data
2. Pre-processing
2.1. Variable Encoding
2.2. Variable Scaling
3. Neural Network

1. Load packages and data
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
%matplotlib inline
rcParams['figure.figsize'] = 10,8
sns.set(style='whitegrid', palette='muted',
        rc={'figure.figsize': (15,10)})
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

from numpy.random import seed
from tensorflow import set_random_seed

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# print(os.listdir("../input"))
print(os.listdir("../input/titanic-cleaned-data"))
['test_clean.csv', 'train_clean.csv']
# Load data as Pandas dataframe
train = pd.read_csv('../input/titanic-cleaned-data/train_clean.csv', )
test = pd.read_csv('../input/titanic-cleaned-data/test_clean.csv')
df = pd.concat([train, test], axis=0, sort=True)
df.head()
Age	Cabin	Embarked	Family_Size	Fare	Name	Parch	PassengerId	Pclass	Sex	SibSp	Survived	Ticket	Title
0	22.0	NaN	S	1	7.2500	Braund, Mr. Owen Harris	0	1	3	male	1	0.0	A/5 21171	Mr
1	38.0	C85	C	1	71.2833	Cumings, Mrs. John Bradley (Florence Briggs Th...	0	2	1	female	1	1.0	PC 17599	Mrs
2	26.0	NaN	S	0	7.9250	Heikkinen, Miss. Laina	0	3	3	female	0	1.0	STON/O2. 3101282	Miss
3	35.0	C123	S	1	53.1000	Futrelle, Mrs. Jacques Heath (Lily May Peel)	0	4	1	female	1	1.0	113803	Mrs
4	35.0	NaN	S	0	8.0500	Allen, Mr. William Henry	0	5	3	male	0	0.0	373450	Mr
def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)

        
display_all(df.describe(include='all').T)
count	unique	top	freq	mean	std	min	25%	50%	75%	max
Age	1309	NaN	NaN	NaN	29.5624	13.1654	0.17	22	30	35.5	80
Cabin	295	186	C23 C25 C27	6	NaN	NaN	NaN	NaN	NaN	NaN	NaN
Embarked	1309	3	S	915	NaN	NaN	NaN	NaN	NaN	NaN	NaN
Family_Size	1309	NaN	NaN	NaN	0.883881	1.58364	0	0	0	1	10
Fare	1309	NaN	NaN	NaN	33.2762	51.7436	0	7.8958	14.4542	31.275	512.329
Name	1309	1307	Kelly, Mr. James	2	NaN	NaN	NaN	NaN	NaN	NaN	NaN
Parch	1309	NaN	NaN	NaN	0.385027	0.86556	0	0	0	0	9
PassengerId	1309	NaN	NaN	NaN	655	378.02	1	328	655	982	1309
Pclass	1309	NaN	NaN	NaN	2.29488	0.837836	1	2	3	3	3
Sex	1309	2	male	843	NaN	NaN	NaN	NaN	NaN	NaN	NaN
SibSp	1309	NaN	NaN	NaN	0.498854	1.04166	0	0	0	1	8
Survived	891	NaN	NaN	NaN	0.383838	0.486592	0	0	0	1	1
Ticket	1309	929	CA. 2343	11	NaN	NaN	NaN	NaN	NaN	NaN	NaN
Title	1309	6	Mr	767	NaN	NaN	NaN	NaN	NaN	NaN	NaN

2. Pre-processing

2.1. Encode Categorical Variables
We need to convert all categorical variables into numeric format. The categorical variables we will be keeping are Embarked, Sex and Title.

The Sex variable can be encoded into single 1-or-0 column, but the other variables will need to be one-hot encoded. Regular label encoding assigns some category labels higher numerical values. This implies some sort of scale (Embarked = 1 is not more than Embarked = 0 - it's just different). One Hot Encoding avoids this problem.

We will assume that there is some ordinality in the Pclass variable, so we will leave that as a single column.

sns.countplot(x='Pclass', data=df, palette='hls', hue='Survived')
plt.xticks(rotation=45)
plt.show()

sns.countplot(x='Sex', data=df, palette='hls', hue='Survived')
plt.xticks(rotation=45)
plt.show()

sns.countplot(x='Embarked', data=df, palette='hls', hue='Survived')
plt.xticks(rotation=45)
plt.show()

# convert to cateogry dtype
df['Sex'] = df['Sex'].astype('category')
# convert to category codes
df['Sex'] = df['Sex'].cat.codes
# subset all categorical variables which need to be encoded
categorical = ['Embarked', 'Title']

for var in categorical:
    df = pd.concat([df, 
                    pd.get_dummies(df[var], prefix=var)], axis=1)
    del df[var]
# drop the variables we won't be using
df.drop(['Cabin', 'Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)
df.head()
Age	Family_Size	Fare	Parch	Pclass	Sex	SibSp	Survived	Embarked_C	Embarked_Q	Embarked_S	Title_Dr	Title_Master	Title_Miss	Title_Mr	Title_Mrs	Title_Rev
0	22.0	1	7.2500	0	3	1	1	0.0	0	0	1	0	0	0	1	0	0
1	38.0	1	71.2833	0	1	0	1	1.0	1	0	0	0	0	0	0	1	0
2	26.0	0	7.9250	0	3	0	0	1.0	0	0	1	0	0	1	0	0	0
3	35.0	1	53.1000	0	1	0	1	1.0	0	0	1	0	0	0	0	1	0
4	35.0	0	8.0500	0	3	1	0	0.0	0	0	1	0	0	0	1	0	0
2.2. Scale Continuous Variables
The continuous variables need to be scaled. This is done using a standard scaler from SkLearn.

continuous = ['Age', 'Fare', 'Parch', 'Pclass', 'SibSp', 'Family_Size']

scaler = StandardScaler()

for var in continuous:
    df[var] = df[var].astype('float64')
    df[var] = scaler.fit_transform(df[var].values.reshape(-1, 1))
display_all(df.describe(include='all').T)
count	mean	std	min	25%	50%	75%	max
Age	1309.0	1.692899e-16	1.000382	-2.233408	-0.574635	0.033252	0.451175	3.832549
Family_Size	1309.0	2.593630e-16	1.000382	-0.558346	-0.558346	-0.558346	0.073352	5.758637
Fare	1309.0	-6.473474e-17	1.000382	-0.643344	-0.490691	-0.363894	-0.038690	9.261749
Parch	1309.0	-8.549311e-17	1.000382	-0.445000	-0.445000	-0.445000	-0.445000	9.956864
Pclass	1309.0	-1.399441e-16	1.000382	-1.546098	-0.352091	0.841916	0.841916	0.841916
Sex	1309.0	6.440031e-01	0.478997	0.000000	0.000000	1.000000	1.000000	1.000000
SibSp	1309.0	-6.632925e-16	1.000382	-0.479087	-0.479087	-0.479087	0.481288	7.203909
Survived	891.0	3.838384e-01	0.486592	0.000000	0.000000	0.000000	1.000000	1.000000
Embarked_C	1309.0	2.070283e-01	0.405331	0.000000	0.000000	0.000000	0.000000	1.000000
Embarked_Q	1309.0	9.396486e-02	0.291891	0.000000	0.000000	0.000000	0.000000	1.000000
Embarked_S	1309.0	6.990069e-01	0.458865	0.000000	0.000000	1.000000	1.000000	1.000000
Title_Dr	1309.0	6.111536e-03	0.077967	0.000000	0.000000	0.000000	0.000000	1.000000
Title_Master	1309.0	4.660046e-02	0.210862	0.000000	0.000000	0.000000	0.000000	1.000000
Title_Miss	1309.0	2.016807e-01	0.401408	0.000000	0.000000	0.000000	0.000000	1.000000
Title_Mr	1309.0	5.859435e-01	0.492747	0.000000	0.000000	1.000000	1.000000	1.000000
Title_Mrs	1309.0	1.535523e-01	0.360657	0.000000	0.000000	0.000000	0.000000	1.000000
Title_Rev	1309.0	6.111536e-03	0.077967	0.000000	0.000000	0.000000	0.000000	1.000000

3. Neural Network
Now, all that is left is to feed our data that has been cleaned, encoded and scaled to our neural network.

But first, we need to separate data_df back into train and test sets.

X_train = df[pd.notnull(df['Survived'])].drop(['Survived'], axis=1)
y_train = df[pd.notnull(df['Survived'])]['Survived']
X_test = df[pd.isnull(df['Survived'])].drop(['Survived'], axis=1)
3.1. Cross-validation
Keras allows us to make use of cross-validation for training our model. So we will use this to train and assess our first model.

Create neural network model
For this task, I have kept the model architecture pretty simple. We have one input layer with 17 nodes which feeds into a hidden layer with 8 nodes and an output layer which is used to predict a passenger's survival.

The output layer has a sigmoid activation function, which is used to 'squash' all our outputs to be between 0 and 1.

We are going to create a function which allows to parameterise the choice of hyperparameters in the neural network. This might seem a little overly complicated now, but it will come in super handy when we move onto tuning our parameters later.

def create_model(lyrs=[8], act='linear', opt='Adam', dr=0.0):
    
    # set random seed for reproducibility
    seed(42)
    set_random_seed(42)
    
    model = Sequential()
    
    # create first hidden layer
    model.add(Dense(lyrs[0], input_dim=X_train.shape[1], activation=act))
    
    # create additional hidden layers
    for i in range(1,len(lyrs)):
        model.add(Dense(lyrs[i], activation=act))
    
    # add dropout, default is none
    model.add(Dropout(dr))
    
    # create output layer
    model.add(Dense(1, activation='sigmoid'))  # output layer
    
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    return model
model = create_model()
print(model.summary())
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_1 (Dense)              (None, 8)                 136       
_________________________________________________________________
dropout_1 (Dropout)          (None, 8)                 0         
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 9         
=================================================================
Total params: 145
Trainable params: 145
Non-trainable params: 0
_________________________________________________________________
None
Train model
At this stage, we have our model. We have chosen a few hyperparameters such as the number of hidden layers, the number of neurons and the activation function.

The next step is to train the model on our training set. This step also requires us to choose a few more hyperparameters such as the loss function, the optimization algorithm, the number of epochs and the batch size.

# train model on full train set, with 80/20 CV split
training = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)
val_acc = np.mean(training.history['val_acc'])
print("\n%s: %.2f%%" % ('val_acc', val_acc*100))
val_acc: 86.53%
Assess results
# summarize history for accuracy
plt.plot(training.history['acc'])
plt.plot(training.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

3.2. Grid search
3.2.1. batch size and epochs
We can see from the graph above that we might be training our network for too long. Let's use grid search to find out what the optimal values for batch_size and epochs are.

# create model
model = KerasClassifier(build_fn=create_model, verbose=0)

# define the grid search parameters
batch_size = [16, 32, 64]
epochs = [50, 100]
param_grid = dict(batch_size=batch_size, epochs=epochs)

# search the grid
grid = GridSearchCV(estimator=model, 
                    param_grid=param_grid,
                    cv=3,
                    verbose=2)  # include n_jobs=-1 if you are using CPU

grid_result = grid.fit(X_train, y_train)
Fitting 3 folds for each of 6 candidates, totalling 18 fits
[CV] batch_size=16, epochs=50 ........................................
[CV] ......................... batch_size=16, epochs=50, total=   7.4s
[CV] batch_size=16, epochs=50 ........................................
[Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    7.5s remaining:    0.0s
[CV] ......................... batch_size=16, epochs=50, total=   7.2s
[CV] batch_size=16, epochs=50 ........................................
[CV] ......................... batch_size=16, epochs=50, total=   7.4s
[CV] batch_size=16, epochs=100 .......................................
[CV] ........................ batch_size=16, epochs=100, total=  13.7s
[CV] batch_size=16, epochs=100 .......................................
[CV] ........................ batch_size=16, epochs=100, total=  13.8s
[CV] batch_size=16, epochs=100 .......................................
[CV] ........................ batch_size=16, epochs=100, total=  14.0s
[CV] batch_size=32, epochs=50 ........................................
[CV] ......................... batch_size=32, epochs=50, total=   4.2s
[CV] batch_size=32, epochs=50 ........................................
[CV] ......................... batch_size=32, epochs=50, total=   4.3s
[CV] batch_size=32, epochs=50 ........................................
[CV] ......................... batch_size=32, epochs=50, total=   4.4s
[CV] batch_size=32, epochs=100 .......................................
[CV] ........................ batch_size=32, epochs=100, total=   7.6s
[CV] batch_size=32, epochs=100 .......................................
[CV] ........................ batch_size=32, epochs=100, total=   8.2s
[CV] batch_size=32, epochs=100 .......................................
[CV] ........................ batch_size=32, epochs=100, total=   7.8s
[CV] batch_size=64, epochs=50 ........................................
[CV] ......................... batch_size=64, epochs=50, total=   3.0s
[CV] batch_size=64, epochs=50 ........................................
[CV] ......................... batch_size=64, epochs=50, total=   3.0s
[CV] batch_size=64, epochs=50 ........................................
[CV] ......................... batch_size=64, epochs=50, total=   2.9s
[CV] batch_size=64, epochs=100 .......................................
[CV] ........................ batch_size=64, epochs=100, total=   4.7s
[CV] batch_size=64, epochs=100 .......................................
[CV] ........................ batch_size=64, epochs=100, total=   4.8s
[CV] batch_size=64, epochs=100 .......................................
[CV] ........................ batch_size=64, epochs=100, total=   5.2s
[Parallel(n_jobs=1)]: Done  18 out of  18 | elapsed:  2.1min finished
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
Best: 0.822671 using {'batch_size': 32, 'epochs': 50}
0.817059 (0.012992) with: {'batch_size': 16, 'epochs': 50}
0.814815 (0.014547) with: {'batch_size': 16, 'epochs': 100}
0.822671 (0.008837) with: {'batch_size': 32, 'epochs': 50}
0.813692 (0.018305) with: {'batch_size': 32, 'epochs': 100}
0.818182 (0.015307) with: {'batch_size': 64, 'epochs': 50}
0.814815 (0.014547) with: {'batch_size': 64, 'epochs': 100}
3.2.2. Optimization Algorithm
# create model
model = KerasClassifier(build_fn=create_model, epochs=50, batch_size=32, verbose=0)

# define the grid search parameters
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Nadam']
param_grid = dict(opt=optimizer)

# search the grid
grid = GridSearchCV(estimator=model, param_grid=param_grid, verbose=2)
grid_result = grid.fit(X_train, y_train)
Fitting 3 folds for each of 6 candidates, totalling 18 fits
[CV] opt=SGD .........................................................
[CV] .......................................... opt=SGD, total=   4.1s
[CV] opt=SGD .........................................................
[Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    4.1s remaining:    0.0s
[CV] .......................................... opt=SGD, total=   4.2s
[CV] opt=SGD .........................................................
[CV] .......................................... opt=SGD, total=   4.4s
[CV] opt=RMSprop .....................................................
[CV] ...................................... opt=RMSprop, total=   4.5s
[CV] opt=RMSprop .....................................................
[CV] ...................................... opt=RMSprop, total=   4.5s
[CV] opt=RMSprop .....................................................
[CV] ...................................... opt=RMSprop, total=   5.0s
[CV] opt=Adagrad .....................................................
[CV] ...................................... opt=Adagrad, total=   4.6s
[CV] opt=Adagrad .....................................................
[CV] ...................................... opt=Adagrad, total=   4.5s
[CV] opt=Adagrad .....................................................
[CV] ...................................... opt=Adagrad, total=   4.6s
[CV] opt=Adadelta ....................................................
[CV] ..................................... opt=Adadelta, total=   5.3s
[CV] opt=Adadelta ....................................................
[CV] ..................................... opt=Adadelta, total=   5.2s
[CV] opt=Adadelta ....................................................
[CV] ..................................... opt=Adadelta, total=   5.3s
[CV] opt=Adam ........................................................
[CV] ......................................... opt=Adam, total=   5.5s
[CV] opt=Adam ........................................................
[CV] ......................................... opt=Adam, total=   5.3s
[CV] opt=Adam ........................................................
[CV] ......................................... opt=Adam, total=   5.7s
[CV] opt=Nadam .......................................................
[CV] ........................................ opt=Nadam, total=   6.1s
[CV] opt=Nadam .......................................................
[CV] ........................................ opt=Nadam, total=   5.9s
[CV] opt=Nadam .......................................................
[CV] ........................................ opt=Nadam, total=   6.1s
[Parallel(n_jobs=1)]: Done  18 out of  18 | elapsed:  1.5min finished
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
Best: 0.822671 using {'opt': 'Adam'}
0.810325 (0.020997) with: {'opt': 'SGD'}
0.820426 (0.010408) with: {'opt': 'RMSprop'}
0.814815 (0.015307) with: {'opt': 'Adagrad'}
0.821549 (0.011983) with: {'opt': 'Adadelta'}
0.822671 (0.008837) with: {'opt': 'Adam'}
0.815937 (0.014108) with: {'opt': 'Nadam'}
3.2.3. Hidden neurons
seed(42)
set_random_seed(42)

# create model
model = KerasClassifier(build_fn=create_model, 
                        epochs=50, batch_size=32, verbose=0)

# define the grid search parameters
layers = [[8],[10],[10,5],[12,6],[12,8,4]]
param_grid = dict(lyrs=layers)

# search the grid
grid = GridSearchCV(estimator=model, param_grid=param_grid, verbose=2)
grid_result = grid.fit(X_train, y_train)
Fitting 3 folds for each of 5 candidates, totalling 15 fits
[CV] lyrs=[8] ........................................................
[CV] ......................................... lyrs=[8], total=   5.8s
[CV] lyrs=[8] ........................................................
[Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    5.8s remaining:    0.0s
[CV] ......................................... lyrs=[8], total=   5.9s
[CV] lyrs=[8] ........................................................
[CV] ......................................... lyrs=[8], total=   5.9s
[CV] lyrs=[10] .......................................................
[CV] ........................................ lyrs=[10], total=   5.9s
[CV] lyrs=[10] .......................................................
[CV] ........................................ lyrs=[10], total=   6.5s
[CV] lyrs=[10] .......................................................
[CV] ........................................ lyrs=[10], total=   6.1s
[CV] lyrs=[10, 5] ....................................................
[CV] ..................................... lyrs=[10, 5], total=   6.7s
[CV] lyrs=[10, 5] ....................................................
[CV] ..................................... lyrs=[10, 5], total=   6.9s
[CV] lyrs=[10, 5] ....................................................
[CV] ..................................... lyrs=[10, 5], total=   6.8s
[CV] lyrs=[12, 6] ....................................................
[CV] ..................................... lyrs=[12, 6], total=   7.1s
[CV] lyrs=[12, 6] ....................................................
[CV] ..................................... lyrs=[12, 6], total=   7.0s
[CV] lyrs=[12, 6] ....................................................
[CV] ..................................... lyrs=[12, 6], total=   7.1s
[CV] lyrs=[12, 8, 4] .................................................
[CV] .................................. lyrs=[12, 8, 4], total=   8.1s
[CV] lyrs=[12, 8, 4] .................................................
[CV] .................................. lyrs=[12, 8, 4], total=   7.8s
[CV] lyrs=[12, 8, 4] .................................................
[CV] .................................. lyrs=[12, 8, 4], total=   8.5s
[Parallel(n_jobs=1)]: Done  15 out of  15 | elapsed:  1.7min finished
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
Best: 0.822671 using {'lyrs': [8]}
0.822671 (0.008837) with: {'lyrs': [8]}
0.822671 (0.008837) with: {'lyrs': [10]}
0.820426 (0.015141) with: {'lyrs': [10, 5]}
0.822671 (0.013561) with: {'lyrs': [12, 6]}
0.818182 (0.016722) with: {'lyrs': [12, 8, 4]}
3.2.4. Dropout
# create model
model = KerasClassifier(build_fn=create_model, 
                        epochs=50, batch_size=32, verbose=0)

# define the grid search parameters
drops = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
param_grid = dict(dr=drops)
grid = GridSearchCV(estimator=model, param_grid=param_grid, verbose=2)
grid_result = grid.fit(X_train, y_train)
Fitting 3 folds for each of 6 candidates, totalling 18 fits
[CV] dr=0.0 ..........................................................
[CV] ........................................... dr=0.0, total=   7.2s
[CV] dr=0.0 ..........................................................
[Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    7.2s remaining:    0.0s
[CV] ........................................... dr=0.0, total=   7.0s
[CV] dr=0.0 ..........................................................
[CV] ........................................... dr=0.0, total=   7.3s
[CV] dr=0.01 .........................................................
[CV] .......................................... dr=0.01, total=   7.4s
[CV] dr=0.01 .........................................................
[CV] .......................................... dr=0.01, total=   7.6s
[CV] dr=0.01 .........................................................
[CV] .......................................... dr=0.01, total=   7.7s
[CV] dr=0.05 .........................................................
[CV] .......................................... dr=0.05, total=   7.5s
[CV] dr=0.05 .........................................................
[CV] .......................................... dr=0.05, total=   8.0s
[CV] dr=0.05 .........................................................
[CV] .......................................... dr=0.05, total=   7.9s
[CV] dr=0.1 ..........................................................
[CV] ........................................... dr=0.1, total=   8.5s
[CV] dr=0.1 ..........................................................
[CV] ........................................... dr=0.1, total=   8.0s
[CV] dr=0.1 ..........................................................
[CV] ........................................... dr=0.1, total=   9.0s
[CV] dr=0.2 ..........................................................
[CV] ........................................... dr=0.2, total=   8.0s
[CV] dr=0.2 ..........................................................
[CV] ........................................... dr=0.2, total=   8.4s
[CV] dr=0.2 ..........................................................
[CV] ........................................... dr=0.2, total=   8.1s
[CV] dr=0.5 ..........................................................
[CV] ........................................... dr=0.5, total=   8.6s
[CV] dr=0.5 ..........................................................
[CV] ........................................... dr=0.5, total=   8.4s
[CV] dr=0.5 ..........................................................
[CV] ........................................... dr=0.5, total=   8.7s
[Parallel(n_jobs=1)]: Done  18 out of  18 | elapsed:  2.4min finished
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
Best: 0.824916 using {'dr': 0.2}
0.822671 (0.008837) with: {'dr': 0.0}
0.822671 (0.014108) with: {'dr': 0.01}
0.823793 (0.015632) with: {'dr': 0.05}
0.823793 (0.015632) with: {'dr': 0.1}
0.824916 (0.017168) with: {'dr': 0.2}
0.819304 (0.018305) with: {'dr': 0.5}
# create final model
model = create_model(lyrs=[8], dr=0.2)

print(model.summary())
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_161 (Dense)            (None, 8)                 136       
_________________________________________________________________
dropout_75 (Dropout)         (None, 8)                 0         
_________________________________________________________________
dense_162 (Dense)            (None, 1)                 9         
=================================================================
Total params: 145
Trainable params: 145
Non-trainable params: 0
_________________________________________________________________
None
# train model on full train set, with 80/20 CV split
training = model.fit(X_train, y_train, epochs=50, batch_size=32, 
                     validation_split=0.2, verbose=0)

# evaluate the model
scores = model.evaluate(X_train, y_train)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
891/891 [==============================] - 0s 68us/step

acc: 83.16%
# summarize history for accuracy
plt.plot(training.history['acc'])
plt.plot(training.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

3.3. Make Predictions on Test Set
Finally, we can attempt to predict which passengers in the test set survived.

# calculate predictions
test['Survived'] = model.predict(X_test)
test['Survived'] = test['Survived'].apply(lambda x: round(x,0)).astype('int')
solution = test[['PassengerId', 'Survived']]
solution.head(10)
PassengerId	Survived
0	892	0
1	893	1
2	894	0
3	895	0
4	896	1
5	897	0
6	898	1
7	899	0
8	900	1
9	901	0
3.4. Output Final Predictions
