import pandas as pd
import matplotlib.pyplot as plt

# PART 1: Pre-processing
# Split the dataset to dependent and independent variables
dataset = pd.read_csv("creditcard.csv")
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Count the total number of observations
total_obs = y.shape[0]

# Count the total number of fraudulent observations 
fraud = [i for i in y if i == 1]
count_fraud = fraud.count(1)

# Calculate the percentage of fraud observations in the dataset
percentage = (float(count_fraud)/float(total_obs)) * 100

# Feature Min-Max Scaling, neural networks work much better with small input values
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
X_scaled = sc.fit_transform(X)

# Split the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state = 0)

# Use SMOTE to balance the dataset to a 1 fraud to 2 non-fraud ratio
# Otherwise, binary accuracy would not be a good metric
from imblearn.over_sampling import SMOTE

resampler = SMOTE(sampling_strategy=1/2)
X_resampled, y_resampled = resampler.fit_sample(X_train, y_train)

# Plot resampled vs original
def plot_data(X, y, i):
    plt.subplot(1, 2, i)
    plt.scatter(X[y == 0, 0], X[y == 0, 1], label="Class #0", alpha=0.5, linewidth=0.15)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], label="Class #1", alpha=0.5, linewidth=0.15, c='r')
    plt.legend()

plot_data(X_train, y_train, 1)
plot_data(X_resampled, y_resampled, 2)
plt.show()

# PART 2: Model creation, hyperparameter tuning
from keras.models import Sequential
from keras.layers import Dense

# Function to create model, required for KerasClassifier
def create_classifier():
    classifier = Sequential()

    # Add the input layer and a single hidden layer
    classifier.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu', input_dim = 29))

    # Add the output layer
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

    # Compile the ANN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['binary_accuracy'])

    return classifier

# We wrap the classifier with KerasClassifier to perform grid search
from keras.wrappers.scikit_learn import KerasClassifier
classifier = KerasClassifier(build_fn=create_classifier, verbose=1)

from sklearn.model_selection import GridSearchCV
# Tune batch size first
parameters = [{'batch_size' : [10, 30, 50, 70, 100], 'epochs' : [5]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'f1',
                           cv = 5,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
# Turns out the best value is 10. Larger considered values get trapped in local minima
# Unfortunately, this is also the slowest value

# Tune number of epochs
parameters = [{'batch_size' : [10], 'epochs' : [5, 10, 15] }]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'f1',
                           cv = 5,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
# Turns out the best value is 10, since 15 appears to overfit the training input

# Fit the ANN to the training set using the calculated hyperparameters
classifier = create_classifier()
classifier.fit(X_train, y_train, batch_size = 10, epochs = 10)

# PART 3 - Make the predictions and evaluate the model

# Predict the test set probabilities
probs = classifier.predict(X_test)

# Print the AUC score
from sklearn.metrics import roc_auc_score
print(f'AUC score = {roc_auc_score(y_test, probs)}')

### Calculate average precision
from sklearn.metrics import average_precision_score, precision_recall_curve
average_precision = average_precision_score(y_test, probs)

### Obtain precision and recall 
precision, recall, _ = precision_recall_curve(y_test, probs)

### Plot the recall precision tradeoff
plt.plot(precision, recall)
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.show()

# Print the classification report and confusion matrix
from sklearn.metrics import classification_report, confusion_matrix

# Decreasing the value to be compared with probs, we trade-off precision for recall
y_pred = (probs > 0.2)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

