from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


gesture_labels = ['like', 'up', 'palm', 'fist', 'rock', 'down','ok', 'peace'] # Replace with your actual gesture names


# Load collected data
with open("gesture_data.pkl", "rb") as f:
    data, labels = pickle.load(f)

    #
    print('Structure of raw data:', len(data), 'samples with', len(data[0]), 'features each.')



# TODO: Preprocess your data here to be hand-agnostic if not done during data collection

# Normalize the data using Min-Max Scaler to a range of [0, 1]
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data)
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data_normalized, labels, test_size=0.2, random_state=42)

# Set the parameters for cross-validation
params = {'n_neighbors': [3, 5, 7, 9, 11]}

# Initialize the KNeighborsClassifier
knn = KNeighborsClassifier()

# Use GridSearchCV to find the best number of neighbors
clf = GridSearchCV(knn, params, cv=5)
clf.fit(X_train, y_train)
print("pouzite k", clf.best_params_)

# Evaluate the classifier
y_pred = clf.predict(X_test)
report = classification_report(y_test, y_pred, target_names=gesture_labels)
# Save the trained model
with open("gesture_model.pkl", "wb") as f:
    pickle.dump({'classifier': clf, 'scaler': scaler}, f)



# Assuming y_test are your true labels and y_pred are the predictions made by your model
report = classification_report(y_test, y_pred, target_names=gesture_labels)

print(report)





