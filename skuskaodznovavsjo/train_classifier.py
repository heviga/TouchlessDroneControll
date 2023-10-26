from sklearn.neighbors import KNeighborsClassifier
import pickle

# Load collected data
with open("gesture_data.pkl", "rb") as f:
    data, labels = pickle.load(f)

clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(data, labels)

# Save the trained model
with open("gesture_model.pkl", "wb") as f:
    pickle.dump(clf, f)
