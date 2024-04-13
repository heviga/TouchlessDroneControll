import pickle
import argparse
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import os

def get_args():
    parser = argparse.ArgumentParser(description="Train a KNN model for face recognition.")
    parser.add_argument('--data_file', type=str, default='face_data.pkl', help='Path to the pickle file containing the data.')
    parser.add_argument('--model_file', type=str, default='knn_model.pkl', help='Path to save the trained KNN model.')
    parser.add_argument('--test_size', type=float, default=0.2, help='Fraction of the dataset to be used as test set.')
    parser.add_argument('--n_neighbors', type=int, default=3, help='Number of neighbors to use for KNN.')
    return parser.parse_args()

def train_model(args):
    # Check if the data file exists
    if not os.path.exists(args.data_file):
        print(f"Data file {args.data_file} not found.")
        return

    # Load data
    with open(args.data_file, 'rb') as f:
        data, labels = pickle.load(f)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=args.test_size, random_state=42)

    # Train KNN classifier
    model = KNeighborsClassifier(n_neighbors=args.n_neighbors)
    model.fit(X_train, y_train)

    # Evaluate the model
    predictions = model.predict(X_test)
    print("Model accuracy:", model.score(X_test, y_test))
    print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
    print("Classification Report:\n", classification_report(y_test, predictions))

    # Save the model
    with open(args.model_file, 'wb') as f:
        pickle.dump(model, f)

if __name__ == '__main__':
    args = get_args()
    train_model(args)
