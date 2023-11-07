import pickle

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# Load the data
with open('gesture_data.pkl', 'rb') as f:
    data, labels = pickle.load(f)

scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data)

first_normalized_gesture = data_normalized[0]
# Extract the first gesture
first_gesture = data[0]

print(first_gesture)
print("dlzka preveho gesta", len(first_gesture))
# Calculate the number of landmarks
# Assuming that the number of coordinates is divisible by 3 (x, y, and z)
num_landmarks = len(first_gesture) // 3

# Unpack the flat list into x, y, and z coordinates
x_coords = first_gesture[:num_landmarks]
y_coords = first_gesture[num_landmarks:2*num_landmarks]
z_coords = first_gesture[2*num_landmarks:]

# Print the landmarks
print("Structured landmarks of the first gesture:")
for idx in range(num_landmarks):
    print(f"Landmark {idx}: x={x_coords[idx]}, y={y_coords[idx]}, z={z_coords[idx]}")

