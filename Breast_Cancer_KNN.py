import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix

# Read the CSV file (place your dataset here)
df = pd.read_csv("data.csv")

# Separate features and target variable
data = df.loc[:, "radius_mean":"fractal_dimension_worst"]  # Features
target = df["diagnosis"]  # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=42, test_size=0.3)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
conf_matrix_list = []
def trainModel(k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_accuracy = confusion_matrix(y_test,y_pred)
    conf_matrix_list.append(conf_accuracy)
    return accuracy

# List to store accuracy values for different k
accuracy_values = []

for k in range(1, 21): # Hyperparameter tuning

    accuracy_values.append(trainModel(k))  # Append accuracy to the list


# Find the highest accuracy and corresponding k value
best_value = max(accuracy_values)
best_k = accuracy_values.index(best_value) + 1  # Add 1 because list index starts at 0


# Train the model again with the best k
final_accuracy = trainModel(best_k)
























