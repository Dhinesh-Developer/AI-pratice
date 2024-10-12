from numpy import loadtxt
from keras.models import model_from_json

# Load the dataset
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")
x = dataset[:, 0:8]  # Input
y = dataset[:, 8]    # Output

# Load the model from JSON file
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()  # Read the content from the file
json_file.close()

# Load the model architecture and weights
model = model_from_json(loaded_model_json)  # Correctly load the model from JSON
model.load_weights("model.h5")
print("Loaded model from disk")

# Make predictions
predictions = model.predict(x)

# Display a few predictions
for i in range(5, 10):
    print('%s => %.2f (expected %d)' % (x[i].tolist(), predictions[i], y[i]))
