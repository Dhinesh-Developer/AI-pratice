from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

# Load the dataset
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")
x = dataset[:, 0:8]  # Input
y = dataset[:, 8]    # Output

# Define the model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Corrected this line

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(x, y, epochs=10, batch_size=10)

# Evaluate the model
_, accuracy = model.evaluate(x, y)
print('Accuracy: %.2f' % (accuracy * 100))

# Save the model
model_json = model.to_json()  # Fixed this variable name
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("Saved model to disk")
