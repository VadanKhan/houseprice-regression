import ydf
import pandas as pd

print("YDF version:", ydf.__version__)

# Create dummy dataset
df = pd.DataFrame({
    "feature1": [1, 2, 3, 4],
    "feature2": ["a", "b", "a", "b"],
    "label": [0, 1, 0, 1]
})

# Initialize the learner with the label column
learner = ydf.RandomForestLearner(label="label")

# Train the model
model = learner.train(df)

# Perform self-evaluation (if available)
evaluation = model.self_evaluation()
if evaluation:
    print("Self-evaluation:")
    print(evaluation)
else:
    print("No self-evaluation available.")

# Make predictions
predictions = model.predict(df)
print("Predictions:")
print(predictions)
