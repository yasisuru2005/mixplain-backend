import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Data and Model
with open('data_yamnet.json', 'r') as f:
    data = json.load(f)

X = np.array(data['embeddings'])
y = np.array(data['labels'])

model = tf.keras.models.load_model('yamnet_genre_model.keras')

with open('yamnet_mapping.json', 'r') as f:
    mapping_data = json.load(f)
genre_names = mapping_data["genres"]

# 2. Get Predictions
predictions = model.predict(X)
y_pred = np.argmax(predictions, axis=1)

# 3. Print the Metrics (For Chapter 8.3)
print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(y, y_pred, target_names=genre_names))

# 4. Generate the Confusion Matrix Graphic (For Chapter 8.3)
cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=genre_names, yticklabels=genre_names)
plt.title('Genre Classification Confusion Matrix')
plt.ylabel('True Genre')
plt.xlabel('Predicted Genre')
plt.savefig('confusion_matrix.png')
print("Saved confusion_matrix.png for your thesis!")

# 5. Generate the Multiclass ROC Curve (For Chapter 8.3)
print("\nGenerating ROC Curve...")

# Binarize the labels (One-vs-Rest format for multiclass)
n_classes = len(genre_names)
y_bin = label_binarize(y, classes=range(n_classes))

# Calculate ROC curve and ROC area for each genre
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    # predictions[:, i] contains the softmax probabilities for class i
    fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], predictions[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot all ROC curves
plt.figure(figsize=(10, 8))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve of {genre_names[i]} (AUC = {roc_auc[i]:0.2f})')

# Draw the random-guessing diagonal line
plt.plot([0, 1], [0, 1], 'k--', lw=2) 
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) per Genre')
plt.legend(loc="lower right")

plt.savefig('roc_curve.png')
print("Saved roc_curve.png for your thesis!")