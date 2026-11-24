
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

csv_path = os.path.join(ROOT, "results", "mae_confusion_matrix.csv")
output_path = os.path.join(ROOT, "results", "mae_confusion_matrix.png")

with open(csv_path, "r") as file:
    rows = list(csv.reader(file))

class_names = rows[0][1:]
matrix = np.array(
    [[int(value) for value in row[1:]] for row in rows[1:]]
)

plt.figure(figsize=(12, 10))

plt.imshow(matrix, cmap="Blues")
plt.colorbar(label="Number of Images")

plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
plt.yticks(range(len(class_names)), class_names)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("MAE Few-Label Classification Confusion Matrix")

for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        plt.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=8)

plt.tight_layout()
plt.savefig(output_path, dpi=200)
plt.close()

print("Confusion matrix saved:", output_path)
