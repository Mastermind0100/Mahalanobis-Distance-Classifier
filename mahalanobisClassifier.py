import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision.transforms import ToTensor
from collections import defaultdict
from alive_progress import alive_it
from sklearn.decomposition import PCA
from scipy.spatial.distance import mahalanobis

training_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

testing_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

labels_map = {
    0: "Airplane",
    1: "Automobile",
    2: "Bird",
    3: "Cat",
    4: "Deer",
    5: "Dog",
    6: "Frog",
    7: "Horse",
    8: "Ship",
    9: "Truck",
}

img, _ = training_data[0]
r = len(img[0])
c = len(img[0][0])

size_of_training_data = len(training_data)
batch_size_of_training_data = size_of_training_data/10   #Specific to this dataset
img_pixels_one_channel = r*c

pca = PCA(2)

class_separated_stats = defaultdict(lambda: {
    "mean": np.zeros(3072),
    "covariance_matrix": []
})

img_list = []
labels_list = []
custom_format_dataset = {}

print("Customizing Dataset... ")
bar = alive_it(training_data)
for data in bar:
    img, label_id = data
    if label_id not in custom_format_dataset:
        custom_format_dataset[label_id] = []
    custom_format_dataset[label_id].append(img.numpy().ravel())
    img_list.append(img.numpy().ravel())
    labels_list.append(label_id)
print("Done! \u2713\n")
print(len(custom_format_dataset[1]), " and ", len(custom_format_dataset[1][0]))

print("Applying PCA/LDA...")

for key, data in custom_format_dataset.items():
    key_list = list(range(5000))
    custom_format_dataset[key] = lda.fit_transform(data, key_list)

print("Done! \u2713")
print("New Shape: ", len(custom_format_dataset[1]), " and ", len(custom_format_dataset[1][0]))

print("Calculating Means for Training Data...\n")
bar = alive_it(custom_format_dataset.items())
for label_id, img_data in bar:
    class_separated_stats[labels_map[label_id]]["mean"] = np.mean(np.array(img_data), axis=0)
    class_separated_stats[labels_map[label_id]]["covariance_matrix"] = np.cov(img_data, rowvar=False)
print("\nMeans have been calculated \u2713\n\n")

print("Test starts!")

custom_format_testing_data = {}
testing_outputs = []
fin_accuracy = 0
iterator = 0
bar = alive_it(testing_data)
for data in bar:
    distances = []
    img, label_id = data
    img_data = pca.transform([img.numpy().ravel()])[0]
    if iterator == 0:
        print("Test dimensions! ", len(img_data))
    for i in range(10):
        mahalanobis_distance = mahalanobis(img_data, class_separated_stats[labels_map[i]]["mean"], np.linalg.inv(class_separated_stats[labels_map[i]]["covariance_matrix"]))
        distances.append(mahalanobis_distance)
    pred = np.argmin(distances)
    if pred == label_id:
        testing_outputs.append(0)
    else:
        testing_outputs.append(1)
    iterator += 1
    fin_accuracy = (sum(testing_outputs)/len(testing_outputs))*100
    bar.text = f'Accuracy --> {fin_accuracy}%'

print("Final Accuracy: ", fin_accuracy)



