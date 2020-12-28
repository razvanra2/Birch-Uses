import numpy as np
import argparse
import csv
from sklearn.cluster import Birch
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns

def load_data(file_name):

    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        line_count = 0
        data = []

        for line in csv_reader:
            if line_count == 0:
                print(f'Column names: [{", ".join(line)}]')
            else:
                data.append(line)
            line_count += 1

    print(f'Loaded {line_count} records')
    return data

def compute_clusters(data):
    birch = Birch(
        branching_factor=50,
        n_clusters=5,
        threshold=0.3,
        copy=True,
        compute_labels=True)

    birch.fit(data)
    predictions = np.array(birch.predict(data))
    return predictions

def show_results(data: np.ndarray, labels: np.ndarray):
    labels = np.reshape(labels, (1, labels.size))
    data = np.concatenate((data, labels.T), axis=1)

    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(111)
    scatter = ax.scatter(data[:,0], data[:, 1], c=data[:, 2], s=50)
    ax.set_title("Clusters")
    ax.set_xlabel("Income")
    ax.set_ylabel("Spending")
    plt.colorbar(scatter)
    plt.show()

def show_data_corelation(data=None, csv_file_name=None):
    data_set = None
    if csv_file_name is None:
        cor = np.corrcoef(data)
        print("Corelation matrix:")
        print(cor)
    else:
        data_set = pd.read_csv(csv_file_name)
        print(data_set.describe())
        data_set = data_set[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]
        cor = data_set.corr()

    plt.figure(figsize = (10,7))
    sns.heatmap(cor, square=True)
    b, t = plt.ylim() # discover the values for bottom and top
    b += 0.5 # Add 0.5 to the bottom
    t -= 0.5 # Subtract 0.5 from the top
    plt.ylim(b, t) # update the ylim(bottom, top) values
    plt.show()
    return data_set


def main():
    data = load_data("Mall_Customers.csv")
    filtered_data = np.array([[item[3], item[4]] for item in data])

    show_data_corelation(csv_file_name="Mall_Customers.csv")

    filtered_data = np.array(filtered_data).astype(np.float64)
    labels = compute_clusters(filtered_data)
    show_results(filtered_data, labels)

if __name__ == "__main__":
    main()

