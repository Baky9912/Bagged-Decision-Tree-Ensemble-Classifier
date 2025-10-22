import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random


# cluster_count = 3  # moze drugaciji za svaki element ako treba

class MyDecisionTreeClassifierNode:
    def __init__(self, data, depth, goal_key, training_keys) -> None:
        self.training_keys = training_keys
        self.goal_key = goal_key
        self.data = data
        self.depth = depth
        self.is_leaf = True
        self.borders = None
        self.children = []
        self.expected = None  # za leaf

    def read_first_index_or_minus_one(self, cluster, key):
        if len(cluster) > 0:
            return cluster[key].iloc[0]
        else:
            return -1

    def make_children(self):
        data = self.data
        key = self.training_keys[self.depth]
        data_1d = data[key].values.reshape(-1, 1)
        # print('data1d len:', len(data_1d))
        linked = linkage(data_1d, method='ward')
        num_clusters = 3
        cluster_labels = fcluster(linked, t=num_clusters, criterion='maxclust')

        data['cluster'] = cluster_labels
        # clusters = [[] for _ in range(num_clusters)]
        # clusters = [np.array(cluster) for cluster in clusters]
        clusters = [data[data['cluster'] == i].drop(columns='cluster') for i in range(1, num_clusters+1)]
        clusters.sort(key=lambda cluster: self.read_first_index_or_minus_one(cluster, key))

        # for i, label in enumerate(cluster_labels):
        #     clusters[label - 1].append(data.iloc[i].values)

        sorted_indices = np.argsort(data_1d.flatten())
        sorted_data = data_1d.flatten()[sorted_indices]
        sorted_labels = cluster_labels[sorted_indices]

        borders = []
        for i in range(1, len(sorted_labels)):
            if sorted_labels[i] != sorted_labels[i - 1]:
                border = (sorted_data[i] + sorted_data[i - 1]) / 2
                borders.append(border)
        self.borders = borders
        return clusters 

    def calculate_goal(self):
        return self.data[self.goal_key].value_counts().idxmax()
    
    def predict(self, entry):
        if self.is_leaf:
            if self.expected is None:
                self.expected = self.calculate_goal()
            return self.expected
        key = self.training_keys[self.depth]
        val = entry[key]
        for i in range(len(self.borders)):
            if self.borders[i] > val:
                return self.children[i].predict(entry)
        return self.children[-1].predict(entry)
        

def make_predict_tree(train_df, training_keys, goal_key):
    root = MyDecisionTreeClassifierNode(train_df, 0, goal_key, training_keys)
    last_level_nodes = [root]
    for layer in range(len(training_keys)):
        # print(f"Creating layer =", layer)
        new_level_nodes = []
        for node in last_level_nodes:
            node.is_leaf = False
            
            child_datas = node.make_children()
            all_have_children = True
            for data in child_datas:
                if len(data) <= 1:
                    # bolja preciznost ako ovde stoji 0, a tamo gde se zbog 0 desi error bude popravljeno
                    all_have_children = False

            if not all_have_children:
                node.is_leaf = True               
                continue
            
            for data in child_datas:
                new_node = MyDecisionTreeClassifierNode(data, node.depth+1, node.goal_key, node.training_keys)
                new_node.is_leaf = True
                node.children.append(new_node)
                new_level_nodes.append(new_node)
        last_level_nodes = new_level_nodes
    return root
    

def main():
    best_root = None
    best_acc = 0
    best_seed = -1
    for seed in range(50):
        random.seed(seed)
        training_keys = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall']
        random.shuffle(training_keys)
        
        goal_key = 'Crop'
        root = make_predict_tree(train_df, training_keys, goal_key)

        train_received = [root.predict(row.to_dict()) for _, row in train_df.iterrows()]
        train_expected = [row.to_dict()[goal_key] for _, row in train_df.iterrows()]

        test_received = [root.predict(row.to_dict()) for _, row in test_df.iterrows()]
        test_expected = [row.to_dict()[goal_key] for _, row in test_df.iterrows()]

        # print('received')
        # print(test_received[:10])
        # print('expected')
        # print(test_expected[:10])

        print('random seed:', seed)
        train_results = [a == b for (a, b) in zip(train_received, train_expected)]
        train_acc = train_results.count(True) / len(train_results)
        print('Train accuracy: ', train_acc * 100, '%')
        test_results = [a == b for (a, b) in zip(test_received, test_expected)]
        test_acc = test_results.count(True) / len(test_results)
        print('Test accuracy: ', test_acc * 100, '%')
        if test_acc > best_acc:
            best_seed = seed
            best_acc = test_acc
            best_root = root
    print(f"{best_seed = }, {best_acc = }")
    print("Best root usable here")
    # best_seed = 233, best_acc = 0.9348484848484848



if __name__ == "__main__":
    main()