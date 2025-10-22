from typing import Counter
import pandas as pd
import random
from z1.a import MyDecisionTreeClassifierNode, make_predict_tree
from sklearn.model_selection import train_test_split

class AnsamblModel:
    def __init__(self, cnt, min_prec) -> None:
        self.cnt = cnt
        self.min_acc = min_prec
        self.models = []
        self.make()
    
    def make(self):
        file_path = 'crop.csv'
        df_crop = pd.read_csv(file_path)
        train_df, test_df = train_test_split(df_crop, test_size=0.3, random_state=2)
        for seed in range(200):
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
            if test_acc > self.min_acc:
                self.models.append(root)
                print(f"Added model #{len(self.models)}")
            else:
                print("Didn't add model")
            if len(self.models) == self.cnt:
                print("Finished making the ansambl")
                return
        print("Made partial ansambl after many failed models")
    
    def predict(self, entry):
        predictions = [model.predict(entry) for model in self.models]
        counter = Counter(predictions)
        print(f"{predictions = }")
        most_common_element, _ = counter.most_common(1)[0]
        print(f"{most_common_element = }")
        # can calc confidence interval
        return most_common_element


def main():
    file_path = 'crop.csv'
    df_crop = pd.read_csv(file_path)
    train_df, test_df = train_test_split(df_crop, test_size=0.3, random_state=2)
    goal_key = 'Crop'
    train_expected = [row.to_dict()[goal_key] for _, row in train_df.iterrows()]
    test_expected = [row.to_dict()[goal_key] for _, row in test_df.iterrows()]

    # print(train_expected[:20])
    # print(test_expected[:20])
    # exit()
    model = AnsamblModel(20, .86)
    

    train_received = [model.predict(row.to_dict()) for _, row in train_df.iterrows()]
    test_received = [model.predict(row.to_dict()) for _, row in test_df.iterrows()]

    received_vs_expected = list(zip(train_received, train_expected))
    #print(received_vs_expected)
    train_results = [a == b for (a, b) in received_vs_expected]
    train_acc = train_results.count(True) / len(train_results)
    print('Train data accuracy: ', train_acc * 100, '%')

    received_vs_expected = list(zip(test_received, test_expected))
    #print(received_vs_expected)
    test_results = [a == b for (a, b) in received_vs_expected]
    test_acc = test_results.count(True) / len(test_results)
    print('Test accuracy: ', test_acc * 100, '%')


if __name__ == "__main__":
    main()