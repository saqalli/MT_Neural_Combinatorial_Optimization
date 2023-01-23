import pickle
import numpy as np
from sqlalchemy import or_

class DataLoader(object):
    def __init__(self, file_path, batch_size, problem="tsp"):
        self.file_path = file_path
        self.batch_size = batch_size
        if problem == "pdp" or problem == "pdptw":
            self.n_ranges = 16
        elif problem == "cvrptw" :
            self.n_ranges = 7
        else:
            self.n_ranges = 40
        self.problem = problem
        # self.epoch_size = 10000 * self.n_ranges

    def load_data(self, index):
        self.dataset = []
        loading_datasets = []
        couples = [[70,60],[60,10],[115,90],[30,15],[330,360],[720,350],[240,70]]
        for i in range(self.n_ranges):
            if self.problem == "pdp" or self.problem == "pdptw":
                n_nodes = 46 + 6 * i + index
            elif self.problem == "cvrptw" :
                n_nodes = 100
                n_couples = couples[i]
            else:
                n_nodes = 101 + 10 * i + index
            loading_datasets.append(n_nodes)
            with open(self.file_path + "/" + str(n_nodes) + "_" + str(n_couples[0]) + "_" + str(n_couples[1]) +".pkl", "rb") as f:
                self.dataset.append(pickle.load(f))
        print ("load datasets wtih nodes " + ", ".join([str(_) for _ in loading_datasets]))
        self.batch_index = 0
        return 0

    def next_batch(self):
        if self.problem == "tsp":
            assert self.batch_index < 125 * 40
        elif self.problem == "cvrp":
            assert self.batch_index < 30 * 40
        elif self.problem == "pdp" or  self.problem == "pdptw":
            assert self.batch_index < 60 * 16
        elif self.problem == "cvrptw":
            assert self.batch_index < 140 * 7
        dataset_index = self.batch_index % self.n_ranges
        dataset = self.dataset[dataset_index]
        n_nodes = dataset["node_feat"].shape[1]
        batch_size = 20 * 200 // n_nodes
        if self.problem == "tsp":
            assert batch_size == dataset["node_feat"].shape[0] // 125
        elif self.problem == "cvrp":
            batch_size = dataset["node_feat"].shape[0] // 30
        elif self.problem == "pdp" or self.problem == "pdptw":
            batch_size = dataset["node_feat"].shape[0] // 60
        elif self.problem == "cvrptw":
            batch_size = dataset["node_feat"].shape[0] // 140

        batch_index_inside_dataset = self.batch_index // self.n_ranges
        node_feat = dataset["node_feat"][batch_index_inside_dataset * batch_size : (batch_index_inside_dataset + 1) * batch_size] # b x 100 x 2
        edge_feat = dataset["edge_feat"][batch_index_inside_dataset * batch_size : (batch_index_inside_dataset + 1) * batch_size] # b x 1,0000 x 2
        edge_index = dataset["edge_index"][batch_index_inside_dataset * batch_size : (batch_index_inside_dataset + 1) * batch_size] # b x 100 x 10
        inverse_edge_index = dataset["inverse_edge_index"][batch_index_inside_dataset * batch_size : (batch_index_inside_dataset + 1) * batch_size] # b x 100 x 10

        if self.problem == "pdp" or self.problem == "cvrptw" or self.problem == "pdptw":
            label1 = dataset["label1"][batch_index_inside_dataset * batch_size : (batch_index_inside_dataset + 1) * batch_size] # b x 1000
            label2 = dataset["label2"][batch_index_inside_dataset * batch_size : (batch_index_inside_dataset + 1) * batch_size] # b x 1000
            self.batch_index += 1
            return (node_feat, edge_feat, label1, label2, edge_index, inverse_edge_index)
        else:
            label = dataset["label"][batch_index_inside_dataset * batch_size : (batch_index_inside_dataset + 1) * batch_size] # b x 1000
            self.batch_index += 1
            return (node_feat, edge_feat, label, edge_index, inverse_edge_index)


