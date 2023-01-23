import os
from subprocess import check_call
from multiprocessing import Pool
import tqdm
import numpy as np
import pickle
import torch
from torch.autograd import Variable
from tqdm import trange
import argparse
import time
import tempfile

def write_instance(instance, instance_name, instance_filename):
    with open(instance_filename, "w") as f:
        x = instance[0]
        demand = instance[1]
        capacity = instance[2]
        a = instance[3]
        b = instance[4]
        service_time = instance[5]
        n_nodes = x.shape[0]
        
        f.write("NAME : " + instance_name + "\n")
        f.write("COMMENT : blank\n")
        f.write("TYPE : PDPTW\n")
        f.write("VEHICLES : 20\n")
        f.write("CAPACITY : " + str(capacity) + "\n")
        f.write("DIMENSION : " + str(n_nodes) + "\nEDGE_WEIGHT_TYPE : EUC_2D\n")
        f.write("NODE_COORD_SECTION\n")
        for l in range(n_nodes):
            f.write(" "+str(l+1)+" "+str(x[l][0]*1000000)[:15]+" "+str(x[l][1]*1000000)[:15]+"\n")
        f.write("PICKUP_AND_DELIVERY_SECTION\n")
        f.write("1 0 0 10000000 0 0 0\n")
        for i in range(n_nodes - 1):
            if i < n_nodes// 2:
                f.write(str(i+2)+ " " + str(int(demand[i]))+ " " + str(float(a[i] * 1000000)) + " " + str(float(b[i] * 1000000)) +
                 " " + str(service_time * 1000000) + " 0 "+str(i+2+(n_nodes )//2) + "\n")
            else:
                f.write(str(i+2)+ " " + str(int(-demand[i-n_nodes//2]))+ " " + str(float(a[i] * 1000000)) + " " + str(float(b[i] * 1000000)) +
                 " " + str(service_time * 1000000) + " " + str(i+2-n_nodes//2) + " 0\n")
        f.write("DEPOT_SECTION\n 1\n -1\n")
        f.write("EOF\n")    

def write_para(dataset_name, instance_name, instance_filename, method, para_filename, max_trials=1000, seed=1234):
    with open(para_filename, "w") as f:
        f.write("PROBLEM_FILE = " + instance_filename + "\n")
        f.write("MAX_TRIALS = " + str(max_trials) + "\n")
        f.write("SPECIAL\nRUNS = 1\n")
        f.write("SEED = " + str(seed) + "\n")

def method_wrapper(args):
    if args[0] == "LKH":
        return solve_LKH(*args[1:])
    elif args[0] == "FeatGen":
        return generate_feat(*args[1:])

def solve_LKH(dataset_name, instance, instance_name, rerun=False, max_trials=1000):
    para_filename = "tmp/" + dataset_name + "/LKH_para/" + instance_name + ".para"
    log_filename = "tmp/" + dataset_name + "/LKH_log/" + instance_name + ".log"
    instance_filename = "tmp/" + dataset_name + "/pdptw/" + instance_name + ".pdptw"
    if rerun or not os.path.isfile(log_filename):
        write_instance(instance, instance_name, instance_filename)
        write_para(dataset_name, instance_name, instance_filename, "LKH", para_filename, max_trials=max_trials)
        with open(log_filename, "w") as f:
            check_call(["./LKH", para_filename], stdout=f)
    return read_results(log_filename, max_trials)

def read_results(log_filename, max_trials):
    with open(log_filename, "r") as f:
        line = f.readlines()[-1]
        result = [int(_line) - 1 for _line in line.split(" ")[:-2]]
    return result

def generate_dataset(n_samples, n_nodes, save_dir):
    capacity = 1000
    service_time = 0.1
    loc = np.random.uniform(size=(n_samples, n_nodes + 1, 2))
    demand = np.random.normal(15, 10, (n_samples, n_nodes)).astype("int")
    demand = np.maximum(np.minimum(np.ceil(np.abs(demand)), 42), 1)
    dist = np.sqrt(((loc[:, 0:1] - loc[:, 1:]) ** 2).sum(-1)) * 100
    a_sample = np.floor(dist) + 1
    b_sample = 1000 - a_sample - 10
    a = np.random.uniform(size=(n_samples, n_nodes))
    a = (a * (b_sample - a_sample) + a_sample).astype("int")

    eps = np.maximum(np.abs(np.random.normal(0, 1, (n_samples, n_nodes))), 0.01)
    b = np.minimum(np.ceil(a + 300 * eps), b_sample)
    a = a / 100
    b = b / 100
    dataset = {"loc":loc,
               "demand":demand,
               "start":a,
               "end":b,
               "capacity":capacity,
               "service_time":service_time}
    if save_dir == "PDPTW_test":
        with open(save_dir + "/pdptw_" + str(n_nodes) + ".pkl", "wb") as f:
            pickle.dump(dataset, f)
        return
    data = [[dataset["loc"][i], dataset["demand"][i], dataset["capacity"], dataset["start"][i], dataset["end"][i], dataset["service_time"]] for i in range(n_samples)]
    n_neighbours = 20
    os.makedirs("tmp/" + str(n_nodes) + "/pdptw", exist_ok=True)
    os.makedirs("tmp/" + str(n_nodes) + "/LKH_para", exist_ok=True) 
    os.makedirs("tmp/" + str(n_nodes) + "/LKH_log", exist_ok=True)

    demand = np.concatenate([np.zeros((n_samples, 1)), dataset['demand'] / 50], -1)
    start = np.concatenate([np.zeros((n_samples, 1)), dataset['start'] / 10], -1)
    end = np.concatenate([np.ones((n_samples, 1)), dataset['end'] / 10], -1)

    capacity = np.concatenate([np.ones((n_samples, 1)), np.zeros((n_samples, n_nodes))], -1)
    x = dataset['loc']
    node_feat = np.concatenate([x,
                                demand.reshape(n_samples, n_nodes + 1, 1),
                                start.reshape(n_samples, n_nodes + 1, 1),
                                end.reshape(n_samples, n_nodes + 1, 1),
                                capacity.reshape(n_samples, n_nodes + 1, 1)], -1)

    dist = x.reshape(n_samples, n_nodes + 1, 1, 2) - x.reshape(n_samples, 1, n_nodes + 1, 2)
    dist = np.sqrt((dist ** 2).sum(-1)) # 10000 x 100 x 100
    edge_index = np.argsort(dist, -1)[:, :, 1:1 + n_neighbours]
    edge_feat = dist[np.arange(n_samples).reshape(-1, 1, 1), np.arange(n_nodes + 1).reshape(1, -1, 1), edge_index]

    inverse_edge_index = -np.ones(shape=[n_samples, n_nodes + 1, n_nodes + 1], dtype="int")
    inverse_edge_index[np.arange(n_samples).reshape(-1, 1, 1), edge_index, np.arange(n_nodes + 1).reshape(1, -1, 1)] = np.arange(n_neighbours).reshape(1, 1, -1) + np.arange(n_nodes + 1).reshape(1, -1, 1) * n_neighbours
    inverse_edge_index = inverse_edge_index[np.arange(n_samples).reshape(-1, 1, 1), np.arange(n_nodes + 1).reshape(1, -1, 1), edge_index]
    with Pool(os.cpu_count()) as pool:
        result = list(tqdm.tqdm(pool.imap(method_wrapper, [("LKH", str(n_nodes), data[i], str(i), True, 10000) for i in range(len(data))]), total=len(data)))
    result = np.array(result) # n_samples x n_nodes
    result[result > n_nodes] = 0

    label1 = np.zeros(shape=[n_samples, n_nodes + 1, n_nodes + 1], dtype="bool")
    label2 = np.zeros(shape=[n_samples, n_nodes + 1, n_nodes + 1], dtype="bool")
    label1[np.arange(n_samples).reshape(-1, 1), result, np.roll(result, 1, -1)] = True
    label2[np.arange(n_samples).reshape(-1, 1), np.roll(result, 1, -1), result] = True
    label1 = label1[np.arange(n_samples).reshape(-1, 1, 1), np.arange(n_nodes + 1).reshape(1, -1, 1), edge_index]
    label2 = label2[np.arange(n_samples).reshape(-1, 1, 1), np.arange(n_nodes + 1).reshape(1, -1, 1), edge_index]

    feat = {"node_feat":node_feat,
            "edge_feat":edge_feat,
            "edge_index":edge_index,
            "inverse_edge_index":inverse_edge_index,
            "label1":label1,
            "label2":label2}
    with open(save_dir + "/" + str(n_nodes) + ".pkl", "wb") as f:
        pickle.dump(feat, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-train", action='store_true', help="Generate training and validation datasets")
    parser.add_argument("-test", action='store_true', help="Generate test datasets")
    args = parser.parse_args()
    if args.train:
        os.makedirs("PDPTW_train", exist_ok=True)
        os.makedirs("PDPTW_val", exist_ok=True)
        for n_nodes in range(184, 203, 6):
            n_samples = 2 * 120000 // n_nodes
            generate_dataset(n_samples, n_nodes, "PDPTW_train") 
        for n_nodes in [40, 100, 200]:
            n_samples = 1000
            generate_dataset(n_samples, n_nodes, "PDPTW_val")
    if args.test:
        os.makedirs("PDPTW_test", exist_ok=True)
        n_samples = 1000
        for n_nodes in [40, 100, 200, 300, 400, 500, 1000]:
            np.random.seed(1234)
            generate_dataset(n_samples, n_nodes, "PDPTW_test")











