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
        num_veh = instance[6]
        n_nodes = x.shape[0]

        f.write("NAME : " + instance_name + "\n")
        f.write("TYPE : CVRPTW\n")
        f.write("DIMENSION : " + str(n_nodes) + "\n")
        f.write("VEHICLES : " + str(num_veh) + "\n")
        f.write("CAPACITY : " + str(capacity) + "\n")
        f.write("SERVICE_TIME : " + str(service_time) + "\n" )
        f.write("EDGE_WEIGHT_TYPE : EXACT_2D\n")
        f.write("NODE_COORD_SECTION\n")
        for l in range(n_nodes):
            f.write(" "+str(l+1)+" "+str(x[l][0]*100)[:15]+" "+str(x[l][1]*100)[:15]+"\n")
        f.write("DEMAND_SECTION\n")
        f.write("1 0\n")
        for l in range(n_nodes - 1):
            f.write(str(l + 2) + " " + str(int(demand[l]))+"\n")
        f.write("TIME_WINDOW_SECTION\n")
        f.write("1 0 1000\n")
        for l in range(n_nodes - 1):
            f.write(str(l + 2) + " " + str(int(a[l])) + " " + str(int(b[l])) + "\n")
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
    instance_filename = "tmp/" + dataset_name + "/sol_cvrptw/" + instance_name + ".cvrptw"
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

def generate_dataset(n_samples, n_nodes, save_dir,mu,sig):
    num_veh = 9
    capacity = 200
    service_time = 10
    loc = np.random.uniform(size=(n_samples, n_nodes + 1, 2))
    demand = np.random.normal(15, 10, (n_samples, n_nodes)).astype("int")
    demand = np.maximum(np.minimum(np.ceil(np.abs(demand)), 42), 1)
    dist = np.sqrt(((loc[:, 0:1] - loc[:, 1:]) ** 2).sum(-1)) * 100
    a_sample = np.floor(dist) + 1
    b_sample = 1000 - a_sample - 10
    a = np.random.uniform(0,0.8,size=(n_samples, n_nodes))
    a = (a * (b_sample - a_sample)).astype("int")
    eps = np.maximum(np.abs(np.random.normal(mu,sig, (n_samples, n_nodes))), 10)
    b = np.minimum(np.ceil(a + eps), b_sample)
    #a = a / 100
    #b = b / 100
    dataset = {"loc":loc,
               "demand":demand,
               "start":a,
               "end":b,
               "capacity":capacity,
               "service_time":service_time,
               "num_veh":num_veh}
    if save_dir == "CVRPTW_test":
        with open(save_dir + "/cvrptw_" + str(n_nodes) + ".pkl", "wb") as f:
            pickle.dump(dataset, f)
        return
    data = [[dataset["loc"][i], dataset["demand"][i], dataset["capacity"], dataset["start"][i], dataset["end"][i], dataset["service_time"], dataset["num_veh"]] for i in range(n_samples)]
    n_neighbours = 20
    os.makedirs("tmp/" + str(n_nodes) + "_" + str(mu)+ "_" + str(sig) + "/sol_cvrptw", exist_ok=True)
    os.makedirs("tmp/" + str(n_nodes) + "_" + str(mu)+ "_" + str(sig) + "/LKH_para", exist_ok=True) 
    os.makedirs("tmp/" + str(n_nodes) + "_" + str(mu)+ "_" + str(sig) + "/LKH_log", exist_ok=True)

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
        result = list(tqdm.tqdm(pool.imap(method_wrapper, [("LKH", str(n_nodes) + "_" + str(mu)+ "_" + str(sig), data[i], str(i), True, 10000) for i in range(len(data))]), total=len(data)))
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
    with open(save_dir + "/" + str(n_nodes) + "_" + str(mu) + "_" + str(sig) + ".pkl", "wb") as f:
        pickle.dump(feat, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-train", action='store_true', help="Generate training and validation datasets")
    parser.add_argument("-test", action='store_true', help="Generate test datasets")
    args = parser.parse_args()
    if args.train:
        os.makedirs("sol_CVRPTW_train", exist_ok=True)
        os.makedirs("sol_CVRPTW_val", exist_ok=True)
        cpl = [[70,60],[60,10],[115,90],[30,15],[330,360],[720,350],[240,70]]
        n_nodes = 100
        n_samples = 5000
        for mu,sig in cpl:
            generate_dataset(n_samples, n_nodes, "sol_CVRPTW_train", mu, sig) 
        for n_nodes in [100]:
            n_samples = 100
            generate_dataset(n_samples, n_nodes, "sol_CVRPTW_val", 240, 70)
    if args.test:
        os.makedirs("CVRPTW_test", exist_ok=True)
        n_samples = 1000
        for n_nodes in [40, 200, 300, 500, 1000, 2000]:
            np.random.seed(1234)
            generate_dataset(n_samples, n_nodes, "CVRPTW_test")
