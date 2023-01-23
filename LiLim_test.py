import os
from subprocess import check_call
from multiprocessing import Pool
import tqdm
import numpy as np
import pickle
from net.sgcn_model import SparseGCNModel
import torch
from torch.autograd import Variable
from tqdm import trange
import argparse
import time
import tempfile

def write_para(instance_name, method, para_filename, opt_value, max_trials=1000):
    with open(para_filename, "w") as f:
        f.write("PROBLEM_FILE = lilim_data/" + instance_name + ".pdptw\n")
        f.write("MAX_TRIALS = " + str(max_trials) + "\n")
        f.write("SPECIAL\nRUNS = 1\n")
        f.write("OPTIMUM = " + str(opt_value) + "\n")
        if method == "NeuroLKH":
            f.write("SEED = 1234\n")
            f.write("CANDIDATE_FILE = result/lilim_pdptw/candidate/" + instance_name + ".txt\n")
        elif method == "FeatGenerate":
            f.write("GerenatingFeature\n")
            f.write("Feat_FILE = result/" + dataset_name + "/feat/" + instance_name + ".txt\n")
        else:
            assert method == "LKH"

def write_candidate(dataset_name, instance_name, candidate1, candidate2):
    n_node = candidate1.shape[0] - 1
    candidate1 = candidate1.astype("int")
    candidate2 = candidate2.astype("int")

    with open("result/" + dataset_name + "/candidate/" + instance_name + ".txt", "w") as f:
        f.write(str((n_node + 20) * 2) + "\n")
        line = "1 0 5 " + str(1 + n_node + 20) + " 0"
        for _ in range(4):
            line += " " + str(2 * n_node + 2 * 20 - _) + " 1"
        f.write(line + "\n")
        for j in range(1, n_node + 1):
            line = str(j + 1) + " 0 5 " + str(j + 1 + n_node + 20) + " 1"
            for _ in range(4):
                line += " " + str(candidate2[j, _] + 1 + n_node + 20) + " 1"
            f.write(line + "\n")
        for j in range(19):
            line = str(n_node + 1 + 1 + j) + " 0 5 " + str(n_node + 1 + 1 + j + n_node + 20) + " 0 " + str(1 + n_node + 20) + " 1"
            for _ in range(3):
                line += " " + str(n_node + 2 + _ + n_node + 20) + " 1" 
            f.write(line + "\n")
        
        line = str(1 + n_node + 20) + " 0 5 1 0"
        for _ in range(4):
            line += " " + str( n_node + 20 - _) + " 1"
        f.write(line + "\n")
        for j in range(1, n_node + 1):
            line = str(j + 1 + n_node + 20) + " 0 5 " + str(j + 1) + " 1"
            for _ in range(4):
                line += " " + str(candidate1[j, _] + 1) + " 1"
            f.write(line + "\n")
        for j in range(19):
            line = str(n_node + 2 + j + n_node + 20) + " 0 5 " + str(n_node + 2 + j) + " 0"
            for _ in range(4):
                line += " " + str(n_node + 20 - _) + " 1"
            f.write(line + "\n")
        f.write("-1\nEOF\n")

def method_wrapper(args):
    if args[0] == "LKH":
        return solve_LKH(*args[1:])
    elif args[0] == "NeuroLKH":
        return solve_NeuroLKH(*args[1:])
    elif args[0] == "FeatGen":
        return generate_feat(*args[1:])

def solve_LKH(instance_name, opt_value, rerun=False, max_trials=1000):
    para_filename = "result/lilim_pdptw/LKH_para/" + instance_name + ".para"
    log_filename = "result/lilim_pdptw/LKH_log/" + instance_name + ".log"
    if rerun or not os.path.isfile(log_filename):
        write_para(instance_name, "LKH", para_filename, opt_value, max_trials=max_trials)
        with open(log_filename, "w") as f:
            check_call(["./LKH", para_filename], stdout=f)
    return read_results(log_filename,max_trials)

def infer_SGN_write_candidate(net, instance_names):
    for instance_name in instance_names:
        with open("lilim_data/" + instance_name + ".pdptw", "r") as f:
            lines = f.readlines()
            assert lines[5] == "EDGE_WEIGHT_TYPE : EXACT_2D\n"
            assert lines[6] == "NODE_COORD_SECTION\n"
            n_nodes = int(lines[2].split(" ")[-1]) 
            num_vehicles = int(lines[3].split(" ")[-1])
            max_nodes = int(n_nodes + num_vehicles - 1)
            x = []
            for i in range(n_nodes):
                line = [float(_) for _ in lines[7 + i].strip().split()]
                assert len(line) == 3
                assert line[0] == i + 1
                x.append([line[1], line[2]])
                
            x = np.array(x)
            
            scale = max(x[:, 0].max() - x[:, 0].min(), x[:, 1].max() -x[:, 1].min()) * (1 + 2 * 1e-4)
            x = x - x.min(0).reshape(1, 2)
            x = x / scale
            x = x + 1e-4
            
            if x[:, 0].max() > x[:, 1].max():
                x[:, 1] += (1 - 1e-4 - x[:, 1].max()) / 2
            else:
                x[:, 0] += (1 - 1e-4 - x[:, 0].max()) / 2

            x = x.reshape(1, n_nodes, 2)
            
            
            demand = np.zeros([1,n_nodes])
            start = []
            end = []
            for i in range(n_nodes):
                line = [int(_) for _ in lines[8 + n_nodes + i].strip().split()]
                assert len(line) == 7
                assert line[0] == i + 1
                demand[:,i] = line[1]
                start.append([line[2]])
                end.append(line[3])
            demand = np.array(demand).astype(int)
            start = np.array(start).astype(int)
            start = start.reshape(1,n_nodes)
            end = np.array(end).astype(int)
            end = end.reshape(1,n_nodes)
            
            demand = demand / 50
            start = start / 10
            end = end / 10 
            
            capacity = np.concatenate([np.ones((1, 1)), np.zeros((1, n_nodes-1))], -1)
                  
        n_edges = 20
        batch_size = 1
        node_feat = np.concatenate([x,
                                    demand.reshape(1, n_nodes , 1),
                                    start.reshape(1, n_nodes , 1),
                                    end.reshape(1, n_nodes , 1),
                                    capacity.reshape(1, n_nodes , 1)], -1)        
        dist = x.reshape(batch_size, n_nodes, 1, 2) - x.reshape(batch_size, 1, n_nodes, 2)
        dist = np.sqrt((dist ** 2).sum(-1))
        edge_index = np.argsort(dist, -1)[:, :, 1:1 + n_edges]
        edge_feat = dist[np.arange(batch_size).reshape(-1, 1, 1), np.arange(n_nodes).reshape(1, -1, 1), edge_index]
        inverse_edge_index = -np.ones(shape=[batch_size, n_nodes, n_nodes], dtype="int")
        inverse_edge_index[np.arange(batch_size).reshape(-1, 1, 1), edge_index, np.arange(n_nodes).reshape(1, -1, 1)] = np.arange(n_edges).reshape(1, 1, -1) + np.arange(n_nodes).reshape(1, -1, 1) * n_edges
        inverse_edge_index = inverse_edge_index[np.arange(batch_size).reshape(-1, 1, 1), np.arange(n_nodes).reshape(1, -1, 1), edge_index]
        edge_index_np = edge_index

        node_feat = Variable(torch.FloatTensor(node_feat).type(torch.cuda.FloatTensor), requires_grad=False) # B x N x 2
        edge_feat = Variable(torch.FloatTensor(edge_feat).type(torch.cuda.FloatTensor), requires_grad=False).view(batch_size, -1, 1) # B x 20N x 1
        edge_index = Variable(torch.LongTensor(edge_index).type(torch.cuda.LongTensor), requires_grad=False).view(batch_size, -1) # B x 20N
        inverse_edge_index = Variable(torch.FloatTensor(inverse_edge_index).type(torch.cuda.LongTensor), requires_grad=False).view(batch_size, -1) # B x 20N
        candidate_test_1 = []
        candidate_test_2 = []
        
        y_edges1, y_edges2,  _, _, y_nodes = net.directed_forward(node_feat, edge_feat, edge_index, inverse_edge_index, None, None, None, 20)
        y_edges1 = y_edges1.detach().cpu().numpy()
        y_edges1 = y_edges1[:, :, 1].reshape(batch_size, n_nodes, n_edges)
        y_edges1 = np.argsort(-y_edges1, -1)
        edge_index = np.array(edge_index_np)
        candidate_index = edge_index[np.arange(batch_size).reshape(-1, 1, 1), np.arange(y_edges1.shape[1]).reshape(1, -1, 1), y_edges1]
        candidate_test_1.append(candidate_index[:, :, :20])

        y_edges2 = y_edges2.detach().cpu().numpy()
        y_edges2 = y_edges2[:, :, 1].reshape(batch_size, n_nodes, n_edges)
        y_edges2 = np.argsort(-y_edges2, -1)
        candidate_index = edge_index[np.arange(batch_size).reshape(-1, 1, 1), np.arange(y_edges2.shape[1]).reshape(1, -1, 1), y_edges2]
        candidate_test_2.append(candidate_index[:, :, :20])
        candidate_test_1 = np.concatenate(candidate_test_1, 0)
        candidate_test_2 = np.concatenate(candidate_test_2, 0)
        with open("result/lilim_pdptw/candidate/" + instance_name + ".txt", "w") as f:
            f.write(str((n_nodes - 1 + num_vehicles) * 2) + "\n")
            line = "1 0 5 " + str(n_nodes + num_vehicles) + " 0"
            for _ in range(4):
                line += " " + str(2 * (n_nodes - 1) + 2 * num_vehicles - _) + " 1"
            f.write(line + "\n")
            for j in range(1, n_nodes):
                line = str(j + 1) + " 0 5 " + str(j  + n_nodes + num_vehicles) + " 1"
                for _ in range(4):
                    line += " " + str(candidate_test_2[0,j, _]  + n_nodes + num_vehicles) + " 1"
                f.write(line + "\n")
            for j in range(num_vehicles - 1):
                line = str(n_nodes + 1 + j) + " 0 5 " + str(n_nodes + j + n_nodes + num_vehicles) + " 0 " + str(n_nodes + num_vehicles) + " 1"
                for _ in range(3):
                    line += " " + str(n_nodes + _ + n_nodes + num_vehicles) + " 1" 
                f.write(line + "\n")
            
            line = str(n_nodes + num_vehicles) + " 0 5 1 0"
            for _ in range(4):
                line += " " + str( n_nodes - 1 + num_vehicles - _) + " 1"
            f.write(line + "\n")
            for j in range(1, n_nodes):
                line = str(j + n_nodes + num_vehicles) + " 0 5 " + str(j + 1) + " 1"
                for _ in range(4):
                    line += " " + str(candidate_test_1[0,j, _] + 1) + " 1"
                f.write(line + "\n")
            for j in range(num_vehicles - 1):
                line = str(n_nodes + j + n_nodes + num_vehicles) + " 0 5 " + str(n_nodes + 1 + j) + " 0"
                for _ in range(4):
                    line += " " + str(n_nodes - 1 + num_vehicles - _) + " 1"
                f.write(line + "\n")
            f.write("-1\nEOF\n")

def solve_NeuroLKH(instance_name, opt_value, rerun=False, max_trials=1000):
    para_filename = "result/lilim_pdptw/NeuroLKH_para/" + instance_name + ".para"
    log_filename = "result/lilim_pdptw/NeuroLKH_log/" + instance_name + ".log"
    if rerun or not os.path.isfile(log_filename):
        write_para(instance_name, "NeuroLKH", para_filename, opt_value, max_trials=max_trials)
        with open(log_filename, "w") as f:
            check_call(["./LKH", para_filename], stdout=f)
    return read_results(log_filename, max_trials)


def read_results(log_filename, max_trials):
    results = []
    objs = []
    penalties = []
    runtimes = []
    with open(log_filename, "r") as f:
        lines = f.readlines()
        successes = int(lines[-12].split(" ")[-2].split("/")[0])
        cost_min = float(lines[-11].split(",")[0].split(" ")[-1])
        cost_avg = float(lines[-11].split(",")[1].split(" ")[-1])
        trials_min = float(lines[-7].split(",")[0].split(" ")[-2])
        trials_avg = float(lines[-7].split(",")[1].split(" ")[-2])
        time = float(lines[-7].split(",")[1].split(" ")[-2])

        for line in lines: # read the obj and runtime for each trial
            if line[:6] == "-Trial":
                line = line.strip().split(" ")
                assert len(objs) + 1 == int(line[-4])
                objs.append(int(line[-2]))
                penalties.append(int(line[-3]))
                runtimes.append(float(line[-1]))
        final_obj = int(lines[-11].split(",")[0].split(" ")[-1])
        assert objs[-1] == final_obj

        return successes, cost_min, cost_avg, trials_min, trials_avg, time, objs, penalties, runtimes

def eval_dataset(instance_names, method, args, opt_values, rerun=True, max_trials=1000):
    os.makedirs("result/lilim_pdptw/" + method + "_para", exist_ok=True)
    os.makedirs("result/lilim_pdptw/" + method + "_log", exist_ok=True)
    if method == "NeuroLKH":
        os.makedirs("result/lilim_pdptw/candidate", exist_ok=True)
        net = SparseGCNModel(problem="pdptw")
        net.cuda()
        saved = torch.load(args.model_path)
        net.load_state_dict(saved["model"])
        sgn_start_time = time.time()
        with torch.no_grad():
            infer_SGN_write_candidate(net, instance_names)
        sgn_runtime = time.time() - sgn_start_time
        with Pool(os.cpu_count()) as pool:
            results = list(tqdm.tqdm(pool.imap(method_wrapper, [("NeuroLKH", instance_names[i], opt_values[instance_names[i]], rerun, max_trials) for i in range(len(instance_names))]), total=len(instance_names)))
    else:
        assert method == "LKH"
        feat_runtime = 0
        sgn_runtime = 0
        with Pool(os.cpu_count()) as pool:
            results = list(tqdm.tqdm(pool.imap(method_wrapper, [("LKH", instance_names[i], opt_values[instance_names[i]], rerun, max_trials) for i in range(len(instance_names))]), total=len(instance_names)))
    #results = np.array(results)
    #dataset_objs = results[:, 0, :].mean(0)
    #dataset_penalties = results[:, 1, :].mean(0)
    #dataset_runtimes = results[:, 2, :].sum(0)
    return results #, dataset_objs, dataset_penalties, dataset_runtimes, sgn_runtime

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_path', type=str, default='pretrained/25.pt', help='')
    parser.add_argument('--n_samples', type=int, default=22, help='')
    parser.add_argument('--lkh_trials', type=int, default=10000, help='')
    parser.add_argument('--neurolkh_trials', type=int, default=10000, help='')
    args = parser.parse_args()
    instance_names = "lc101 lc102 lc103 lc104 lc105 lc106 lc107 lc108 lc109 LC1_2_1 LC1_2_2 LC1_2_3 LC1_2_4 LC1_2_5 LC1_2_6 LC1_2_7 LC1_2_8 LC1_2_9 LC1_2_10 LC1_4_1 LC1_4_2 LC1_4_3 LC1_4_4 LC1_4_5 LC1_4_6 LC1_4_7 LC1_4_8 LC1_4_9 LC1_4_10 LC1_8_1 LC1_8_2 LC1_8_3 LC1_8_4 LC1_8_5 LC1_8_6 LC1_8_7 LC1_8_8 LC1_8_9 LC1_8_10 LC1_10_1 LC1_10_2 LC1_10_3 LC1_10_4 LC1_10_5 LC1_10_6 LC1_10_7 LC1_10_8 LC1_10_9 LC1_10_10"
    instance_names = instance_names.split(" ")[:args.n_samples]
    with open("lilim_data/opt.pkl", "rb") as f:
        opt_values = pickle.load(f)
        
    neurolkh_results = eval_dataset(instance_names, "NeuroLKH", args=args , opt_values=opt_values, rerun=True, max_trials=args.neurolkh_trials) 
    lkh_results = eval_dataset(instance_names, "LKH", args=args , opt_values=opt_values, rerun=True, max_trials=args.lkh_trials)

    print ("Successes Best Avgerage Trials_Min Trials_Avg Time")
    for i in range(len(lkh_results)):
        print ("------%s------" % (instance_names[i]))
        print (lkh_results[i])
        print (neurolkh_results[i])



