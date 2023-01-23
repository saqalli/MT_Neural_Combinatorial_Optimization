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
        f.write("PROBLEM_FILE = cvrplib_data/" + instance_name + ".vrp\n")
        f.write("MAX_TRIALS = " + str(max_trials) + "\n")
        f.write("SPECIAL\nRUNS = 1\n")
        f.write("OPTIMUM = " + str(opt_value) + "\n")
        if method == "NeuroLKH":
            f.write("SEED = 1234\n")
            f.write("CANDIDATE_FILE = result/cvrplib/candidate/" + instance_name + ".txt\n")
        elif method == "FeatGenerate":
            f.write("GerenatingFeature\n")
            f.write("Feat_FILE = result/" + dataset_name + "/feat/" + instance_name + ".txt\n")
        else:
            assert method == "LKH"

def read_feat(feat_filename, max_nodes):
    n_neighbours = 20
    edge_index = np.zeros([1, max_nodes, n_neighbours], dtype="int")
    with open(feat_filename, "r") as f:
        lines = f.readlines()
        n_nodes_extend = int(lines[0].strip())
        for j in range(n_nodes_extend):
            line = lines[j + 1].strip().split(" ")
            line = [int(_) for _ in line]
            assert len(line) == 43
            assert line[0] == j + 1
            for _ in range(n_neighbours):
                edge_index[0, j, _] = line[3 + _ * 2] - 1
    feat_runtime = float(lines[-2].strip())
    return edge_index, n_nodes_extend, feat_runtime

def write_candidate(dataset_name, instance_name, candidate, n_nodes_extend):
    n_node = candidate.shape[0]
    with open("result/" + dataset_name + "/candidate/" + instance_name + ".txt", "w") as f:
        f.write(str(n_nodes_extend) + "\n")
        for j in range(n_nodes_extend):
            line = str(j + 1) + " 0 5"
            for _ in range(5):
                line += " " + str(int(candidate[j, _]) + 1) + " " + str(_ * 100)
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
    para_filename = "result/cvrplib/LKH_para/" + instance_name + ".para"
    log_filename = "result/cvrplib/LKH_log/" + instance_name + ".log"
    if rerun or not os.path.isfile(log_filename):
        write_para(instance_name, "LKH", para_filename, opt_value, max_trials=max_trials)
        with open(log_filename, "w") as f:
            check_call(["./LKH", para_filename], stdout=f)
    return read_results(log_filename,max_trials)

def infer_SGN_write_candidate(net, instance_names):
    for instance_name in instance_names:
        with open("cvrplib_data/" + instance_name + ".vrp", "r") as f:
            lines = f.readlines()
            assert lines[4] == "EDGE_WEIGHT_TYPE : \tEUC_2D\t\n"
            assert lines[6] == "NODE_COORD_SECTION\t\t\n"
            n_nodes = int(lines[3].split(" ")[-1])
            for i in range(len(lines[0])):
                if (lines[0][i] == '\t') and (lines[0][i-3] == 'k'):
                    num_vehicles = lines[0][(i-2):i]
                elif (lines[0][i] == '\t') and (lines[0][i-4] == 'k'):
                    num_vehicles = lines[0][(i-3):i]
            num_vehicles = int(num_vehicles)
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
            demand = []
            for i in range(n_nodes):
                line = [int(_) for _ in lines[8 + n_nodes + i].strip().split()]
                assert len(line) == 2
                assert line[0] == i + 1
                demand.append(line[1])
            demand = np.zeros([1,n_nodes])
            for i in range(n_nodes):
                line = [int(_) for _ in lines[8 + n_nodes + i].strip().split()]
                assert len(line) == 2
                assert line[0] == i + 1
                demand[:,i] = line[1]
            demand = np.array(demand).astype(int)
            demand = np.concatenate([np.zeros([1, 1]), demand, np.zeros([1, max_nodes - n_nodes - 1])], -1)
            demand = demand/int(lines[5].split(" ")[-1])
            capacity = np.zeros([1, max_nodes])
            capacity[:, 0] = 1
            capacity[:, n_nodes + 1:] = 1
            
            x = np.concatenate([x] + [x[:, 0:1, :] for _ in range(max_nodes - n_nodes)], 1)
           
        n_edges = 20
        batch_size = 1
        node_feat = np.concatenate([x, demand.reshape([1, max_nodes, 1]), capacity.reshape([1, max_nodes, 1])], -1)
        dist = node_feat[:,:,:2].reshape(batch_size, max_nodes, 1, 2) - node_feat[:,:,:2].reshape(batch_size, 1, max_nodes, 2)
        dist = np.sqrt((dist ** 2).sum(-1))
        edge_index = np.argsort(dist, -1)[:, :, 1:1 + n_edges]
        edge_feat = dist[np.arange(batch_size).reshape(-1, 1, 1), np.arange(max_nodes).reshape(1, -1, 1), edge_index]
        inverse_edge_index = -np.ones(shape=[batch_size, max_nodes, max_nodes], dtype="int")
        inverse_edge_index[np.arange(batch_size).reshape(-1, 1, 1), edge_index, np.arange(max_nodes).reshape(1, -1, 1)] = np.arange(n_edges).reshape(1, 1, -1) + np.arange(max_nodes).reshape(1, -1, 1) * n_edges
        inverse_edge_index = inverse_edge_index[np.arange(batch_size).reshape(-1, 1, 1), np.arange(max_nodes).reshape(1, -1, 1), edge_index]
        edge_index_np = edge_index

        node_feat = Variable(torch.FloatTensor(node_feat).type(torch.cuda.FloatTensor), requires_grad=False) # B x N x 2
        edge_feat = Variable(torch.FloatTensor(edge_feat).type(torch.cuda.FloatTensor), requires_grad=False).view(batch_size, -1, 1) # B x 20N x 1
        edge_index = Variable(torch.LongTensor(edge_index).type(torch.cuda.LongTensor), requires_grad=False).view(batch_size, -1) # B x 20N
        inverse_edge_index = Variable(torch.FloatTensor(inverse_edge_index).type(torch.cuda.LongTensor), requires_grad=False).view(batch_size, -1) # B x 20N
        candidate_test = []
        label = None
        edge_cw = None

        y_edges, _, y_nodes = net.forward(node_feat, edge_feat, edge_index, inverse_edge_index, label, edge_cw, n_edges)
        y_edges = y_edges.detach().cpu().numpy()
        y_edges = y_edges[:, :, 1].reshape(batch_size, max_nodes, n_edges)
        y_edges = np.argsort(-y_edges, -1)
        edge_index = np.array(edge_index_np)
        candidate_index = edge_index[np.arange(batch_size).reshape(-1, 1, 1), np.arange(max_nodes).reshape(1, -1, 1), y_edges]
        candidate_test.append(candidate_index[:, :, :5])
        candidate_test = np.concatenate(candidate_test, 0)
        with open("result/cvrplib/candidate/" + instance_name + ".txt", "w") as f:
            f.write(str(max_nodes) + "\n")
            for j in range(max_nodes):
                line = str(j + 1) + " 0 5"
                for _ in range(5):
                    line += " " + str(candidate_test[0, j, _] + 1) + " 1"
                f.write(line + "\n")
            f.write("-1\nEOF\n")

def solve_NeuroLKH(instance_name, opt_value, rerun=False, max_trials=1000):
    para_filename = "result/cvrplib/NeuroLKH_para/" + instance_name + ".para"
    log_filename = "result/cvrplib/NeuroLKH_log/" + instance_name + ".log"
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
    os.makedirs("result/cvrplib/" + method + "_para", exist_ok=True)
    os.makedirs("result/cvrplib/" + method + "_log", exist_ok=True)
    if method == "NeuroLKH":
        os.makedirs("result/cvrplib/candidate", exist_ok=True)
        net = SparseGCNModel(problem="cvrp")
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
    parser.add_argument('--model_path', type=str, default='pretrained/cvrp_neurolkh.pt', help='')
    parser.add_argument('--n_samples', type=int, default=2, help='')
    parser.add_argument('--lkh_trials', type=int, default=10000, help='')
    parser.add_argument('--neurolkh_trials', type=int, default=10000, help='')
    args = parser.parse_args()
    instance_names = "X-n303-k21 X-n308-k13 X-n313-k71 X-n317-k53 X-n322-k28 X-n327-k20 X-n331-k15 X-n336-k84 X-n344-k43 X-n351-k40 X-n359-k29 X-n367-k17 X-n376-k94 X-n384-k52 X-n393-k38 X-n401-k29 X-n411-k19 X-n420-k130 X-n429-k61 X-n439-k37 X-n449-k29 X-n459-k26 X-n469-k138 X-n480-k70 X-n491-k59 X-n502-k39 X-n513-k21 X-n524-k153 X-n536-k96 X-n548-k50 X-n561-k42 X-n573-k30 X-n586-k159 X-n599-k92 X-n613-k62 X-n627-k43 X-n641-k35 X-n655-k131 X-n670-k130 X-n685-k75 X-n701-k44 X-n716-k35 X-n733-k159 X-n749-k98 X-n766-k71 X-n783-k48 X-n801-k40 X-n819-k171 X-n837-k142 X-n856-k95 X-n876-k59 X-n895-k37 X-n916-k207 X-n936-k151 X-n957-k87 X-n979-k58 X-n1001-k43"
    instance_names = instance_names.split(" ")[:args.n_samples]
    with open("cvrplib_data/opt.pkl", "rb") as f:
        opt_values = pickle.load(f)
    neurolkh_results = eval_dataset(instance_names, "NeuroLKH", args=args, opt_values=opt_values, rerun=True, max_trials=args.neurolkh_trials)         
    lkh_results = eval_dataset(instance_names, "LKH", args=args, opt_values=opt_values,rerun=True, max_trials=args.lkh_trials)

    print ("Successes Best Avgerage Trials_Min Trials_Avg Time")
    for i in range(len(lkh_results)):
        print ("------%s------" % (instance_names[i]))
        print (lkh_results[i])
        print (neurolkh_results[i])







    


