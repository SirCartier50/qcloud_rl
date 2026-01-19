# job scheduler define how to schedule job, should work with controoller
import random
import time
from job import job, job_generator
from cluster import qCloud, create_random_topology
from pytket import Circuit, OpType, qasm
import math
import networkx as nx
import pymetis
from des import Event, FinishedJob
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from pytket.utils import Graph
from flowScheduler import  flow_scheduler_1
from des import DES, generatingJob
from pytket.circuit.display import get_circuit_renderer

def generate_splits(total, parts):
    if parts == 1:
        yield (total,)
        return
    for i in range(total + 1):
        for split in generate_splits(total - i, parts - 1):
            yield (i,) + split

def max_scheduled_tasks_with_machines(jobs, machines):
    n_jobs = len(jobs)
    n_machines = len(machines)
    
    # Initialize DP table
    dp = np.zeros([n_jobs + 1] + [m + 1 for m in machines], dtype=int)
    allocations = [[{} for _ in range(np.prod([m + 1 for m in machines]))] for _ in range(n_jobs + 1)]
    
    # Fill DP table
    for i in range(1, n_jobs + 1):
        job = jobs[i - 1]
        for res in np.ndindex(*[m + 1 for m in machines]):
            # Copy previous state
            dp[i][res] = dp[i - 1][res]
            allocations[i][np.ravel_multi_index(res, [m + 1 for m in machines])] = allocations[i - 1][np.ravel_multi_index(res, [m + 1 for m in machines])].copy()
            
            # Try to allocate job to current resources
            for split in generate_splits(job, len(machines)):
                if all(res[j] >= split[j] for j in range(n_machines)):
                    new_res = tuple(res[j] - split[j] for j in range(n_machines))
                    if dp[i - 1][new_res] + 1 > dp[i][res]:
                        dp[i][res] = dp[i - 1][new_res] + 1
                        allocations[i][np.ravel_multi_index(res, [m + 1 for m in machines])] = allocations[i - 1][np.ravel_multi_index(new_res, [m + 1 for m in machines])].copy()
                        allocations[i][np.ravel_multi_index(res, [m + 1 for m in machines])][i - 1] = split

    # Find maximum number of scheduled tasks
    max_tasks = 0
    final_res = tuple(m for m in machines)
    for res in np.ndindex(*[m + 1 for m in machines]):
        if dp[n_jobs][res] > max_tasks:
            max_tasks = dp[n_jobs][res]
            final_res = res

    # Backtrack to find job allocations
    scheduled_jobs = [j for j in allocations[n_jobs][np.ravel_multi_index(final_res, [m + 1 for m in machines])].keys()]
    job_allocations = [allocations[n_jobs][np.ravel_multi_index(final_res, [m + 1 for m in machines])][j] for j in scheduled_jobs]

    # Create a dictionary for job allocations
    allocation_dict = {}
    for job_index, allocation in zip(scheduled_jobs, job_allocations):
        allocation_dict[job_index] = allocation

    return max_tasks, allocation_dict
  



class placement:
    def __init__(self, job, partition, qpu_mapping, wig):
        self.job = job
        if partition == 1:
            self.partition = partition
        else:
            self.partition = partition[0]
        self.qpu_mapping = qpu_mapping
        # self.cost = self.get_cost()
        self.communication_cost = None
        self.dag_longest_path_length = None
        self.score = None
        self.time = None
        self.wig = wig
        self.improvement = False

    def get_time(self, partition):
        dag = self.get_remote_DAG(partition)
        longest_path = nx.dag_longest_path(dag)
        self.dag_longest_path_length = len(longest_path)
        # pos = nx.spring_layout(dag)  # positions for all nodes
        # nx.draw_networkx(dag, pos)
        # plt.show()
        self.time = len(longest_path)
        pass

    def get_remote_DAG(self, partition):
        # big error previously: wig has another mapping from node to index:
        # eg: suppose wig nodes are: [q[6], q[7], q[5], q[8], q[0], q[1], q[2], q[3], q[4]]
        # and result of partition is [0, 0, 0, 0, 1, 1, 1, 1, 1]
        # this means q[6], q[7], q[5], q[8] are in partition 0, and q[0], q[1], q[2], q[3], q[4] are in partition 1
        # not q[0], q[1], q[2], q[3], q[4] are in partition 0, and q[6], q[7], q[5], q[8] are in partition 1
        one_qubit_gates = [OpType.H, OpType.T, OpType.S, OpType.X, OpType.Y, OpType.Z, OpType.Rx, OpType.Ry, OpType.Rz,
                           OpType.U1, OpType.U2, OpType.U3, OpType.Barrier]
        circuit = self.job.circuit
        node_to_int = {node: i for i, node in enumerate(self.wig.nodes())}
        modified_circuit = Circuit()
        for qubit in circuit.qubits:
            modified_circuit.add_qubit(qubit)
        qubits = list(circuit.qubits)
        part_vert = partition[0]
        partitions = {i: [] for i in set(part_vert)}
        graph = nx.Graph()
        for i, partition in enumerate(part_vert):
            partitions[partition].append(qubits[i])
        for command in circuit:
            op_type = command.op.type
            qubits = list(command.args)

            # if op_type not in one_qubit_gates:
            if len(qubits) == 2 and op_type != OpType.Measure:
                node1 = qubits[0]
                node2 = qubits[1]
                # in_different_partitions = any(qubit in partitions[0] for qubit in qubits) and any(
                #     qubit in partitions[1] for qubit in qubits)
                # add cx gate for all two-qubit gate for now:
                part_1 = self.partition[node_to_int[node1]]
                part_2 = self.partition[node_to_int[node2]]
                if self.partition[node_to_int[node1]] != self.partition[node_to_int[node2]]:
                    modified_circuit.add_gate(op_type, qubits)
                    if not graph.has_edge(part_1, part_2):
                        graph.add_edge(part_1, part_2, weight=1)
                    else:
                        graph[part_1][part_2]['weight'] += 1
        self.modified_circuit = modified_circuit
        self.remote_wig = graph
        # nx.draw(graph, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000,
        #         font_size=15)
        # plt.show()

        g = Graph(modified_circuit)
        # g.save_DAG("dag.txt")
        dag = g.as_nx()
        self.remote_dag = dag
        return dag

    def get_communication_cost(self, wig):
        sum = 0
        node_to_int = {node: i for i, node in enumerate(wig.nodes())}
        int_to_node = {i: node for node, i in node_to_int.items()}
        for edge in wig.edges(data=True):
            node1, node2, data = edge
            if self.partition[node_to_int[node1]] != self.partition[node_to_int[node2]]:
                sum += data['weight']
        self.communication_cost = sum

    def get_colloboration_data(self, qpu_mapping):
        pass

class GraphPartitioner:
    def partition_circuit(self, n_parts, sizes, graph):
        node_to_int = {node: i for i, node in enumerate(graph.nodes())}
        int_to_node = {i: node for node, i in node_to_int.items()}  # Reverse mapping

        xadj = [0]
        adjncy = []
        eweights = []

        for node in graph.nodes():
            int_node = node_to_int[node]
            neighbors = [node_to_int[neighbor] for neighbor in graph.neighbors(node)]
            adjncy.extend(neighbors)
            xadj.append(xadj[-1] + len(neighbors))
            for int_neighbor in neighbors:
                orig_node = int_to_node[int_node]
                orig_neighbor = int_to_node[int_neighbor]
                if graph.has_edge(orig_node, orig_neighbor):
                    weight = graph[orig_node][orig_neighbor]['weight']
                    eweights.append(weight)

        # Initial partitioning with total partitions equal to the sum of sizes
        total_partitions = sum(sizes)
        cutcount, part_vert = pymetis.part_graph(total_partitions, xadj=xadj, adjncy=adjncy, eweights=eweights)

        # Map each node to its initial partition
        initial_partitions = [[] for _ in range(total_partitions)]
        for node, part in enumerate(part_vert):
            initial_partitions[part].append(node)

        # Sort the initial partitions by size
        initial_partitions.sort(key=len, reverse=True)

        # Initialize the final partitions
        final_partitions = [[] for _ in range(n_parts)]
        partition_sizes = [0] * n_parts

        # Allocate partitions while respecting the size constraints
        for i in range(total_partitions):
            for j in range(n_parts):
                if partition_sizes[j] + len(initial_partitions[i]) <= sizes[j]:
                    final_partitions[j].extend(initial_partitions[i])
                    partition_sizes[j] += len(initial_partitions[i])
                    break

        self.kernighan_lin_refinement(graph, final_partitions, partition_sizes, sizes, n_parts)

        final_partitions = [[int_to_node[node] for node in partition] for partition in final_partitions]

        final_edgecuts = self.calculate_edge_cuts(graph, final_partitions)

        return final_partitions, final_edgecuts

    def calculate_edge_cuts(self, graph, partitions):
        edge_cuts = 0
        part_map = {}
        for part_id, nodes in enumerate(partitions):
            for node in nodes:
                part_map[node] = part_id

        for u, v, data in graph.edges(data=True):
            if u in part_map and v in part_map and part_map[u] != part_map[v]:
                edge_cuts += data['weight']

        return edge_cuts

    def kernighan_lin_refinement(self, graph, partitions, partition_sizes, sizes, n_parts):
        max_iterations = 10  # Number of iterations for refinement
        for iteration in range(max_iterations):
            improved = False
            print(f"Iteration {iteration}")
            for i in range(n_parts):
                nodes_to_check = list(partitions[i])  # Make a copy to avoid modifying while iterating
                for node in nodes_to_check:
                    best_partition = i
                    best_swap_node = None
                    min_cut = self.calculate_edge_cuts(graph, partitions)
                    for j in range(n_parts):
                        if j != i:
                            if partition_sizes[j] < sizes[j]:
                                # Temporarily move node to partition j
                                partitions[i].remove(node)
                                partitions[j].append(node)
                                partition_sizes[i] -= 1
                                partition_sizes[j] += 1
                                new_cut = self.calculate_edge_cuts(graph, partitions)
                                if new_cut < min_cut:
                                    best_partition = j
                                    best_swap_node = None
                                    min_cut = new_cut
                                # Move node back
                                partitions[j].remove(node)
                                partitions[i].append(node)
                                partition_sizes[j] -= 1
                                partition_sizes[i] += 1
                            elif partition_sizes[j] == sizes[j]:
                                # Try swapping nodes
                                for swap_node in partitions[j]:
                                    partitions[i].remove(node)
                                    partitions[j].remove(swap_node)
                                    partitions[j].append(node)
                                    partitions[i].append(swap_node)
                                    new_cut = self.calculate_edge_cuts(graph, partitions)
                                    if new_cut < min_cut:
                                        best_partition = j
                                        best_swap_node = swap_node
                                        min_cut = new_cut
                                    # Swap nodes back
                                    partitions[j].remove(node)
                                    partitions[i].remove(swap_node)
                                    partitions[j].append(swap_node)
                                    partitions[i].append(node)
                    # If a better partition was found, move/swap the node
                    if best_partition != i:
                        if best_swap_node is None:
                            print(f"Moving node {node} from partition {i} to partition {best_partition}")
                            partitions[i].remove(node)
                            partitions[best_partition].append(node)
                            partition_sizes[i] -= 1
                            partition_sizes[best_partition] += 1
                        else:
                            print(f"Swapping node {node} in partition {i} with node {best_swap_node} in partition {best_partition}")
                            partitions[i].remove(node)
                            partitions[best_partition].remove(best_swap_node)
                            partitions[i].append(best_swap_node)
                            partitions[best_partition].append(node)
                        improved = True
            if not improved:
                break

class job_scheduler:
    def __init__(self, job_queue, des, qcloud, scheduler_type="default", flow_scheduler_type="default", logger= None):
        self.job_queue = job_queue
        self.qcloud = qcloud
        # self.controller = controller(self.)
        self.des = des
        self.scheduled_job = []
        # unscheduled_job stores the job that currenntly can't be processed.
        self.unscheduled_job = []
        # registered_job stores the job that has has been assigend placement and waiting for flofw_scheduling
        self.registered_jobs = []
    # a simple method to register all unscheduled job and enters the flow scheduling step.
        self.schduler_type = scheduler_type
        self.flow_scheduler_type = flow_scheduler_type
        self.logger = logger
    def regiester_unscheduled_job(self):
        while self.job_queue: 
            job = self.job_queue.pop(0)
            self.unscheduled_job.append(job)
        
    
    def test_placement_single_ga(self, logger):
        print("test placement single ga sb")
        partition, result = self.qcloud.ga_find_placement(self.job_queue[0].circuit)
        print(result)
        possible_placements = self.find_simple_placement(self.job_queue[0])
        best_placement = self.score(possible_placements)
        old_result = self.qcloud.compute_weighted_length(best_placement.partition, best_placement.remote_wig, best_placement.qpu_mapping[0])
        pp = self.qcloud.improve_placement(best_placement)
        if pp is not None:
            print(pp)
            best_placement = pp
            print(best_placement.qpu_to_dict)
        # gcm_result = self.qcloud.compute_weighted_length(best_placement.partition, best_placement.remote_wig, best_placement.qpu_mapping[0])
            gcm_result = self.qcloud.calculate_cost(best_placement.qpu_to_dict, self.job_queue[0].circuit)
        else: 
            gcm_result = self.qcloud.compute_weighted_length(best_placement.partition, best_placement.remote_wig, best_placement.qpu_mapping[0])
        print(partition)
        print(old_result, result, gcm_result)
        logger.log(self.job_queue[0].name,  ga=result, gcm = gcm_result[1] if isinstance(gcm_result, tuple) else gcm_result, ncp = self.qcloud.qpu_qubit_num)

    def test_placement_single_cp(self, logger):
        print("test placement single ga")
        partition, result = self.qcloud.ga_find_placement(self.job_queue[0].circuit)
        annealing_partition, annealing_result = self.qcloud.sa_find_placement(self.job_queue[0].circuit)
        # print(result)
        random_partition = self.qcloud.ramdom_find_placement(self.job_queue[0].circuit)
        possible_placements = self.find_simple_placement(self.job_queue[0])
        best_placement = self.score(possible_placements)
        old_result = self.qcloud.compute_weighted_length(best_placement.partition, best_placement.remote_wig, best_placement.qpu_mapping[0])
        pp = self.qcloud.improve_placement(best_placement)
        if pp:
            
            best_placement = pp
            print(best_placement.qpu_to_dict)
        possible_placements_bfs = self.find_simple_placement_bfs(self.job_queue[0])
        try:
            best_placement_bfs = self.score(possible_placements_bfs)
        except:
            best_placement_bfs = possible_placements_bfs[0]
        result_bfs = self.qcloud.compute_weighted_length(best_placement_bfs.partition, best_placement_bfs.remote_wig, best_placement_bfs.qpu_mapping[0])
        # gcm_result = self.qcloud.compute_weighted_length(best_placement.partition, best_placement.remote_wig, best_placement.qpu_mapping[0])
        gcm_result = self.qcloud.calculate_cost(best_placement.qpu_to_dict, self.job_queue[0].circuit)
        qubit_partition_dict = {qubit: key for key, value in random_partition.items() for qubit in value}
        random_placement_cost = self.qcloud.calculate_cost_qpu(qubit_partition_dict, self.job_queue[0].circuit)
        print(partition)
        print(old_result, result, gcm_result)
        logger.log(self.job_queue[0].name,  ga=result, gcm = gcm_result[1] if isinstance(gcm_result, tuple) else gcm_result, sa = annealing_result, bfs=result_bfs[1], random=random_placement_cost,ncp = self.qcloud.qpu_qubit_num)

    def test_placement_single(self, logger):
        random_partition = self.qcloud.ramdom_find_placement(self.job_queue[0].circuit)
        possible_placements = self.find_simple_placement(self.job_queue[0])
        annealing_partition, result = self.qcloud.sa_find_placement(self.job_queue[0].circuit)
        print({key: len(value) for key, value in annealing_partition.items()})
        try:
            best_placement = self.score(possible_placements)
        except:
            best_placement = possible_placements[0]
        gcm_result = self.qcloud.compute_weighted_length(best_placement.partition, best_placement.remote_wig, best_placement.qpu_mapping[0])
        print(gcm_result)
        pp = self.qcloud.improve_placement(best_placement)
        if pp:
            best_placement = pp
            gcm_result = self.qcloud.calculate_cost(best_placement.qpu_to_dict, self.job_queue[0].circuit)
        else:
            gcm_result = self.qcloud.compute_weighted_length(best_placement.partition, best_placement.remote_wig, best_placement.qpu_mapping[0])
        qubit_partition_dict = {qubit: key for key, value in random_partition.items() for qubit in value}
        random_placement_cost = self.qcloud.calculate_cost_qpu(qubit_partition_dict, self.job_queue[0].circuit)
        possible_placements_bfs = self.find_simple_placement_bfs(self.job_queue[0])
        try:
            best_placement_bfs = self.score(possible_placements_bfs)
        except:
            best_placement_bfs = possible_placements_bfs[0]
        result_bfs = self.qcloud.compute_weighted_length(best_placement_bfs.partition, best_placement_bfs.remote_wig, best_placement_bfs.qpu_mapping[0])
        print(gcm_result)
        logger.log(self.job_queue[0].name, bfs=result_bfs[1], annealing= result, random=random_placement_cost, ga=0, gcm=gcm_result[1] if isinstance(gcm_result, tuple) else gcm_result)
    
    def test_placement_new(self, logger):
        possible_placements = self.find_simple_placement(self.job_queue[0])
        res = []
        for i in range(len(possible_placements)):
            res.append(self.qcloud.compute_weighted_length(possible_placements[i].partition, possible_placements[i].remote_wig, possible_placements[i].qpu_mapping[0]))
        print(res)


    def test_placement_single_smooth(self, logger, iteration):
        random_partition = self.qcloud.ramdom_find_placement(self.job_queue[0].circuit)
        possible_placements = self.find_simple_placement(self.job_queue[0])
        annealing_partition, result = self.qcloud.sa_find_placement(self.job_queue[0].circuit)
        print({key: len(value) for key, value in annealing_partition.items()})
        try:
            best_placement = self.score(possible_placements)
        except:
            best_placement = possible_placements[0]
        gcm_result = self.qcloud.compute_weighted_length(best_placement.partition, best_placement.remote_wig, best_placement.qpu_mapping[0])
        print(gcm_result)
        pp = self.qcloud.improve_placement(best_placement)
        if pp:
            best_placement = pp
            gcm_result = self.qcloud.calculate_cost(best_placement.qpu_to_dict, self.job_queue[0].circuit)
        else:
            gcm_result = self.qcloud.compute_weighted_length(best_placement.partition, best_placement.remote_wig, best_placement.qpu_mapping[0])
        qubit_partition_dict = {qubit: key for key, value in random_partition.items() for qubit in value}
        random_placement_cost = self.qcloud.calculate_cost_qpu(qubit_partition_dict, self.job_queue[0].circuit)
        # possible_placements_bfs = self.find_simple_placement_bfs(self.job_queue[0])
        # try:
        #     best_placement_bfs = self.score(possible_placements_bfs) 
        # except:
        #     best_placement_bfs = possible_placements_bfs[0]
        # result_bfs = self.qcloud.compute_weighted_length(best_placement_bfs.partition, best_placement_bfs.remote_wig, best_placement_bfs.qpu_mapping[0])

        bfs_reult_list= []
        for i in range(20):
            possible_placements_bfs = self.find_simple_placement_bfs(self.job_queue[0])
            try:
                best_placement_bfs = self.score(possible_placements_bfs) 
            except:
                best_placement_bfs = possible_placements_bfs[0]
            result_bfs = self.qcloud.compute_weighted_length(best_placement_bfs.partition, best_placement_bfs.remote_wig, best_placement_bfs.qpu_mapping[0])
            bfs_reult_list.append(result_bfs[1])
        result_bfs_mean = np.mean(bfs_reult_list)




        # logger.log(self.job_queue[0].name, bfs=result_bfs[1], annealing= result, random=random_placement_cost, ga=0, gcm=gcm_result[1] if isinstance(gcm_result, tuple) else gcm_result)
        logger.log(self.job_queue[0].name, ncp = self.qcloud.qpu_qubit_num , iteration=iteration, bfs=result_bfs_mean , annealing= result ,random=random_placement_cost, gcm = gcm_result[1] if isinstance(gcm_result, tuple) else gcm_result)

 
    def test_flow_scheduler_new(self, logger, epr, iteration):
        possible_placements = self.find_simple_placement(self.job_queue[0])
        best_placement = self.score(possible_placements)
        # pp = self.qcloud.improve_placement(best_placement)
        self._schedule_new(best_placement, self.job_queue[0])
        flowscheduler = flow_scheduler_1(self.des, self.qcloud, epr, "test")
        random_time, average_time, priority_time, greedy_time = flowscheduler.test_flow_all()
        logger.log(self.job_queue[0].name, iteration = iteration, epr = epr, ncm = self.qcloud.ncm_qubits, random =random_time, average=average_time, priority=priority_time, greedy=greedy_time)



    # this method test different flow scheduling methods on a single circuit.
    def test_flowscheduler(self, logger ):
        possible_placements = self.find_simple_placement(self.job_queue[0])
        best_placement = self.score(possible_placements)
        pp = self.qcloud.improve_placement(best_placement)
        # self._schedule_new(best_placement, self.job_queue[0])
        flowscheduler = flow_scheduler_1(self.des, self.qcloud, 0.5, "test")
        random_time, average_time, priority_time, greedy_time = flowscheduler.test_flow_all()
        logger.log(self.job_queue[0].name, random =random_time, average=average_time, priority=priority_time, greedy=greedy_time)

        # print(random_time, average_time, priority_time, greedy_time)
        # if pp:
        #     best_placement = pp
        # print(best_placement.qpu_to_dict)
        # circuit_renderer = get_circuit_renderer()
        # circuit_renderer.render_circuit_jupyter(best_placement.modified_circuit)
        
        # for op in best_placement.modified_circuit:
        #     print(op)
        # circuit_renderer.render_circuit_jupyter(best_placement.modified_circuit)
        # self._schedule_new(best_placement,self.job_queue[0])
        # flowscheduler = flow_scheduler_1(self.des, self.qcloud, 0.5, "test")
        # random_time, average_time, priority_time, greedy_time = flowscheduler.test_flow_all()
        # # logger.log(self.job_queue[0].name, random =random_time, average=average_time, priority=priority_time, greedy=greedy_time)
        # print(random_time, average_time, priority_time, greedy_time)
        # print("hh")
    
    def test_flowscheduler_epr(self,logger, epr):
        possible_placements = self.find_simple_placement(self.job_queue[0])
        best_placement = self.score(possible_placements)
        self._schedule_new(best_placement,self.job_queue[0])
        flowscheduler = flow_scheduler_1(self.des, self.qcloud, epr, "test")
        random_time, average_time, priority_time, greedy_time = flowscheduler.test_flow_all()
        logger.log(self.job_queue[0].name, random =random_time, average=average_time, priority=priority_time, greedy=greedy_time)



    def schedule_choice(self):
        if self.schduler_type == "bfs":
            self.schedule_bfs()
        elif self.schduler_type == "default":
            self.schedule()
        elif self.schduler_type == "greedy":
            self.schedule_greedy()

        elif self.schduler_type == "fifo":
            self.schedule_fifo()

        elif self.schduler_type == "annealing":
            self.schedule_annealing()
        
        elif self.schduler_type == 'rl':
            self.schedule_rl()
    
    def schedule_rl(self):
        self.job_queue.sort(
            key=lambda x:   ( x.circuit.n_2qb_gates() /(x.circuit.depth()* x.circuit.n_qubits)), reverse=True)
        print([(job.name, job.circuit.depth()) for job in self.job_queue])
        while len(self.job_queue) > 0:
            available_qubits = self.qcloud.get_available_qubits()
            all_none = all([job.circuit.n_qubits > available_qubits for job in self.job_queue])
            if all_none:
                # case 1 the cloud can not schedule any job in the queue. all jobs will be put into unscheduled_job
                self.regiester_unscheduled_job()
                break
            job = self.job_queue[0]
            if job.circuit.n_qubits > available_qubits:
                # case 2 just current job can't be scheudled, but the others may be scheduled, so need to continue
                self.unscheduled_job.append(self.job_queue.pop(0))
                print("no qpu available")
                continue
            possible_placements = self.find_simple_placement(job)
            if not possible_placements:
                # case 3 no placement can be found for the current job
                # self.regiester_unscheduled_job()
                # break
                self.unscheduled_job.append(self.job_queue.pop(0))
                continue
                

            best_placement = self.score(possible_placements)
            print("find_placement")
            current_time = self.des.current_time
            best_placement.start_time = current_time
            self._schedule_new(best_placement, job, "BFS")
            self.scheduled_job.append(job)
            self.job_queue.pop(0)
            count = Counter(best_placement.partition)
            used_qpu_qubits = {value: count[key]        for key, value in best_placement.qpu_mapping[0].items() }

            print("jobs are scheduled")
        all_available_qubits = self.qcloud.get_available_qubits()
        available_qubits = {qpu.qpuid: qpu.available_qubits for qpu in self.qcloud.qpus}
        flow_scheduler = flow_scheduler_1(self.des, self.qcloud, epr_p=0.3, name="BFS")
        flow_scheduler.run()
        return
            
    def schedule_annealing(self):
        print("annealing_start")
        while len(self.job_queue) > 0:
            available_qubits = self.qcloud.get_available_qubits()
            all_none = all([job.circuit.n_qubits > available_qubits for job in self.job_queue])
            if all_none:
                # case 1 the cloud can not schedule any job in the queue. all jobs will be put into unscheduled_job
                self.regiester_unscheduled_job()
                break
            job = self.job_queue[0]
            if job.circuit.n_qubits > available_qubits:
                # case 2 just current job can't be scheudled, but the others may be scheduled, so need to continue
                self.unscheduled_job.append(self.job_queue.pop(0))
                print("no qpu available")
                continue
            possible_placements = self.find_simple_placement_annealing(job)
            if not possible_placements:
                # case 3 no placement can be found for the current job
                self.regiester_unscheduled_job()
                break
            best_placement = self.score(possible_placements)
            print(best_placement.partition)
            current_time = self.des.current_time
            best_placement.start_time = current_time
            self._schedule_new(best_placement, job, "BFS")
            self.scheduled_job.append(job)
            self.job_queue.pop(0)
            all_available_qubits = self.qcloud.get_available_qubits()
            print("a")
        all_available_qubits = self.qcloud.get_available_qubits()
        available_qubits = {qpu.qpuid: qpu.available_qubits for qpu in self.qcloud.qpus}
        flow_scheduler = flow_scheduler_1(self.des, self.qcloud, epr_p=0.3, name="BFS")
        flow_scheduler.run()
        return


    
    def schedule_fifo(self):
        while len(self.job_queue) > 0:
            available_qubits = self.qcloud.get_available_qubits()
            all_none = all([job.circuit.n_qubits > available_qubits for job in self.job_queue])
            if all_none:
                # case 1 the cloud can not schedule any job in the queue. all jobs will be put into unscheduled_job
                self.regiester_unscheduled_job()
                break
            job = self.job_queue[0]
            if job.circuit.n_qubits > available_qubits:
                # case 2 just current job can't be scheudled, but the others may be scheduled, so need to continue
                self.unscheduled_job.append(self.job_queue.pop(0))
                print("no qpu available")
                continue
            possible_placements = self.find_simple_placement(job)
            if not possible_placements:
                # case 3 no placement can be found for the current job
                # self.regiester_unscheduled_job()
                # break
                self.unscheduled_job.append(self.job_queue.pop(0))
                continue
                
            best_placement = self.score(possible_placements)
            self._schedule_new(best_placement, job, "BFS")
            current_time = self.des.current_time
            best_placement.start_time = current_time
            self.scheduled_job.append(job)
            self.job_queue.pop(0)
            count = Counter(best_placement.partition)
            used_qpu_qubits = {value: count[key]        for key, value in best_placement.qpu_mapping[0].items() }

            print("jobs are scheduled")
        all_available_qubits = self.qcloud.get_available_qubits()
        available_qubits = {qpu.qpuid: qpu.available_qubits for qpu in self.qcloud.qpus}
        flow_scheduler = flow_scheduler_1(self.des, self.qcloud, epr_p=0.3, name="BFS")
        flow_scheduler.run()
        return
    
    def schedule(self):
        # self.job_queue.sort(
        #     key=lambda x: 0.3 * x.circuit.n_qubits + 0.3 * (x.circuit.n_2qb_gates() / x.circuit.n_qubits) + 0.4 * (
        #         x.circuit.depth()), reverse=True)
        # self.job_queue.sort( key=lambda x: x.circuit.n_qubits ,reverse=True)
        # in the while loop. find the maximum number of jobs that can be scheduled
        self.job_queue.sort(
            key=lambda x:   ( x.circuit.n_2qb_gates() /(x.circuit.depth()* x.circuit.n_qubits)), reverse=True)
        print([(job.name, job.circuit.depth()) for job in self.job_queue])
        while len(self.job_queue) > 0:
            available_qubits = self.qcloud.get_available_qubits()
            all_none = all([job.circuit.n_qubits > available_qubits for job in self.job_queue])
            if all_none:
                # case 1 the cloud can not schedule any job in the queue. all jobs will be put into unscheduled_job
                self.regiester_unscheduled_job()
                break
            job = self.job_queue[0]
            if job.circuit.n_qubits > available_qubits:
                # case 2 just current job can't be scheudled, but the others may be scheduled, so need to continue
                self.unscheduled_job.append(self.job_queue.pop(0))
                print("no qpu available")
                continue
            possible_placements = self.find_simple_placement(job)
            if not possible_placements:
                # case 3 no placement can be found for the current job
                # self.regiester_unscheduled_job()
                # break
                self.unscheduled_job.append(self.job_queue.pop(0))
                continue
                

            best_placement = self.score(possible_placements)
            # pp = self.qcloud.improve_placement(best_placement)
            # if pp is not None:
            #     best_placement = pp
            #     best_placement.improvement= True
            print("find_placement")
            current_time = self.des.current_time
            best_placement.start_time = current_time
            self._schedule_new(best_placement, job, "BFS")
            self.scheduled_job.append(job)
            self.job_queue.pop(0)
            count = Counter(best_placement.partition)
            used_qpu_qubits = {value: count[key]        for key, value in best_placement.qpu_mapping[0].items() }

            print("jobs are scheduled")
        all_available_qubits = self.qcloud.get_available_qubits()
        available_qubits = {qpu.qpuid: qpu.available_qubits for qpu in self.qcloud.qpus}
        flow_scheduler = flow_scheduler_1(self.des, self.qcloud, epr_p=0.3, name="BFS")
        flow_scheduler.run()
        return

    def max_tasks_scheduled(self, R, requests):
        n = len(requests)
        dp = [[0] * (R + 1) for _ in range(n + 1)]
        keep = [[False] * (R + 1) for _ in range(n + 1)]

        for i in range(1, n + 1):
            for j in range(R + 1):
                if dp[i-1][j] > dp[i][j]:
                    dp[i][j] = dp[i-1][j]
                    keep[i][j] = False
                if j >= requests[i-1] and dp[i-1][j-requests[i-1]] + 1 > dp[i][j]:
                    dp[i][j] = dp[i-1][j-requests[i-1]] + 1
                    keep[i][j] = True

        # Find the tasks that were chosen
        chosen_tasks = []
        j = R
        for i in range(n, 0, -1):
            if keep[i][j]:
                chosen_tasks.append(i-1)  # Add the task index (i-1 because tasks are 0-indexed)
                j -= requests[i-1]

        chosen_tasks.reverse()  # Reverse to get the order in which tasks were added
        return dp[n][R], chosen_tasks

    def schedule_greedy(self):
        # available_qubits = self.qcloud.get_available_qubits()
        # qubits_requirement = [job.circuit.n_qubits for job in self.job_queue]
        # chosen_circuit  = self.max_tasks_scheduled(available_qubits, qubits_requirement)
        # for idx in chosen_circuit:
        #     job = self.job_queue[idx]
        #     possible_placements = self.find_simple_placement(job)
        #     best_placement = self.score(possible_placements)
        #     print(possible_placements)
        qubits_requirement = [job.circuit.n_qubits for job in self.job_queue]
        qubits_qpu = [qpu.available_qubits for qpu in self.qcloud.qpus]
        selected_jobs =  max_scheduled_tasks_with_machines(qubits_requirement, qubits_qpu)
        print(selected_jobs)
        for idx in selected_jobs:
            possible_

    


    def schedule_bfs(self):
        self.job_queue.sort(
            key=lambda x: 0.3 * x.circuit.n_qubits + 0.3 * (x.circuit.n_2qb_gates() / x.circuit.n_qubits) + 0.4 * (
                x.circuit.depth()), reverse=True)

        # in the while loop. find the maximum number of jobs that can be scheduled
        while len(self.job_queue) > 0:
            available_qubits = self.qcloud.get_available_qubits()
            all_none = all([job.circuit.n_qubits > available_qubits for job in self.job_queue])
            if all_none:
                # case 1 the cloud can not schedule any job in the queue. all jobs will be put into unscheduled_job
                self.regiester_unscheduled_job()
                break
            job = self.job_queue[0]
            if job.circuit.n_qubits > available_qubits:
                # case 2 just current job can't be scheudled, but the others may be scheduled, so need to continue
                self.unscheduled_job.append(self.job_queue.pop(0))
                print("no qpu available")
                continue
            possible_placements = self.find_simple_placement_bfs(job)
            if not possible_placements:
                # case 3 no placement can be found for the current job
                self.regiester_unscheduled_job()
                break
            #print("possible placement: ", possible_placements)
            best_placement = self.score(possible_placements)
            #print("best placement: ", best_placement)
            # pp = self.qcloud.improve_placement(best_placement)
            # if pp is not None:
            #     best_placement = pp
            print("find_placement")
            current_time = self.des.current_time
            best_placement.start_time = current_time
            self._schedule_new(best_placement, job, "BFS")
            self.scheduled_job.append(job)
            self.job_queue.pop(0)
            all_available_qubits = self.qcloud.get_available_qubits()
            print("a")
        # print("all jobs are scheduled")

        flow_scheduler = flow_scheduler_1(self.des, self.qcloud, epr_p=0.3, name="BFS")
        flow_scheduler.run()
    
    
    def save_dag(self, circuit, name):
        g = Graph(circuit)
        g.save_DAG(name)

    # score each placement, return the best one
    def score(self, placement_list):
        if len(placement_list) == 1:
            return placement_list[0]
        time_list = [placement.time for placement in placement_list]
        communication_cost_list = [placement.communication_cost for placement in placement_list]
        min_time, max_time = min(time_list), max(time_list)
        min_communication_cost, max_communication_cost = min(communication_cost_list), max(communication_cost_list)
        if min_time == max_time:
            inverse_normalized_time_list = [1 for time in time_list]
        else:
            inverse_normalized_time_list = [1 - (time - min_time) / (max_time - min_time) for time in time_list]
        inverse_normalized_communication_cost_list = [
            1 - (communication_cost - min_communication_cost) / (max_communication_cost - min_communication_cost) for
            communication_cost in communication_cost_list]
        # both time and communication are better with smaller value
        for i, placement in enumerate(placement_list):
            placement.score = 0.5 * inverse_normalized_time_list[i] + 0.5 * inverse_normalized_communication_cost_list[
                i]
        # return the placement with highest score
        return max(placement_list, key=lambda x: x.score)
        # pass

    # schedule one job with given placement, register corresponding events to DES, run DAG
    def _schedule(self, placement, job, name=None):
        if placement.partition == 1:
            selected_qpu = random.choice(placement.qpu_mapping)
            selected_qpu.allocate_job(job, job.circuit.n_qubits)
            used_qpu = {selected_qpu.qpuid: job.circuit.n_qubits}
            single_finished_event = FinishedJob(placement.time, job, used_qpu, str(job.id) + ' finished')
            self.des.schedule_event(single_finished_event)
            return
        counts = Counter(placement.partition)
        # qpu mapping a list of tuples, first element is qpu index,second is partition index
        # now qpu mapping is a dict, key is partition index, value is qpu index
        for par_idx, qpu_id in placement.qpu_mapping[0].items():
            # modify the status of qpu
            qpu = self.qcloud.network.nodes[qpu_id]['qpu']
            qpu.allocate_job(job, counts[par_idx])

        # then use flow scheduler to simulate the running of the job, the running and des regostering should be done with flow scheduler
        f = flow_scheduler(self.des, self.qcloud, job, placement, 0.5, name)
        # f.simulate_run_dag_without_allocation()
        start_time = time.time()
        f.simulate_run_dag_with_allocation_simple_1()
        end_time_1 = time.time()
        running_time = end_time_1 - start_time
        print("running time is " + str(running_time) + " without deepcopy")
        f.simulate_run_dag_with_allocation_simple_fast()
        end_time_2 = time.time()
        running_time = end_time_2 - end_time_1
        print("running time is " + str(running_time) + " with deepcopy")
        # f.simulate_run_dag_with_allocation_simple_non_topo()
        # f.simulate_run_dag_with_allocation_average()

        print('hh')

    def _schedule_new(self, placement, job, name=None):
        if placement.partition == 1:
            selected_qpu = random.choice(placement.qpu_mapping)
            selected_qpu.allocate_job(job, job.circuit.n_qubits)
            used_qpu = {selected_qpu.qpuid: job.circuit.n_qubits}
            # self.registered_jobs.append((job, placement))
            single_finished_event = FinishedJob(placement.time, job, used_qpu, str(job.id) + ' finished')
            self.des.schedule_event(single_finished_event)
            return
        counts = Counter(placement.partition)
        for par_idx, qpu_id in placement.qpu_mapping[0].items():
            # modify the status of qpu
            qpu = self.qcloud.network.nodes[qpu_id]['qpu']
            print("count: ", counts[par_idx])
            qpu.allocate_job(job, counts[par_idx])
        self.registered_jobs.append((job, placement))



    def bfs_find_qpus(self, start_node, circuit_size):
        # use bfs to find qpus that can run this job(consider greedy qpu usage)
        visited = set()
        queue = [start_node]
        total_qubits = 0
        qpu_combination = []

        while queue:
            node = queue.pop(0)
            if node not in visited:
                visited.add(node)
                qpu = self.qcloud.network.nodes[node]['qpu']
                total_qubits += qpu.ncp_qubits
                qpu_combination.append(qpu)

                if total_qubits >= circuit_size:
                    return qpu_combination

                for neighbor in self.qcloud.network.neighbors(node):
                    if neighbor not in visited:
                        queue.append(neighbor)

        return None

    def find_simple_placement_bfs(self, job):
        size = job.circuit.n_qubits
        circuit = job.circuit
        comb = {}
        wig = self.convert_to_weighted_graph(circuit)
        # find all qpus that can run this job itself, if single qpu is enough just return qpu list
        if size < max([qpu.available_qubits for qpu in self.qcloud.qpus]):
            single_qpu_list = [qpu for qpu in self.qcloud.qpus if qpu.available_qubits >= size]
            comb[1] = single_qpu_list
            res = placement(job, 1, single_qpu_list, wig)
            res.communication_cost = 0
            single_graph = Graph(circuit)
            single_dag = single_graph.as_nx()
            res.time = len(nx.dag_longest_path(single_dag)) // 10
            res.modified_circuit = circuit  # select random qpu

            return [res]
        else:
            for i in range(math.ceil(size / self.qcloud.qpu_qubit_num),
                           min(len(self.qcloud.qpus), math.ceil(size / self.qcloud.qpu_qubit_num)) + 2):
                res = self.partition_circuit(i, wig)
                sublist_counts = [Counter(sublist) for sublist in res]
                comb[i] = res
            possible_placements = self.qcloud.find_placement_bfs(comb, wig)
            placement_list = []
            if not possible_placements:
                return None
            for key, value in possible_placements.items():
                if not value:
                    continue
                all_none = all(x is None for x in value)
                if all_none:
                    continue
                single_placement = placement(job, comb[key], value, wig)
                single_placement.get_communication_cost(wig)
                single_placement.get_time(comb[key])
                placement_list.append(single_placement)
            return placement_list
    
    def find_simple_placement_annealing( self, job):
        size = job.circuit.n_qubits
        circuit = job.circuit
        comb = {}
        wig = self.convert_to_weighted_graph(circuit)
        # find all qpus that can run this job itself, if single qpu is enough just return qpu list
        if size < max([qpu.available_qubits for qpu in self.qcloud.qpus]):
            single_qpu_list = [qpu for qpu in self.qcloud.qpus if qpu.available_qubits >= size]
            comb[1] = single_qpu_list
            res = placement(job, 1, single_qpu_list, wig)
            res.communication_cost = 0
            single_graph = Graph(circuit)
            single_dag = single_graph.as_nx()
            res.time = len(nx.dag_longest_path(single_dag)) // 10
            res.modified_circuit = circuit  # select random qpu

            return [res]
        else:

            possible_placements ,res = self.qcloud.sa_find_placement(job.circuit)
            placement_list = []
            if not possible_placements:
                return None
            for key, value in possible_placements.items():
                if not value:
                    continue
                all_none = all(x is None for x in value)
                if all_none:
                    continue
                single_placement = placement(job, comb[key], value, wig)
                single_placement.get_communication_cost(wig)
                single_placement.get_time(comb[key])
                placement_list.append(single_placement)
            return placement_list

    def find_simple_placement(self, job):
        #     finnd several possible placement for one single job.
        size = job.circuit.n_qubits
        circuit = job.circuit
        comb = {}
        wig = self.convert_to_weighted_graph(circuit)
        # find all qpus that can run this job itself, if single qpu is enough just return qpu list
        if size < max([qpu.available_qubits for qpu in self.qcloud.qpus]):
            single_qpu_list = [qpu for qpu in self.qcloud.qpus if qpu.available_qubits >= size]
            comb[1] = single_qpu_list
            res = placement(job, 1, single_qpu_list, wig)
            res.communication_cost = 0
            single_graph = Graph(circuit)
            single_dag = single_graph.as_nx()
            res.time = len(nx.dag_longest_path(single_dag)) // 10
            res.modified_circuit = circuit  # select random qpu

            return [res]
        else:
            for i in range(math.ceil(size / self.qcloud.qpu_qubit_num),
                           min(len(self.qcloud.qpus), math.ceil(size / self.qcloud.qpu_qubit_num)) + 2):
                res = self.partition_circuit(i, wig)
                sublist_counts = [Counter(sublist) for sublist in res]
                comb[i] = res
                # print(sublist_counts)ƒ
            possible_placements = self.qcloud.find_placement(comb, wig)
            placement_list = []
            if not possible_placements:
                return None
            for key, value in possible_placements.items():
                if not value:
                    continue
                all_none = all(x is None for x in value)
                if all_none:
                    continue
                single_placement = placement(job, comb[key], value, wig)
                single_placement.get_communication_cost(wig)
                single_placement.get_time(comb[key])
                placement_list.append(single_placement)
            return placement_list
        
    def find_simple_placement_rl(self, job):
        #     finnd several possible placement for one single job.
        size = job.circuit.n_qubits
        circuit = job.circuit
        comb = {}
        wig = self.convert_to_weighted_graph(circuit)
        # find all qpus that can run this job itself, if single qpu is enough just return qpu list
        if size < max([qpu.available_qubits for qpu in self.qcloud.qpus]):
            single_qpu_list = [qpu for qpu in self.qcloud.qpus if qpu.available_qubits >= size]
            comb[1] = single_qpu_list
            res = placement(job, 1, single_qpu_list, wig)
            res.communication_cost = 0
            single_graph = Graph(circuit)
            single_dag = single_graph.as_nx()
            res.time = len(nx.dag_longest_path(single_dag)) // 10
            res.modified_circuit = circuit  # select random qpu

            return [res]
        else:
            for i in range(math.ceil(size / self.qcloud.qpu_qubit_num),
                           min(len(self.qcloud.qpus), math.ceil(size / self.qcloud.qpu_qubit_num)) + 2):
                res = self.partition_circuit(i, wig)
                sublist_counts = [Counter(sublist) for sublist in res]
                comb[i] = res
                # print(sublist_counts)ƒ
            possible_placements = self.qcloud.find_placement(comb, wig)
            placement_list = []
            if not possible_placements:
                return None
            for key, value in possible_placements.items():
                if not value:
                    continue
                all_none = all(x is None for x in value)
                if all_none:
                    continue
                single_placement = placement(job, comb[key], value, wig)
                single_placement.get_communication_cost(wig)
                single_placement.get_time(comb[key])
                placement_list.append(single_placement)
            return placement_list

    def partition_circuit(self, n_parts, graph):
        node_to_int = {node: i for i, node in enumerate(graph.nodes())}
        int_to_node = {i: node for node, i in node_to_int.items()}  # Reverse mapping

        xadj = [0]
        adjncy = []
        eweights = []

        for node in graph.nodes():
            int_node = node_to_int[node]
            neighbors = [node_to_int[neighbor] for neighbor in graph.neighbors(node)]
            adjncy.extend(neighbors)
            xadj.append(xadj[-1] + len(neighbors))
            for int_neighbor in neighbors:
                orig_node = int_to_node[int_node]
                orig_neighbor = int_to_node[int_neighbor]
                if graph.has_edge(orig_node, orig_neighbor):
                    weight = graph[orig_node][orig_neighbor]['weight']
                    eweights.append(weight)
                else:
                    print(f"Edge not found: {orig_node} - {orig_neighbor}")
        # ufactor_list only contains 1 for now.
        ufactor_list = [20]
        # ufactor_list only contains 1 for now.
        xadj = np.array(xadj)
        adjncy = np.array(adjncy)
        eweights = np.array(eweights)
        res = []
        for ufactor in ufactor_list:
            opt = pymetis.Options()
            opt.ufactor = ufactor
            cutcount, part_vert = pymetis.part_graph(n_parts, xadj=xadj, adjncy=adjncy, eweights=eweights, options=opt)
            res.append(part_vert)

        return res

    def convert_to_weighted_graph(self, circuit):
        graph = nx.Graph()
        for command in circuit:
            op_type = command.op.type
            qubits = command.args
            # print(len(qubits))
            # if op_type in [OpType.CX, OpType.CZ, OpType.SWAP, OpType.ISWAP, OpType.CRz, OpType.CU1, OpType.CU3] and len(
            #         qubits) == 2:
            if op_type != OpType.Measure and len(
                    qubits) == 2:
                q1, q2 = qubits
                if not graph.has_edge(q1, q2):
                    graph.add_edge(q1, q2, weight=1)
                else:
                    graph[q1][q2]['weight'] += 1
        return graph

    # find all qpus that can run this job with one other job
    def find_multiple_placement(self):
        #     find optimal placement for all job in the queue
        pass

    def add_job(self, job):
        self.job_queue.append(job)
        self.job_queue.sort(key=lambda x: x.time, reverse=False)


def main():
    simple_test()


def simple_test():
    num_qpus = 10
    probability = 0.5
    cloud = qCloud(num_qpus, create_random_topology, probability)
    nx.draw(cloud.network, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=15)
    a = cloud.network.nodes[0]['qpu']
    # Show the plot
    plt.show()
    # large_job = job_generator().generate_large_circuit_job()
    job = job_generator().generate_fixed_job("/Users/mac/Desktop/qCloud/circuit/small/vqe_n4/vqe_n4_transpiled.qasm")
    scheduler = job_scheduler([job], None, cloud)
    scheduler.schedule()



def test_keep_generating():
    num_qpus = 20
    probability = 0.5
    cloud = qCloud(num_qpus, create_random_topology, probability)
    # probability define the possibility to generate different size jobs, large, medium, small
    proabability = [0.5, 0.5, 0]
    time_frame = 100
    des = DES(cloud)
    for i in range(10):
        job_queue = job_generator().generate_job(10, time_frame, i, proabability)
        generating_event = generatingJob(i * time_frame, job_queue)
        des.schedule_event(generating_event)
    des.run()


def test_schedule_several():
    num_qpus = 20
    probability = 0.5
    cloud = qCloud(num_qpus, create_random_topology, probability)

    # probability define the possibility to generate different size jobs, large, medium, small
    proabability = [0.5, 0.5, 0]
    # generate fixed job list
    job1 = job_generator().generate_fixed_job(
        "/Users/mac/Desktop/qCloud/circuit/medium/sat_n11/sat_n11_transpiled.qasm", 0)
    job2 = job_generator().generate_fixed_job(
        "/Users/mac/Desktop/qCloud/circuit/medium/dnn_n16/dnn_n16_transpiled.qasm", 0)
    job3 = job_generator().generate_fixed_job(
        "/Users/mac/Desktop/qCloud/circuit/medium/qf21_n15/qf21_n15_transpiled.qasm", 0)
    job4 = job_generator().generate_fixed_job("/Users/mac/QASMBench/medium/seca_n11/seca_n11_transpiled.qasm", 0)
    job5 = job_generator().generate_fixed_job(
        "/Users/mac/Desktop/qCloud/circuit/medium/sat_n11/sat_n11_transpiled.qasm", 0)

    job_queue = [job1, job2, job3, job4, job5]
    job_queue = job_generator().generate_random_job_list(10, proabability)
    des = DES()
    scheduler = job_scheduler(job_queue, des, cloud)

    scheduler.schedule()
    pass


if __name__ == '__main__':
    main()
