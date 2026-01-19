# flow scheduler defines how to schedule flow
import copy
import math
from itertools import combinations, permutations
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import random
from functools import lru_cache
import networkx as nx
from des import Event, FinishedJob
from utils import compute_probability, compute_probability_with_distance
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
import concurrent.futures

p = 0.4
sys.setrecursionlimit(1000000)


def _is_allocatabale(to_go_pool, resources):
    # check we can still allocate to any group in to_go_pool
    # for i, value in to_go_pool.items():
    #     for key in value.keys():
    #         qpu1, qpu2 = key
    #         if resources[qpu1] > 0 and resources[qpu2] > 0:
    #             return True
    for key, value in to_go_pool.items():
        qpu1, qp2 = key
        if resources[qpu1] > 0 and resources[qp2] > 0:
            return True

    return False
def allocate_resources_between_sets(tasks, machine_resources):
    # Initialize the available resources for each machine set
    set_available_resources = {
        machine_set: min(machine_resources[machine] for machine in machine_set)
        for machine_set in tasks.keys() if tasks[machine_set]
    }

    # Initialize allocations
    allocations = {machine_set: 0 for machine_set in tasks.keys() if tasks[machine_set]}

    # Flag to indicate if we can still allocate resources
    can_allocate = True

    while can_allocate:
        # print(allocations)
        can_allocate = False  # Assume no allocation is possible until proven otherwise

        for machine_set, jobs in tasks.items():
            if not jobs or set_available_resources[machine_set] <= 0:
                continue

            # Determine the allocation for this iteration
            job_count = len(jobs)
            allocation_per_job = min(1, set_available_resources[machine_set] / job_count)
            total_allocation = allocation_per_job * job_count

            if total_allocation > 0:
                can_allocate = True  # Allocation is possible, continue in the next iteration

                # Update allocations and available resources
                allocations[machine_set] += total_allocation
                for machine in machine_set:
                    machine_resources[machine] -= total_allocation

                # Update available resources for all sets sharing these machines
                for other_set in set_available_resources:
                    if any(machine in other_set for machine in machine_set):
                        set_available_resources[other_set] = min(machine_resources[m] for m in other_set)

    return allocations
def allocate_resources_between_sets_priority(tasks, machine_resources, task_priorities):
    # Initialize the available resources for each machine set
    set_available_resources = {
        machine_set: min(machine_resources[machine] for machine in machine_set)
        for machine_set in tasks.keys() if tasks[machine_set]
    }

    # Calculate the total priority for each set
    set_total_priorities = {
        machine_set: sum(task_priorities[job] for job in jobs)
        for machine_set, jobs in tasks.items() if jobs
    }

    # Initialize allocations
    allocations = {machine_set: 0 for machine_set in tasks.keys() if tasks[machine_set]}

    # Flag to indicate if we can still allocate resources
    can_allocate = True

    while can_allocate:
        can_allocate = False  # Assume no allocation is possible until proven otherwise

        for machine_set, jobs in tasks.items():
            if not jobs or set_available_resources[machine_set] <= 0:
                continue

            # Calculate the allocation for this set based on total priorities
            total_set_priority = set_total_priorities[machine_set]
            allocation_for_set = (set_available_resources[machine_set] * total_set_priority) / sum(
                set_total_priorities.values())

            if allocation_for_set > 0:
                can_allocate = True  # Allocation is possible, continue in the next iteration

                # Update allocations and available resources
                allocations[machine_set] += allocation_for_set
                for machine in machine_set:
                    machine_resources[machine] -= allocation_for_set

                # Update available resources for all sets sharing these machines
                for other_set in set_available_resources:
                    if any(machine in other_set for machine in machine_set):
                        set_available_resources[other_set] = min(machine_resources[m] for m in other_set)

    return allocations
def allocate_resources_random(to_go_pool, resources):
#     randomly assign resources to the nodes in front layer when possible
    allocation_result = {value:0 for key, values in to_go_pool.items() for value in values}
    while _is_allocatabale(to_go_pool, resources):
        key = random.choice(list(to_go_pool.keys()))
        node = random.choice(to_go_pool[key])
        allocation_result[node] +=1
        resources[key[0]] -=1
        resources[key[1]] -=1
    return allocation_result
def allocate_resources_greedy(to_go_pool, resources, priority):
    # allocate resources to the nodes in front layer when possible
    allocation_result = {value:0 for key, values in to_go_pool.items() for value in values}
    allocated_node = set()
    while _is_allocatabale(to_go_pool, resources):
        candidates = [value for key, values in to_go_pool.items() for value in values if value not in allocated_node]
        most_importan_ndoe= max(candidates, key=lambda x: priority[x])
        # get qpu of the most important node
        qpu1, qpu2 = [qpu for qpu in to_go_pool.keys() if most_importan_ndoe in to_go_pool[qpu]][0]
        maximum_resources = min(resources[qpu1], resources[qpu2])
        allocation_result[most_importan_ndoe] +=maximum_resources
        resources[qpu1] -= maximum_resources
        resources[qpu2] -= maximum_resources
        allocated_node.add(most_importan_ndoe)
    return allocation_result
def find_non_overlapping_indices(sets):
    non_overlapping_indices = []

    for i, current_set in enumerate(sets):
        overlap = False
        for j, other_set in enumerate(sets):
            if i != j and not current_set.isdisjoint(other_set):
                overlap = True
                break
        if not overlap:
            non_overlapping_indices.append(i)

    return non_overlapping_indices


def group_tasks_by_overlap(tasks):
    # Create a graph
    G = nx.Graph()

    # Add nodes for each task
    for i in range(len(tasks)):
        G.add_node(i)

    # Add edges between nodes if their sets intersect
    for i in range(len(tasks)):
        for j in range(i + 1, len(tasks)):
            if tasks[i].intersection(tasks[j]):
                G.add_edge(i, j)

    # Find connected components (groups of overlapping tasks)
    groups = list(nx.connected_components(G))

    # Return the groups with indices
    result = []
    for group in groups:
        result.append(sorted(list(group)))

    return result


def find_two_farthest_parents(graph, target_node):
    # Function to trace back the farthest parent for a specific port
    def trace_farthest_parent(node, port, visited):
        if node in visited:
            return node
        visited.add(node)

        for pred in graph.predecessors(node):
            edge_data_list = graph.get_edge_data(pred, node).values()
            for data in edge_data_list:
                if data['tgt_port'] == port:
                    return trace_farthest_parent(pred, data['src_port'], visited)
        return node

    # Find the two farthest ancestors based on the ports
    farthest_parents = set()
    for a, b, edge_data in graph.in_edges(target_node, data=True):
        farthest_parent = trace_farthest_parent(target_node, edge_data['tgt_port'], set())
        farthest_parents.add(farthest_parent)
        # if len(farthest_parents) == 2:
        #     break

    return list(farthest_parents)


class flow_scheduler_1:
    def __init__(self, des, qcloud, epr_p, name):
        self.des = des
        self.qcloud = qcloud
        self.EPR_p = epr_p
        self.name = name
    def allocate_resources_between_sets_priority(self, tasks, machine_resources, task_priorities):
        # Initialize the available resources for each machine set
        set_available_resources = {
            machine_set: min(machine_resources[machine] for machine in machine_set)
            for machine_set in tasks.keys() if tasks[machine_set]
        }

        # Calculate the total priority for each set
        set_total_priorities = {
            machine_set: sum(task_priorities[job] for job in jobs)
            for machine_set, jobs in tasks.items() if jobs
        }

        # Initialize allocations
        allocations = {machine_set: 0 for machine_set in tasks.keys() if tasks[machine_set]}

        # Flag to indicate if we can still allocate resources
        can_allocate = True

        while can_allocate:
            can_allocate = False  # Assume no allocation is possible until proven otherwise

            for machine_set, jobs in tasks.items():
                if not jobs or set_available_resources[machine_set] <= 0:
                    continue

                # Calculate the allocation for this set based on total priorities
                total_set_priority = set_total_priorities[machine_set]
                allocation_for_set = (set_available_resources[machine_set] * total_set_priority) / sum(
                    set_total_priorities.values())

                if allocation_for_set > 0:
                    can_allocate = True  # Allocation is possible, continue in the next iteration

                    # Update allocations and available resources
                    allocations[machine_set] += allocation_for_set
                    for machine in machine_set:
                        machine_resources[machine] -= allocation_for_set

                    # Update available resources for all sets sharing these machines
                    for other_set in set_available_resources:
                        if any(machine in other_set for machine in machine_set):
                            set_available_resources[other_set] = min(machine_resources[m] for m in other_set)

        return allocations



    def run_multi(self):
        # Step 1: Detect whether multiple job runs on the same QPU, find the QPU that has no conflict with others
        print("start flow scheduling")

        # Find the job in the queue that has no QPU mapping conflict with others
        qpu_mapping_list = [job[1].qpu_mapping[0] for job in self.des.scheduler.registered_jobs]
        qpu_set_list = [set(qpu_mapping.values()) for qpu_mapping in qpu_mapping_list]

        # Find the QPU that has no conflict with others
        non_conflict_job_indices = find_non_overlapping_indices(qpu_set_list)
        grouped_tasks = group_tasks_by_overlap(qpu_set_list)

        # Calculate the number of qubits used in each task in each QPU
        print(grouped_tasks)

        # Sort the grouped tasks by the length of each group
        grouped_tasks = sorted(grouped_tasks, key=lambda x: len(x), reverse=False)

        # Use ProcessPoolExecutor for multi-processing
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Submit each group to be processed concurrently
            futures = [executor.submit(self.simulate_run_group, group) for group in grouped_tasks]

            # Optionally, wait for all futures to complete and get their results
            results = [future.result() for future in concurrent.futures.as_completed(futures)]





    def run(self):
        #         step 1:detect whether multiple job runs on the same qpu, find the qpu that has no conflict with others
        print("start flow scheduling")
        #         find the job in the queue that has no qpu mapping conflict with others
        qpu_mapping_list = [job[1].qpu_mapping[0] for job in self.des.scheduler.registered_jobs]
        qpu_set_list = [set(qpu_mapping.values()) for qpu_mapping in qpu_mapping_list]
        #         find the qpu that has no conflict with others
        non_conflict_job_indices = find_non_overlapping_indices(qpu_set_list)
        grouped_tasks = group_tasks_by_overlap(qpu_set_list)
        # calculate the number of qubits used in each task in each qpu

        print(grouped_tasks)
        # map(self.simulate_run_group, grounped_tasks)
        #   apply simulate_run_group to each group
        # sort task in group by the longest job in the group
        # grouped_tasks = sorted(grouped_tasks, key=lambda x: max([job for job in se]), reverse=False)
        grouped_tasks = sorted(grouped_tasks, key=lambda x: len(x), reverse= True)
        for group in grouped_tasks:
            print(group)
            self.simulate_run_group(group)
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     # Submit each group to be processed concurrently
        #     futures = [executor.submit(self.simulate_run_group, group) for group in grouped_tasks]
        #
        #     # Optionally, wait for all futures to complete and get their results
        #     results = [future.result() for future in concurrent.futures.as_completed(futures)]
        print("finished flow scheduling")
        self.des.scheduler.registered_jobs.clear()
    def _simulate_run_single(self, group):
        # this fast version only retrieve dag data once and thus faster
        # define a table to record whther each gate is ready to be executed
        # change key of dict from qubit to str of it
        placement = self.des.scheduler.registered_jobs[group[0]][1]
        remote_dag = placement.remote_dag
        node_to_int = {str(node): i for i, node in enumerate(placement.wig.nodes())}
        current_step = 0
        # to_go_table ={node: False for node in self.remote_dag.nodes()}
        graph_to_process = copy.deepcopy(remote_dag )
        print("finished deepcopy")
        priority = self._compute_priority_1(remote_dag )
        print("finished priority")
        input_layer = [node for node in remote_dag .nodes() if
                       remote_dag .nodes._nodes[node]['desc'] == 'Input']
        # status table records when each node is finished
        status_table = {node: 0 for node in remote_dag.nodes()}
        for node in input_layer:
            # to_go_table[node] = True
            graph_to_process.remove_node(node)
        # initialize the front layer,at this step  graph_to_process only doesn't contain input layer

        output_layer = [node for node in remote_dag.nodes() if
                        remote_dag.nodes._nodes[node]['desc'] == 'Output']
        for node in output_layer:
            graph_to_process.remove_node(node)
        to_go_table = {node: 'unready' for node in remote_dag.nodes()}
        used_qpus = sorted(list(set(placement.qpu_mapping[0].values())))
        remote_combo = list(combinations(used_qpus, 2))
        for node in input_layer:
            to_go_table[node] = 'finished'
        parents = {node: [] for node in graph_to_process.nodes()}
        parents_qpu = {node: [] for node in graph_to_process.nodes()}
        # first get the parent node for each node,
        # here the index of parent is the index in the dag
        # parents hold the partion for each node
        # {23 : [0,1]} means node 23 has two parents, one is in partition 0, the other is in partition 1
        for node in graph_to_process.nodes():
            parents[node] = find_two_farthest_parents(remote_dag, node)
        print("finished parent step 1")
        data_type = placement.modified_circuit._dag_data[6]
        for node in parents.keys():
            # get the qubit index for each parent in the circuit
            # eg  0  -----> q[0]
            qubit_index = [data_type[node] for node in parents[node]]
            parents[node] = [placement.partition[node_to_int[node]] for node in qubit_index]
            parents_partition = [placement.partition[node_to_int[node]] for node in qubit_index]
            parents_qpu[node] = sorted([placement.qpu_mapping[0][par] for par in parents_partition])

        
        nodes_to_deletes= []
        for node in graph_to_process.nodes():
            if parents_qpu[node][0] == parents_qpu[node][1]:
                nodes_to_deletes.append(node)
                to_go_table[node] = 'finished'
        for node in nodes_to_deletes:
            graph_to_process.remove_node(node)
        
        front_layer = [node for node in graph_to_process.nodes() if
                       graph_to_process.in_degree(node) == 0 and graph_to_process.out_degree(node) != 0]

        # filiter out front layer nodes that have the same qpu
        # for node in front_layer:
        print("finished parent step 2,  start flow scheduling",  "job_name" , self.des.scheduler.registered_jobs[group[0]][0].name)
        while len(graph_to_process.nodes()) != 0:
            current_step += 1
            node_to_delet = []
            node_to_add = []
            to_go_pool = {combo: [] for combo in remote_combo}
            # resources {partition_id : #number of communication qubits}
            # resources denotes the number of communication qubits for each partition，
            # qpu[1] is the partition id, qpu[0] is the qpu id
            resources = {qpu_idx: self.qcloud.network.nodes[qpu_idx]['qpu'].ncm_qubits for par_idx, qpu_idx in
                         placement.qpu_mapping[0].items()}
            allocation_result = {node: 0 for node in front_layer}
            competing_sets = {}
            for node in front_layer:
                #     check which queue should node go to
                for combo in to_go_pool.keys():
                    if parents_qpu[node] == list(combo) or parents_qpu[node] == list(combo)[::-1]:
                        to_go_pool[combo].append(node)
            to_go_pool = {key: value for key, value in to_go_pool.items() if value}
            # get inter-set allocation
            inter_sets_allocations = allocate_resources_between_sets_priority(to_go_pool, resources, priority)
            # allocate resources within machine sets
            for comb, nodes in to_go_pool.items():
                if nodes:
                    total_resources = inter_sets_allocations[comb]
                    total_prioriy = sum([priority[node] for node in nodes])
                    for node in nodes:
                        allocation_result[node] = round(total_resources * priority[node] / total_prioriy)
                        total_resources -= allocation_result[node]
                    if total_resources > 0:
                        sorted_nodes = sorted(nodes, key=lambda n: priority[n], reverse=True)
                        for node in sorted_nodes:
                            if total_resources > 0:
                                allocation_result[node] += 1
                                total_resources -= 1

            # print(allocation_result, to_go_pool,resources)
            # print(allocation_result, to_go_pool,resources,self.des.scheduler.registered_jobs[group[0]][0].name,len(graph_to_process.nodes()))
            for node in front_layer:
                # print(remote_dag.nodes[node]['desc'])
                parent_nodes = set(parents[node])
                if len(parent_nodes) == 2:
                    qpu_parent_nodes = [placement.qpu_mapping[0][par] for par in parent_nodes]
                    distance = nx.shortest_path_length(self.qcloud.network, qpu_parent_nodes[0],
                                                       qpu_parent_nodes[1])
                    seed = random.uniform(0, 1)
                    p_topo = compute_probability_with_distance(allocation_result[node], self.EPR_p, distance)
                    # p_topo = 1
                    if seed >= 1 - p_topo:
                        status_table[node] = current_step
                        node_to_delet.append(node)
                        to_go_table[node] = 'finished'
                        for succ in graph_to_process.succ[node]:
                            if all([to_go_table[pred] == 'finished' for pred in graph_to_process.pred[succ]]):
                                to_go_table[succ] = 'ready'
                                if node not in output_layer:
                                    node_to_add.append(succ)

                elif len(parent_nodes) == 1:
                    pass

                else:
                    pass
            for node in node_to_delet:
                front_layer.remove(node)
                graph_to_process.remove_node(node)
            for node in node_to_add:
                front_layer.append(node)


        curren_time = self.des.current_time + current_step
        # given partition and qpu mapping, compute the qpumapping
        count = Counter(placement.partition)
        qpu_mapping = placement.qpu_mapping[0]
        used_qpu = {value: count[key] for key, value in qpu_mapping.items()}
        tmp = self.des.scheduler.registered_jobs[group[0]][0]
        print(tmp)
        # job scheduler shouldn't put event in the des directly, it should return the event and let des handle it
        finished_event = FinishedJob(curren_time, self.des.scheduler.registered_jobs[group[0]][0], used_qpu,
                                     str(self.des.scheduler.registered_jobs[group[0]][0].id) + ' finished+ withallocation' + self.name,  self.des.scheduler.registered_jobs[group[0]][1])
        self.des.schedule_event(finished_event)
        return current_step

    def _simulate_run_single_average(self, group):
        # this fast version only retrieve dag data once and thus faster
        # define a table to record whther each gate is ready to be executed
        # change key of dict from qubit to str of it
        placement = self.des.scheduler.registered_jobs[group[0]][1]
        remote_dag = placement.remote_dag
        node_to_int = {str(node): i for i, node in enumerate(placement.wig.nodes())}
        current_step = 0
        # to_go_table ={node: False for node in self.remote_dag.nodes()}
        graph_to_process = copy.deepcopy(remote_dag )
        print("finished deepcopy")
        priority = self._compute_priority_1(remote_dag )
        print("finished priority")
        input_layer = [node for node in remote_dag .nodes() if
                       remote_dag .nodes._nodes[node]['desc'] == 'Input']
        # status table records when each node is finished
        status_table = {node: 0 for node in remote_dag.nodes()}
        for node in input_layer:
            # to_go_table[node] = True
            graph_to_process.remove_node(node)
        # initialize the front layer,at this step  graph_to_process only doesn't contain input layer
        front_layer = [node for node in graph_to_process.nodes() if
                       graph_to_process.in_degree(node) == 0 and graph_to_process.out_degree(node) != 0]
        output_layer = [node for node in remote_dag.nodes() if
                        remote_dag.nodes._nodes[node]['desc'] == 'Output']
        for node in output_layer:
            graph_to_process.remove_node(node)
        to_go_table = {node: 'unready' for node in remote_dag.nodes()}
        used_qpus = sorted(list(set(placement.qpu_mapping[0].values())))
        remote_combo = list(combinations(used_qpus, 2))
        for node in input_layer:
            to_go_table[node] = 'finished'
        parents = {node: [] for node in graph_to_process.nodes()}
        parents_qpu = {node: [] for node in graph_to_process.nodes()}
        # first get the parent node for each node,
        # here the index of parent is the index in the dag
        # parents hold the partion for each node
        # {23 : [0,1]} means node 23 has two parents, one is in partition 0, the other is in partition 1
        for node in graph_to_process.nodes():
            parents[node] = find_two_farthest_parents(remote_dag, node)
        print("finished parent step 1")
        data_type = placement.modified_circuit._dag_data[6]
        for node in parents.keys():
            # get the qubit index for each parent in the circuit
            # eg  0  -----> q[0]
            qubit_index = [data_type[node] for node in parents[node]]
            parents[node] = [placement.partition[node_to_int[node]] for node in qubit_index]
            parents_partition = [placement.partition[node_to_int[node]] for node in qubit_index]
            parents_qpu[node] = sorted([placement.qpu_mapping[0][par] for par in parents_partition])
        print("finished parent step 2")
        a = {i: 0 for i in range(1, 10)}
        while len(graph_to_process.nodes()) != 0:
            current_step += 1
            node_to_delet = []
            node_to_add = []
            to_go_pool = {combo: [] for combo in remote_combo}
            # resources {partition_id : #number of communication qubits}
            # resources denotes the number of communication qubits for each partition，
            # qpu[1] is the partition id, qpu[0] is the qpu id
            resources = {qpu_idx: self.qcloud.network.nodes[qpu_idx]['qpu'].ncm_qubits for par_idx, qpu_idx in
                         placement.qpu_mapping[0].items()}
            allocation_result = {node: 0 for node in front_layer}
            competing_sets = {}
            for node in front_layer:
                #     check which queue should node go to
                if parents_qpu[node][0] == parents_qpu[node][1]:
                    graph_to_process.remove_node(node)

                for combo in to_go_pool.keys():
                    if parents_qpu[node] == list(combo) or parents_qpu[node] == list(combo)[::-1]:
                        to_go_pool[combo].append(node)
            to_go_pool = {key: value for key, value in to_go_pool.items() if value}
            # get inter-set allocation
            inter_sets_allocations = allocate_resources_between_sets(to_go_pool,resources)
            # allocate resources within machine sets
            if all([value == 0 for value in inter_sets_allocations.values()]):
                print("bug")
            for comb, nodes in to_go_pool.items():
                if nodes:
                    total_resources = inter_sets_allocations[comb]
                    share  = math.ceil(total_resources / len(nodes))
                    remaining_resources = total_resources
                    for node in nodes:
                        if remaining_resources > share:
                            allocation_result[node] = share
                            remaining_resources -= share
                        else:
                            allocation_result[node] = remaining_resources
                            remaining_resources = 0

            print(allocation_result,to_go_pool)
            # print("hh")
            for node in front_layer:
                # print(remote_dag.nodes[node]['desc'])
                parent_nodes = set(parents[node])
                if len(parent_nodes) == 2:
                    qpu_parent_nodes = [placement.qpu_mapping[0][par] for par in parent_nodes]
                    distance = nx.shortest_path_length(self.qcloud.network, qpu_parent_nodes[0],
                                                       qpu_parent_nodes[1])
                    if distance > 1:
                        if distance in a:
                            a[distance] += 1
                        else:
                            a[distance] = 0
                    else:
                        a[1] += 1
                        # print("hh")
                    seed = random.uniform(0, 1)
                    p_topo = compute_probability_with_distance(allocation_result[node], self.EPR_p, distance)
                    # p_topo = 1
                    if seed >= 1 - p_topo:
                        status_table[node] = current_step
                        node_to_delet.append(node)
                        to_go_table[node] = 'finished'
                        for succ in graph_to_process.succ[node]:
                            if all([to_go_table[pred] == 'finished' for pred in graph_to_process.pred[succ]]):
                                to_go_table[succ] = 'ready'
                                if node not in output_layer:
                                    node_to_add.append(succ)

                elif len(parent_nodes) == 1:
                    pass

                else:
                    pass
            for node in node_to_delet:
                front_layer.remove(node)
                graph_to_process.remove_node(node)
            for node in node_to_add:
                front_layer.append(node)
        print(a)

        curren_time = self.des.current_time + current_step
        # given partition and qpu mapping, compute the qpumapping
        count = Counter(placement.partition)
        qpu_mapping = placement.qpu_mapping[0]
        used_qpu = {value: count[key] for key, value in qpu_mapping.items()}
        tmp = self.des.scheduler.registered_jobs[group[0]][0]
        print(tmp)
        # job scheduler shouldn't put event in the des directly, it should return the event and let des handle it
        # finished_event = FinishedJob(curren_time, self.des.scheduler.registered_jobs[group[0]][0], used_qpu,
        #                              str(self.des.scheduler.registered_jobs[group[0]][0].id) + ' finished+ withallocation' + self.name)
        finished_event = FinishedJob(curren_time, self.des.scheduler.registered_jobs[group[0]][0], used_qpu,
                                     str(self.des.scheduler.registered_jobs[group[0]][0].id) + ' finished+ withallocation' + self.name,  self.des.scheduler.registered_jobs[group[0]][1])
        self.des.schedule_event(finished_event)
        return current_step
    

    def _simulate_run_single_random(self, group):
        placement = self.des.scheduler.registered_jobs[group[0]][1]
        remote_dag = placement.remote_dag
        node_to_int = {str(node): i for i, node in enumerate(placement.wig.nodes())}
        current_step = 0
        # to_go_table ={node: False for node in self.remote_dag.nodes()}
        graph_to_process = copy.deepcopy(remote_dag )
        print("finished deepcopy")
        priority = self._compute_priority_1(remote_dag )
        print("finished priority")
        input_layer = [node for node in remote_dag .nodes() if
                       remote_dag .nodes._nodes[node]['desc'] == 'Input']
        # status table records when each node is finished
        status_table = {node: 0 for node in remote_dag.nodes()}
        for node in input_layer:
            # to_go_table[node] = True
            graph_to_process.remove_node(node)
        # initialize the front layer,at this step  graph_to_process only doesn't contain input layer
        front_layer = [node for node in graph_to_process.nodes() if
                       graph_to_process.in_degree(node) == 0 and graph_to_process.out_degree(node) != 0]
        output_layer = [node for node in remote_dag.nodes() if
                        remote_dag.nodes._nodes[node]['desc'] == 'Output']
        for node in output_layer:
            graph_to_process.remove_node(node)
        to_go_table = {node: 'unready' for node in remote_dag.nodes()}
        used_qpus = sorted(list(set(placement.qpu_mapping[0].values())))
        remote_combo = list(combinations(used_qpus, 2))
        for node in input_layer:
            to_go_table[node] = 'finished'
        parents = {node: [] for node in graph_to_process.nodes()}
        parents_qpu = {node: [] for node in graph_to_process.nodes()}
        # first get the parent node for each node,
        # here the index of parent is the index in the dag
        # parents hold the partion for each node
        # {23 : [0,1]} means node 23 has two parents, one is in partition 0, the other is in partition 1
        for node in graph_to_process.nodes():
            parents[node] = find_two_farthest_parents(remote_dag, node)
        print("finished parent step 1")
        data_type = placement.modified_circuit._dag_data[6]
        for node in parents.keys():
            # get the qubit index for each parent in the circuit
            # eg  0  -----> q[0]
            qubit_index = [data_type[node] for node in parents[node]]
            parents[node] = [placement.partition[node_to_int[node]] for node in qubit_index]
            parents_partition = [placement.partition[node_to_int[node]] for node in qubit_index]
            parents_qpu[node] = sorted([placement.qpu_mapping[0][par] for par in parents_partition])
        print("finished parent step 2")
        a = {i: 0 for i in range(1, 10)}
        while len(graph_to_process.nodes()) != 0:
            current_step += 1
            node_to_delet = []
            node_to_add = []
            to_go_pool = {combo: [] for combo in remote_combo}
            # resources {partition_id : #number of communication qubits}
            # resources denotes the number of communication qubits for each partition，
            # qpu[1] is the partition id, qpu[0] is the qpu id
            resources = {qpu_idx: self.qcloud.network.nodes[qpu_idx]['qpu'].ncm_qubits for par_idx, qpu_idx in
                         placement.qpu_mapping[0].items()}
            competing_sets = {}
            for node in front_layer:
                #     check which queue should node go to

                for combo in to_go_pool.keys():
                    if parents_qpu[node] == list(combo) or parents_qpu[node] == list(combo)[::-1]:
                        to_go_pool[combo].append(node)
            to_go_pool = {key: value for key, value in to_go_pool.items() if value}
            # get inter-set allocation
            allocation_result = allocate_resources_random(to_go_pool, resources)
            # print(allocation_result,to_go_pool, front_layer)
            # print("hh")
            for node in front_layer:
                # print(remote_dag.nodes[node]['desc'])
                parent_nodes = set(parents[node])
                if len(parent_nodes) == 2:
                    qpu_parent_nodes = [placement.qpu_mapping[0][par] for par in parent_nodes]
                    distance = nx.shortest_path_length(self.qcloud.network, qpu_parent_nodes[0],
                                                       qpu_parent_nodes[1])
                    if distance > 1:
                        if distance in a:
                            a[distance] += 1
                        else:
                            a[distance] = 0
                    else:
                        a[1] += 1
                        # print("hh")
                    seed = random.uniform(0, 1)
                    p_topo = compute_probability_with_distance(allocation_result[node], self.EPR_p, distance)
                    # p_topo = 1
                    # print(seed, p_topo,allocation_result[node], self.EPR_p, distance )
                    if seed >= 1 - p_topo:
                        status_table[node] = current_step
                        node_to_delet.append(node)
                        to_go_table[node] = 'finished'
                        for succ in graph_to_process.succ[node]:
                            if all([to_go_table[pred] == 'finished' for pred in graph_to_process.pred[succ]]):
                                to_go_table[succ] = 'ready'
                                if node not in output_layer:
                                    node_to_add.append(succ)

                elif len(parent_nodes) == 1:
                    pass

                else:
                    pass
            for node in node_to_delet:
                front_layer.remove(node)
                graph_to_process.remove_node(node)
            for node in node_to_add:
                front_layer.append(node)
        print(a)

        curren_time = self.des.current_time + current_step
        # given partition and qpu mapping, compute the qpumapping
        count = Counter(placement.partition)
        qpu_mapping = placement.qpu_mapping[0]
        used_qpu = {value: count[key] for key, value in qpu_mapping.items()}
        tmp = self.des.scheduler.registered_jobs[group[0]][0]
        print(tmp)
        # job scheduler shouldn't put event in the des directly, it should return the event and let des handle it
        # finished_event = FinishedJob(curren_time, self.des.scheduler.registered_jobs[group[0]][0], used_qpu,
        #                              str(self.des.scheduler.registered_jobs[group[0]][0].id) + ' finished+ withallocation' + self.name)
        finished_event = FinishedJob(curren_time, self.des.scheduler.registered_jobs[group[0]][0], used_qpu,
                                     str(self.des.scheduler.registered_jobs[group[0]][0].id) + ' finished+ withallocation' + self.name,  self.des.scheduler.registered_jobs[group[0]][1])
        self.des.schedule_event(finished_event)
        return current_step

    def _simulate_run_single_greedy(self, group):
        placement = self.des.scheduler.registered_jobs[group[0]][1]
        remote_dag = placement.remote_dag
        node_to_int = {str(node): i for i, node in enumerate(placement.wig.nodes())}
        current_step = 0
        # to_go_table ={node: False for node in self.remote_dag.nodes()}
        graph_to_process = copy.deepcopy(remote_dag )
        print("finished deepcopy")
        priority = self._compute_priority_1(remote_dag )
        print("finished priority")
        input_layer = [node for node in remote_dag .nodes() if
                       remote_dag .nodes._nodes[node]['desc'] == 'Input']
        # status table records when each node is finished
        status_table = {node: 0 for node in remote_dag.nodes()}
        for node in input_layer:
            # to_go_table[node] = True
            graph_to_process.remove_node(node)
        # initialize the front layer,at this step  graph_to_process only doesn't contain input layer
        front_layer = [node for node in graph_to_process.nodes() if
                       graph_to_process.in_degree(node) == 0 and graph_to_process.out_degree(node) != 0]
        output_layer = [node for node in remote_dag.nodes() if
                        remote_dag.nodes._nodes[node]['desc'] == 'Output']
        for node in output_layer:
            graph_to_process.remove_node(node)
        to_go_table = {node: 'unready' for node in remote_dag.nodes()}
        used_qpus = sorted(list(set(placement.qpu_mapping[0].values())))
        remote_combo = list(combinations(used_qpus, 2))
        for node in input_layer:
            to_go_table[node] = 'finished'
        parents = {node: [] for node in graph_to_process.nodes()}
        parents_qpu = {node: [] for node in graph_to_process.nodes()}
        # first get the parent node for each node,
        # here the index of parent is the index in the dag
        # parents hold the partion for each node
        # {23 : [0,1]} means node 23 has two parents, one is in partition 0, the other is in partition 1
        for node in graph_to_process.nodes():
            parents[node] = find_two_farthest_parents(remote_dag, node)
        print("finished parent step 1")
        data_type = placement.modified_circuit._dag_data[6]
        for node in parents.keys():
            # get the qubit index for each parent in the circuit
            # eg  0  -----> q[0]
            qubit_index = [data_type[node] for node in parents[node]]
            parents[node] = [placement.partition[node_to_int[node]] for node in qubit_index]
            parents_partition = [placement.partition[node_to_int[node]] for node in qubit_index]
            parents_qpu[node] = sorted([placement.qpu_mapping[0][par] for par in parents_partition])
        print("finished parent step 2")
        a = {i: 0 for i in range(1, 10)}
        while len(graph_to_process.nodes()) != 0:
            current_step += 1
            node_to_delet = []
            node_to_add = []
            to_go_pool = {combo: [] for combo in remote_combo}
            # resources {partition_id : #number of communication qubits}
            # resources denotes the number of communication qubits for each partition，
            # qpu[1] is the partition id, qpu[0] is the qpu id
            resources = {qpu_idx: self.qcloud.network.nodes[qpu_idx]['qpu'].ncm_qubits for par_idx, qpu_idx in
                         placement.qpu_mapping[0].items()}
            competing_sets = {}
            for node in front_layer:
                #     check which queue should node go to

                for combo in to_go_pool.keys():
                    if parents_qpu[node] == list(combo) or parents_qpu[node] == list(combo)[::-1]:
                        to_go_pool[combo].append(node)
            to_go_pool = {key: value for key, value in to_go_pool.items() if value}
            # get inter-set allocation
            allocation_result = allocate_resources_greedy(to_go_pool, resources, priority)
            # print(allocation_result,to_go_pool.keys())
            # print("hh")
            for node in front_layer:
                # print(remote_dag.nodes[node]['desc'])
                parent_nodes = set(parents[node])
                if len(parent_nodes) == 2:
                    qpu_parent_nodes = [placement.qpu_mapping[0][par] for par in parent_nodes]
                    distance = nx.shortest_path_length(self.qcloud.network, qpu_parent_nodes[0],
                                                       qpu_parent_nodes[1])
                    if distance > 1:
                        if distance in a:
                            a[distance] += 1
                        else:
                            a[distance] = 0
                    else:
                        a[1] += 1
                        # print("hh")
                    seed = random.uniform(0, 1)
                    p_topo = compute_probability_with_distance(allocation_result[node], self.EPR_p, distance)
                    # p_topo = 1
                    if seed >= 1 - p_topo:
                        status_table[node] = current_step
                        node_to_delet.append(node)
                        to_go_table[node] = 'finished'
                        for succ in graph_to_process.succ[node]:
                            if all([to_go_table[pred] == 'finished' for pred in graph_to_process.pred[succ]]):
                                to_go_table[succ] = 'ready'
                                if node not in output_layer:
                                    node_to_add.append(succ)

                elif len(parent_nodes) == 1:
                    pass

                else:
                    pass
            for node in node_to_delet:
                front_layer.remove(node)
                graph_to_process.remove_node(node)
            for node in node_to_add:
                front_layer.append(node)
        print(a)

        curren_time = self.des.current_time + current_step
        # given partition and qpu mapping, compute the qpumapping
        count = Counter(placement.partition)
        qpu_mapping = placement.qpu_mapping[0]
        used_qpu = {value: count[key] for key, value in qpu_mapping.items()}
        tmp = self.des.scheduler.registered_jobs[group[0]][0]
        print(tmp)
        # job scheduler shouldn't put event in the des directly, it should return the event and let des handle it
        # finished_event = FinishedJob(curren_time, self.des.scheduler.registered_jobs[group[0]][0], used_qpu,
        #                              str(self.des.scheduler.registered_jobs[group[0]][0].id) + ' finished+ withallocation' + self.name)
        finished_event = FinishedJob(curren_time, self.des.scheduler.registered_jobs[group[0]][0], used_qpu,
                                     str(self.des.scheduler.registered_jobs[group[0]][0].id) + ' finished+ withallocation' + self.name,  self.des.scheduler.registered_jobs[group[0]][1])
        self.des.schedule_event(finished_event)
        return current_step


    def test_flow_all(self):
        group = [0]
        priority_time = self._simulate_run_single(group)
        random_time = self._simulate_run_single_random(group)
        average_time = self._simulate_run_single_average(group)
        greedy_time = self._simulate_run_single_greedy(group)
        print(0, average_time, priority_time, greedy_time,"finished running")
        
        return random_time, average_time, priority_time, greedy_time
        # Use ThreadPoolExecutor for I/O-bound tasks, or ProcessPoolExecutor for CPU-bound tasks
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     # Submit the functions to be executed in parallel
        #     future_priority = executor.submit(self._simulate_run_single, group)
        #     future_average = executor.submit(self._simulate_run_single_average, group)
        #
        #     # Retrieve the results
        #     priority_time = future_priority.result()
        #     average_time = future_average.result()
        # print(average_time,priority_time, "finished running")
        # print("hh")




    def simulate_run_group(self, group, method = None):
        #  this function simulate the flow scheduling process for a group of jobs that have  conflict with each other
        #         step 1 get necessary information for the group
        #  group is the index of the job in the job queue
        if len(group) == 1:

            self._simulate_run_single(group)
            return
        graph_to_process = {i: copy.deepcopy(self.des.scheduler.registered_jobs[i][1].remote_dag) for i in group}
        priority = {i: self._compute_priority_1(graph_to_process[i]) for i in group}
        input_layer = {}
        status_table = {}
        front_layer = {}
        output_layer = {}
        to_go_table = {}
        remote_combo = {}
        parents = {}
        parents_qpu = {}
        qpu_inolved_list = {}
        for i in group:
            placement = self.des.scheduler.registered_jobs[i][1]
            qpus_involved = set(placement.qpu_mapping[0].values())
            remote_dag = placement.remote_dag
            node_to_int = {str(node): i for i, node in enumerate(placement.wig.nodes())}
            input_layer[i] = [node for node in remote_dag.nodes() if remote_dag.nodes._nodes[node]['desc'] == 'Input']
            status_table[i] = {node: 0 for node in remote_dag.nodes()}
            for node in input_layer[i]:
                graph_to_process[i].remove_node(node)

            output_layer[i] = [node for node in remote_dag.nodes() if
                               remote_dag.nodes._nodes[node]['desc'] == 'Output']
            for node in output_layer[i]:
                graph_to_process[i].remove_node(node)
            to_go_table[i] = {node: 'unready' for node in remote_dag.nodes()}
            qpu_list = sorted(list(qpus_involved))
            qpu_inolved_list[i] = qpu_list
            remote_combo[i] = sorted(list(combinations(qpu_list, 2)))
            for node in input_layer[i]:
                to_go_table[i][node] = 'finished'
            parents[i] = {node: [] for node in remote_dag.nodes()}
            parents_qpu[i] = {node: [] for node in remote_dag.nodes()}
            for node in graph_to_process[i].nodes():
                print(node)
                parents[i][node] = find_two_farthest_parents(remote_dag, node)
            data_type = placement.modified_circuit._dag_data[6]
            for node in parents[i].keys():
                qubit_index = [data_type[node] for node in parents[i][node]]
                if qubit_index:
                    parents[i][node] = [qubit_index]
                    parents_partition = [placement.partition[node_to_int[node]] for node in qubit_index]
                    parents_qpu[i][node] = sorted([placement.qpu_mapping[0][par] for par in parents_partition])

            
            nodes_to_delete= []
            for node in graph_to_process[i].nodes():
                if parents_qpu[i][node][0] == parents_qpu[i][node][1]:
                    nodes_to_delete.append(node)
                    to_go_table[i][node] = 'finished'
            for node in nodes_to_delete:
                graph_to_process[i].remove_node(node)

            front_layer[i] = [node for node in graph_to_process[i].nodes() if
                              graph_to_process[i].in_degree(node) == 0 and graph_to_process[i].out_degree(node) != 0]
        print("finished step 1")
        step = 0
        finished_group= set()
        # get all involved  qpus in current group
        print("finished step 1 start flow scheduling", "job_name", [self.des.scheduler.registered_jobs[i][0].name for i in group])
        while any(len(graph_to_process[i].nodes()) != 0 for i in group):
            step +=1
            node_to_delete = {i: [] for i in group}
            node_to_add = {i: [] for i in group}
            to_go_pool = {i: {combo: [] for combo in remote_combo[i]} for i in group }
            # print("hh")
            for i in group:
                for node in front_layer[i]:
                    # if parents_qpu[i][node][0] == parents_qpu[i][node][1]:
                    #     graph_to_process[i].remove_node(node)
                    for combo in to_go_pool[i].keys():
                        if parents_qpu[i][node] == list(combo) or parents_qpu[i][node] == list(combo)[::-1]:
                            to_go_pool[i][combo].append(node)

    #       filter out the empty value in the to_go_pool
            # print(to_go_pool)
            to_go_pool = {i: {key: value for key, value in to_go_pool[i].items() if value } for i in group}
            # print(to_go_pool)
            allocation_result = {i: {node: 0 for node in front_layer[i]} for i in group}
            competion_set = self.check_competition_between_jobs(to_go_pool)
            resources = self.get_resources(to_go_pool)

            # resources allocation process
            if not competion_set:
                for i in group:
                    if i in finished_group:
                        continue
                    inter_sets_allocations = allocate_resources_between_sets_priority(to_go_pool[i],
                                                                                      resources, priority[i])
                    for comb, nodes in to_go_pool[i].items():
                        if nodes:
                            total_resources = inter_sets_allocations[comb]
                            total_prioriy = sum([priority[i][node] for node in nodes])
                            for node in nodes:
                                allocation_result[i][node] = round(total_resources * priority[i][node] / total_prioriy)
                                total_resources -= allocation_result[i][node]
                            if total_resources > 0:
                                sorted_nodes = sorted(nodes, key=lambda n: priority[i][n], reverse=True)
                                for node in sorted_nodes:
                                    if total_resources > 0:
                                        allocation_result[i][node] += 1
                                        total_resources -= 1
            else:
                while self._is_allocatabale(to_go_pool,resources):
                    print("allocating", allocation_result)
                    for i in group:
                        if i in finished_group:
                            continue
                        candidates = []
                        allocation_level = 0
                        while not candidates and allocation_level <= max(allocation_result[i].values()):
                            candidates = [node for node in front_layer[i]
                                          
                              if allocation_result[i][node] == allocation_level
                              and any(resources[q1] > 0 and resources[q2] > 0 for q1, q2 in to_go_pool[i].keys() if node in to_go_pool[i][q1, q2])]
                            allocation_level += 1
                        allocated = False
                        while candidates and not allocated:
                            chosen_node = max(candidates, key=lambda n: priority[i][n])
                            qpus = [key for key, value in to_go_pool[i].items() if chosen_node in value][0]
                            if resources[qpus[0]] > 0 and resources[qpus[1]] > 0:
                                allocation_result[i][chosen_node] += 1
                                resources[qpus[0]] -= 1
                                resources[qpus[1]] -= 1
                                allocated = True
                            else:
                                candidates.remove(chosen_node)
            # print('finish allcoation', allocation_result, to_go_pool, resources)
            # print(allocation_result,to_go_pool, resources, [len(graph_to_process[i].nodes()) for i in group], [self.des.scheduler.registered_jobs[i][0].name for i in group])

            for i in group:
                if len(graph_to_process[i].nodes()) == 0:
                    continue
                for node in front_layer[i]:
                    if allocation_result[i][node] > 0 :
                        tmp = parents_qpu[i][node]
                        distance = nx.shortest_path_length(self.qcloud.network,
                                                           parents_qpu[i][node][0],parents_qpu[i][node][1])
                        p_topo = compute_probability_with_distance(allocation_result[i][node], self.EPR_p, distance)
                        seed = random.uniform(0, 1)
                        if seed >= 1 - p_topo:
                            status_table[i][node] = step
                            node_to_delete[i].append(node)
                            to_go_table[i][node] = 'finished'
                            for succ in graph_to_process[i].succ[node]:
                                if all([to_go_table[i][pred] == 'finished' for pred in graph_to_process[i].pred[succ]]):
                                    to_go_table[i][succ] = 'ready'
                                    if node not in output_layer[i]:
                                        node_to_add[i].append(succ)
                        else:
                            pass
                for node in node_to_delete[i]:
                    front_layer[i].remove(node)
                    graph_to_process[i].remove_node(node)
                for node in node_to_add[i]:
                    front_layer[i].append(node)

                if len(graph_to_process[i].nodes()) == 0:
                    curren_time = self.des.current_time + step
                    count = Counter(self.des.scheduler.registered_jobs[i][1].partition)
                    qpu_mapping = self.des.scheduler.registered_jobs[i][1].qpu_mapping[0]
                    used_qpu = {value: count[key] for key, value in qpu_mapping.items()}
                    finished_event = FinishedJob(curren_time, self.des.scheduler.registered_jobs[i][0], used_qpu,
                                                 str(self.des.scheduler.registered_jobs[i][0].id) + ' finished+ withallocation' + self.name , self.des.scheduler.registered_jobs[i][1])
                    self.des.schedule_event(finished_event)
                    finished_group.add(i)
                    print('finished event' + str(i))
                    # print("hh")

        print("all finished")
        return 
        























    def check_competition_between_jobs(self,to_go_pool):
        # Create a list of QPU sets and a corresponding list of job keys
        qpu_set_list = [set(qpu for qpu_pair in job.keys() for qpu in qpu_pair) for job in to_go_pool.values()]
        job_keys = list(to_go_pool.keys())

        competition_set = {}
        for (i, set1), (j, set2) in combinations(enumerate(qpu_set_list), 2):
            intersection = set1.intersection(set2)
            if intersection:
                job_key_pair = (job_keys[i], job_keys[j])
                competition_set[job_key_pair] = list(intersection)

        if not competition_set:
            return False
        else:
            return competition_set


    def check_competition_between_jobs(self, to_go_pool):
        # qpu_set_list = [qpu_pair for qpu_pair in to_go_pool.values().keys()]
        qpu_set_list = [list(set.keys()) for set in to_go_pool.values()]
        qpu_set_list = [set(sorted(set().union(*qpu_pair))) for qpu_pair in qpu_set_list]
    #     check if any two qpu set has intersection
        competition_set = {}
        for (i,set1), (j,set2) in combinations(enumerate(qpu_set_list), 2):
            if set1.intersection(set2):
                if (i,j) not in competition_set:
                    competition_set[(i,j)] = set1.intersection(set2)
    #     check whther there is any competition between jobs
        if not competition_set:
            return False
        else:
            return competition_set




    def get_resources(self, togo_pool):
        single_key_list = []
        for key, value in togo_pool.items():
            for key1, value1 in value.items():
                single_key_list.append((key1))
        # print("hh")
        result_set = set().union(*single_key_list)
        resources = {qpu_idx: self.qcloud.network.nodes[qpu_idx]['qpu'].ncm_qubits for qpu_idx in result_set}
        return resources

    def _is_allocatabale(self, to_go_pool, resources):
        # check we can still allocate to any group in to_go_pool
        for i, value in to_go_pool.items():
            for key in value.keys():
                qpu1, qpu2 = key
                if resources[qpu1] > 0 and resources[qpu2] > 0:
                    return True
        return False

    def _compute_priority_1(self, dag):
        leaf_nodes = [node for node in dag.nodes() if dag.out_degree(node) == 0]

        # Use memoization to avoid redundant calculations
        @lru_cache(maxsize=None)
        def find_paths_to_leaves(start_node):
            if start_node in leaf_nodes:
                return 1  # Length of the path from a leaf to itself is 1

            max_length = 0
            for neighbor in dag.successors(start_node):
                max_length = max(max_length, 1 + find_paths_to_leaves(neighbor))
            return max_length

        return {node: find_paths_to_leaves(node) for node in dag.nodes()}

