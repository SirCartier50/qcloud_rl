# cluster defines the cloud components, including how many qpu, what's each qpu like, and the state of each qpu and qubits'
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from collections import Counter, deque
from itertools import combinations
from networkx import isomorphism
import pulp
import copy
from pytket import Circuit, OpType, qasm
from deap import base, creator, tools, algorithms
from pytket.utils import Graph
import multiprocessing
class annealer:
    def __init__(self, circuit, cloud):
        self.circuit = circuit
        self.cloud = cloud

    def annealing(self, initial_placement, iterations, temperature):
        # partition is a dictionary, the key is the qpu id, the value is a list of qubit id
        #
        overhead_histy = []
        partition = initial_placement
        qubit_partition_dict = {qubit: key for key, value in partition.items() for qubit in value}
        cost = self.calculate_cost(qubit_partition_dict, self.circuit)
        for i in range(iterations):
            # print(i, cost)
            #  choose one random  qubit in random qpu to swap out
            qpu_to_swap = random.choice(list(partition.keys()))
            qubit_to_swap = random.choice(partition[qpu_to_swap])

            #         find destination qpu
            destination_qpu = random.choice(list(set(partition.keys()) - {qpu_to_swap}))
            potential_new_partition = copy.deepcopy(partition)

            # case 1 if the destination qpu has available qubits, just remove the qubit from the qpu_to_swap and add it to the destination qpu
            if len(potential_new_partition[destination_qpu]) < self.cloud.network.nodes[destination_qpu][
                'qpu'].available_qubits:

                potential_new_partition[qpu_to_swap].remove(qubit_to_swap)
                potential_new_partition[destination_qpu].append(qubit_to_swap)
            # case 2 if the destination qpu has no available qubits, we need to swap one qubit back
            elif len(potential_new_partition[destination_qpu]) == self.cloud.network.nodes[qpu_to_swap][
                'qpu'].available_qubits:
                potential_new_partition[qpu_to_swap].remove(qubit_to_swap)
                potential_new_partition[destination_qpu].append(qubit_to_swap)
                qubit_to_swap_back = random.choice(partition[destination_qpu])
                potential_new_partition[destination_qpu].remove(qubit_to_swap_back)
                potential_new_partition[qpu_to_swap].append(qubit_to_swap_back)
            else:
                print('error',i , {key: len(value) for key, value in potential_new_partition.items()})

            qubit_partition_dict = {qubit: key for key, value in potential_new_partition.items() for qubit in value}
            new_cost = self.calculate_cost(qubit_partition_dict, self.circuit)
            delta_cost = new_cost - cost

            if delta_cost < 0:
                partition = potential_new_partition
                cost = new_cost
                temperature = temperature * 0.99
            else:
                p = np.exp(-delta_cost / temperature)
                random_number = random.random()
                if random_number < p:
                    alpha = delta_cost / temperature
                    partition = potential_new_partition
                    cost = new_cost
            overhead_histy.append(cost)
        return partition, overhead_histy

    def calculate_cost(self, partition, circuit):
        cost = 0
        for command in circuit:
            type = command.op.type
            qubits = command.qubits
            if len(qubits) == 2 and type != OpType.Measure and type != OpType.Reset:

                if partition[qubits[0]] != partition[qubits[1]]:
                    distance = nx.shortest_path_length(self.cloud.network, partition[qubits[0]], partition[qubits[1]])
                    cost += distance

        return cost


class cp_qubit:
    def __init__(self, id, qpu):
        self.qid = id
        self.qpu = qpu
        self.occupied = False
        # denotes which job thi s qubit is currently occupied by
        self.job_id = None

    def allocate(self, job_id):
        self.occupied = True
        self.job_id = job_id


class cm_qubit:
    def __init__(self, id, qpu):
        self.qid = id
        self.qpu = qpu
        self.occupied = False
        # denotes which job this qubit is currently occupied by
        self.job_id = None

    def allocate(self, job_id):
        self.occupied = True
        self.job_id = job_id


class switch:
    def __init__(self, name, qpus, ncm_qubits):
        self.name = name
        self.qpus = qpus
        self.ncm_qubits = ncm_qubits


class qpu:
    def __init__(self, id, ncm_qubits, ncp_qubits):
        self.qpuid = id
        self.occupied = False
        self.job_id = []
        self.job_status = {}
        self.ncm_qubits = ncm_qubits
        self.ncp_qubits = ncp_qubits
        self.cm_qubits = []
        self.cp_qubits = []
        self.init_qpu()
        self.available_qubits = ncp_qubits
        self.collaboration_data = None

    def allocate_qubits(self, job_id, n):
        self.occupied = True
        self.job_id = job_id
        self.available_qubits -= n

    def init_qpu(self):
        for i in range(self.ncm_qubits):
            self.cm_qubits.append(cm_qubit(i, self))
        for i in range(self.ncp_qubits):
            self.cp_qubits.append(cp_qubit(i, self))

    def allocate_job(self, job, n_qubits):
        self.occupied = True
        self.job_id.append(job.id)
        self.job_status[job.id] = 'running'
        self.available_qubits -= n_qubits

    def free_qubits(self, n_qubits, job):
        self.job_status[job.id] = 'finished'
        self.available_qubits += n_qubits


class qCloud:
    def __init__(self, num_qpus, topology_func, topology_args, ncm_qubits=5, ncp_qubits=30, need_switch=False,
                 swicth_number=0, topology = None):
        # Generate the topology
        if need_switch:
            self.network = topology_func(num_qpus, topology_args)
            self.qpus = []
            for node in range(num_qpus):
                qpu_instance = qpu(node, ncm_qubits, ncp_qubits)
                self.network.nodes[node]['type'] = 'qpu'
                self.network.nodes[node]['qpu'] = qpu_instance
                self.qpus.append(qpu_instance)
                self.network.nodes[node]['available_qubits'] = [qubit for qubit in qpu_instance.cp_qubits if
                                                                not qubit.occupied]
            server_per_switch = num_qpus // swicth_number
            for i in range(swicth_number):
                switch_name = i + num_qpus
                self.network.add_node(switch_name)
                qpu_list = []
                for j in range(server_per_switch * i, server_per_switch * (i + 1)):
                    self.network.add_edge(switch_name, j)
                    qpu_list.append(j)
                switch_instance = switch(switch_name, qpu_list, 20)
                self.network.nodes[switch_name]['type'] = 'qpu'
                self.network.nodes[switch_name]['switch'] = switch_instance
            self.qpu_qubit_num = ncp_qubits
            colors = ['lightblue' if self.network.nodes[node]['type'] == 'qpu' else 'lightgreen' for node in
                      self.network]
            pos = nx.spring_layout(self.network)  # Position nodes using the spring layout for better visualization
            #nx.draw(self.network, pos, with_labels=True, node_color=colors, edge_color='gray', node_size=2000,
                    #font_size=15)
            ##plt.show()

            # print(self.network.nodes)

        else:
            if topology is not None:
                self.network = topology
                print("using the given topology")
            else:

                self.network = topology_func(num_qpus, topology_args)
            self.qpus = []
            self.collboration_data = None
            # Add QPU instances to the topology
            for node in range(num_qpus):
                qpu_instance = qpu(node, ncm_qubits, ncp_qubits)
                self.network.nodes[node]['type'] = 'qpu'
                self.network.nodes[node]['qpu'] = qpu_instance
                self.qpus.append(qpu_instance)
                self.network.nodes[node]['available_qubits'] = [qubit for qubit in qpu_instance.cp_qubits if
                                                                not qubit.occupied]
            self.qpu_qubit_num = ncp_qubits
            self.set_collaboration_data()
            self.ncm_qubits = ncm_qubits
    def test_legal(self, partition):
        #     partition is a dict where the key is qpu and the value is a list of qubits
            for qpu in partition.keys():
                if len(partition[qpu]) > self.network.nodes[qpu]['qpu'].available_qubits:
                    return False
            return True
    def calculate_cost_qpu(self, partition, circuit):
        cost = 0
        for command in circuit:
            type = command.op.type
            qubits = command.qubits
            if len(qubits) == 2 and type != OpType.Measure and type != OpType.Reset:
                part_1 = partition[qubits[0]]
                part_2 = partition[qubits[1]]

                if partition[qubits[0]] != partition[qubits[1]]:
                    distance = nx.shortest_path_length(self.network, partition[qubits[0]], partition[qubits[1]])
                    cost += distance

        return cost      
    # def calculate_cost_qpu(self, partition, circuit):
    #     # input here is a dtctionary: partition, where key is QPU and value is the qubits assigend to the QPU
    #     pass

    def _whether_imrpovable(self, qpu_to_qubit):
        count = 0
        for qpu in qpu_to_qubit.keys():
            qubits = self.network.nodes[qpu]['qpu'].available_qubits
            if qubits > len(qpu_to_qubit[qpu]):
                count += 1
        if count > 1:
            return True
        return False
    def find_best_imrpovable_qubit(self, smallest_qpu, qpu_to_qubit, wig): 
        result = None
        largest_qpu = None
        max_weight = 0
        for qubit in qpu_to_qubit[smallest_qpu]:
            candidates = [qpu for qpu in qpu_to_qubit.keys() if len(qpu_to_qubit[qpu]) < self.network.nodes[qpu]['qpu'].available_qubits and qpu!= smallest_qpu]
            for candidate in candidates:
                sum = 0
                for qubit_x in qpu_to_qubit[candidate]:
                    if wig.has_edge(qubit, qubit_x):
                        sum += wig[qubit][qubit_x]['weight']
                for qubit_y in qpu_to_qubit[smallest_qpu]:
                    if qubit_y != qubit and wig.has_edge(qubit, qubit_y):
                        sum -= wig[qubit][qubit_y]['weight']
                if sum > max_weight:
                    max_weight = sum
                    result = qubit
                    largest_qpu = candidate

        return result,largest_qpu 

        
    def improve_placement(self, placement):
        wig = placement.wig
        node_to_int = {node: i for i, node in enumerate(wig.nodes())}
        int_to_node = {i: node for i, node in enumerate(wig.nodes())}
        qpu_mapping = placement.qpu_mapping[0]
        try:
            reverse_mapping = {value: key for key, value in qpu_mapping.items()}
        except:
            print("error")
        
        partition = {node:  qpu_mapping[placement.partition[node_to_int[node]]] for node in wig.nodes()}
        old_cost = self.calculate_cost_qpu(partition, placement.job.circuit)
        # reverse mapping from qpu to partition
        qpu_to_qubit = {qpu: [] for qpu in qpu_mapping.values()}
        for key, value in partition.items():
            qpu_to_qubit[value].append(key)
        adjusted_edge = set()
        tested_qpu = set()
        while self._whether_imrpovable(qpu_to_qubit):
            # candiates = [qpu for qpu in qpu_to_qubit.keys() if len(qpu_to_qubit[qpu]) < self.network.nodes[qpu]['qpu'].available_qubits]
            smallest_qpu = min(qpu_to_qubit.keys(), key=lambda x: len(qpu_to_qubit[x]))
            # find the qubit ine the samlles_qpu that has the largest interaction with other qubits:
            qubit, largest_qpu = self.find_best_imrpovable_qubit(smallest_qpu, qpu_to_qubit, wig)
            if qubit is None:
                break
            else:
                qpu_to_qubit[smallest_qpu].remove(qubit)
                qpu_to_qubit[largest_qpu].append(qubit)
                partition[qubit] = largest_qpu
        old_partition = placement.partition
        # print(old_partition)
        # need to adjust in original placement: communication cost, modified_circuit, partition, remote_dag, remote_wig
        # recompute the communication cost
        new_cost = self.calculate_cost_qpu(partition, placement.job.circuit)
        # print("old_cost:   ", old_cost)
        # print("new_cost:   ", new_cost)
        if old_cost < new_cost:
            return None
        # recompute modified_circuit, remote_dag
        # test
        placement.old_partition = old_partition
        placement.old_rmote_dag = placement.remote_dag
        placement.old_modified_circuit = placement.modified_circuit

        new_modified_circuit, new_remote_dag = self.reconstruct_circuit(partition, placement)
        placement.modified_circuit = new_modified_circuit
        placement.remote_dag = new_remote_dag  
        placement.qpu_to_dict = qpu_to_qubit     
        for i in range(len(placement.partition)):
            item = placement.partition[i]
            node = int_to_node[i]
            qpu = partition[node]
            part = reverse_mapping[qpu]
            if part != placement.partition[i]:
                placement.partition[i] = part
        new_partition = placement.partition
        # print(new_partition)
        print("finsih_improvement")
        placement.cost = new_cost
        # print(old_cost == placement.cost)
        return placement
        
   
    def reconstruct_circuit(self, partition, placement):
        circuit = placement.job.circuit
        modified_circuit = Circuit()
        for qubit in circuit.qubits:
            modified_circuit.add_qubit(qubit)
        for command in circuit:
            op_type = command.op.type
            qubits = list(command.args)
            if len(qubits) == 2 and op_type != OpType.Measure:
                node1 = qubits[0]
                node2 = qubits[1]
                part_1 = partition[node1]
                part_2 = partition[node2]
                if part_1 != part_2:
                    # add a swap gate
                    modified_circuit.add_gate(op_type, qubits)
        g = Graph(modified_circuit)
        dag = g.as_nx()
        # placement.modified_circuit = modified_circuit
        # placement.remote_dag = dag
        return modified_circuit, dag




    def ga_find_placement_multi(self, circuit):
        n_qubits = circuit.n_qubits
        qpu_list = list(self.find_connected_qpus(n_qubits))

        if not qpu_list:
            raise ValueError("No connected QPUs found that can accommodate the circuit.")

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", dict, fitness=creator.FitnessMin)

        def init_individual():
            partition = self.random_find_placement(circuit)
            individual = creator.Individual(partition)
            return individual

        def mutate(individual):
            partition = individual
            qpu_list = list(partition.keys())

            qpu_to_swap = random.choice(qpu_list)
            destination_qpu = random.choice(qpu_list)
            while qpu_to_swap == destination_qpu:
                destination_qpu = random.choice(qpu_list)

            qubit_to_swap = random.choice(partition[qpu_to_swap])
            des_qubit_to_swap = random.choice(partition[destination_qpu])

            partition[qpu_to_swap].remove(qubit_to_swap)
            partition[destination_qpu].remove(des_qubit_to_swap)
            partition[qpu_to_swap].append(des_qubit_to_swap)
            partition[destination_qpu].append(qubit_to_swap)

            return individual,

        def crossover(ind1, ind2):
            partition1 = ind1
            partition2 = ind2

            for command in circuit:
                type = command.op.type
                qubits = command.qubits
                swap_prb = random.random()

                if len(qubits) == 2 and type != OpType.Measure and type != OpType.Reset and swap_prb < 0.1:
                    p1_qpu_i = None
                    p1_qpu_j = None
                    for pi in partition1:
                        if qubits[0] in partition1[pi]:
                            p1_qpu_i = pi
                        if qubits[1] in partition1[pi]:
                            p1_qpu_j = pi

                    p2_qpu_i = None
                    p2_qpu_j = None
                    for pi in partition2:
                        if qubits[0] in partition2[pi]:
                            p2_qpu_i = pi
                        if qubits[1] in partition2[pi]:
                            p2_qpu_j = pi

                    if p1_qpu_i != p2_qpu_i and p1_qpu_j != p2_qpu_j:
                        if (len(partition1[p2_qpu_i]) < self.qpu_qubit_num and len(partition1[p2_qpu_j]) < self.qpu_qubit_num and len(partition2[p1_qpu_i]) < self.qpu_qubit_num and len(partition2[p1_qpu_j]) < self.qpu_qubit_num):
                            partition1[p2_qpu_i].append(qubits[0])
                            partition1[p2_qpu_j].append(qubits[1])
                            partition1[p1_qpu_i].remove(qubits[0])
                            partition1[p1_qpu_j].remove(qubits[1])
                            partition2[p1_qpu_i].append(qubits[0])
                            partition2[p1_qpu_j].append(qubits[1])
                            partition2[p2_qpu_i].remove(qubits[0])
                            partition2[p2_qpu_j].remove(qubits[1])
            assert self.test_legal(partition1) and self.test_legal(partition2)

            return ind1, ind2

        toolbox = base.Toolbox()
        toolbox.register("individual", init_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", crossover)
        toolbox.register("mutate", mutate)
        toolbox.register("select", tools.selTournament, tournsize=3)

        def evaluate(individual):
            cost = self.calculate_cost(individual, circuit)
            if not self.test_legal(individual):
                return 100000 + cost,
            return cost,

        toolbox.register("evaluate", evaluate)

        # Use multiprocessing for parallel evaluation
        pool = multiprocessing.Pool()
        toolbox.register("map", pool.map)

        population = toolbox.population(n=50)
        ngen = 10
        cxpb = 0.7
        mutpb = 0.2

        result_population, logbook = algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen, verbose=False)

        pool.close()
        pool.join()

        best_individual = tools.selBest(result_population, 1)[0]
        partition = best_individual
        final_cost = self.calculate_cost(partition, circuit)
        return partition, final_cost


    def ga_find_placement(self, circuit):

        # def calculate_cost(partition, circuit):
        #     cost = 0
        #     for command in circuit:
        #         type = command.op.type
        #         qubits = command.qubits
        #         if len(qubits) == 2 and type != OpType.Measure and type != OpType.Reset:
        #             #if qubits[0] in partition and qubits[1] in partition:
        #                 qpu_i = None
        #                 qpu_j = None
        #                 for pi in partition:
        #                     if qubits[0] in partition[pi]:
        #                         qpu_i  = pi
        #                     if qubits[1] in partition[pi]:
        #                         qpu_j = pi
        #
        #                 # if partition[qubits[0]] != partition[qubits[1]]:
        #                 if qpu_i != qpu_j:
        #
        #                     cost += nx.shortest_path_length(self.network, qpu_i, qpu_j)
        #     return cost

        n_qubits = circuit.n_qubits
        qpu_list = list(self.find_connected_qpus(n_qubits))


        if not qpu_list:
            raise ValueError("No connected QPUs found that can accommodate the circuit.")

        # qpu_qubit_usage = {qpu: 0 for qpu in qpu_list}

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", dict, fitness=creator.FitnessMin)

        def init_individual():
            partition = self.ramdom_find_placement(circuit)
            # Convert the partition to a list of qpu indices, where the index of qubits is not starting from 0

            # qubit_to_qpu = {qubit: qpu for qpu, qubits in partition.items() for qubit in qubits}
            # mapping = {qubit: random.choice(qpu_list) for qubit in range(n_qubits)}
            # individual = creator.Individual([mapping[qubit] for qubit in range(n_qubits)]) 
            individual = creator.Individual(partition)
            return individual

        def mutate(individual):
            #     randomly select two qubits and swap their qpu
            partition = individual
            qpu_list = list(partition.keys())

            qpu_to_swap = random.choice(qpu_list)
            destination_qpu = random.choice(qpu_list)
            while qpu_to_swap == destination_qpu:
                destination_qpu = random.choice(qpu_list)

            qubit_to_swap = random.choice(partition[qpu_to_swap])
            des_qubit_to_swap = random.choice(partition[destination_qpu])

            partition[qpu_to_swap].remove(qubit_to_swap)
            partition[destination_qpu].remove(des_qubit_to_swap)
            partition[qpu_to_swap].append(des_qubit_to_swap)
            partition[destination_qpu].append(qubit_to_swap)

            # if len(partition[destination_qpu]) < self.qpu_qubit_num:
            #     partition[qpu_to_swap].remove(qubit_to_swap)
            #     partition[destination_qpu].append(qubit_to_swap)
            return individual,
    
        def crossover(ind1, ind2):
            #     randomly select a qubit and swap the qpu
            partition1 = ind1
            partition2 = ind2
            count = 0
            for command in circuit:
                type = command.op.type
                qubits = command.qubits
                swap_prb = random.random()

                if len(qubits) == 2 and type != OpType.Measure and type != OpType.Reset and swap_prb < 0.1:
                    #if qubits[0] in partition and qubits[1] in partition:
                        p1_qpu_i = None
                        p1_qpu_j = None
                        for pi in partition1:
                            if qubits[0] in partition1[pi]:
                                p1_qpu_i = pi
                            if qubits[1] in partition1[pi]:
                                p1_qpu_j = pi

                        
                        p2_qpu_i = None
                        p2_qpu_j = None
                        for pi in partition2:
                            if qubits[0] in partition2[pi]:
                                p2_qpu_i = pi
                            if qubits[1] in partition2[pi]:
                                p2_qpu_j = pi

                        # swap the two qubits assignment of two individuals
                        if p1_qpu_i != p2_qpu_i and p1_qpu_j != p2_qpu_j:
                            # swap the qubits and make sure the final result is legal
                            if(len(partition1[p2_qpu_i]) < self.qpu_qubit_num and len(partition1[p2_qpu_j]) < self.qpu_qubit_num and len(partition2[p1_qpu_i]) < self.qpu_qubit_num and len(partition2[p1_qpu_j]) < self.qpu_qubit_num):
                                partition1[p2_qpu_i].append(qubits[0])
                                partition1[p2_qpu_j].append(qubits[1])
                                partition1[p1_qpu_i].remove(qubits[0])
                                partition1[p1_qpu_j].remove(qubits[1])
                                partition2[p1_qpu_i].append(qubits[0])
                                partition2[p1_qpu_j].append(qubits[1])
                                partition2[p2_qpu_i].remove(qubits[0])
                                partition2[p2_qpu_j].remove(qubits[1])
            assert self.test_legal(partition1) and self.test_legal(partition2)





            # qpu_to_swap = random.choice(qpu_list)
            # qubit_to_swap = random.choice(partition1[qpu_to_swap])
            # if len(partition2[qpu_to_swap]) < self.qpu_qubit_num:
            #     partition1[qpu_to_swap].remove(qubit_to_swap)
            #     partition2[qpu_to_swap].append(qubit_to_swap)
            return ind1, ind2

        toolbox = base.Toolbox()
        toolbox.register("individual", init_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", crossover)
        toolbox.register("mutate", mutate)
        toolbox.register("select", tools.selTournament, tournsize=3)

        def evaluate(individual):
            # mapping = {qubit: individual[qubit] for qubit in range(n_qubits)}
            # partition = {qpu: [] for qpu in qpu_list}
            # for qubit, qpu in mapping.items():
            #     if qpu in partition:
            #         partition[qpu].append(qubit)

            # qubit_partition_dict = {qubit: key for key, value in partition.items() for qubit in value}
            # cost = calculate_cost(qubit_partition_dict, circuit)
            cost = self.calculate_cost(individual, circuit)
            if not self.test_legal(individual):
                return 100000 + cost,
            
            return cost,


        toolbox.register("evaluate", evaluate)

        population = toolbox.population(n=50)
        ngen = 100
        cxpb = 0.7
        mutpb = 0.2

        result_population, logbook = algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen, verbose=False)

        best_individual = tools.selBest(result_population, 1)[0]
        # best_mapping = {qubit: best_individual[qubit] for qubit in range(n_qubits)}

        # partition = {qpu: [] for qpu in qpu_list}
        # for qubit, qpu in best_mapping.items():
        #     partition[qpu].append(qubit)
        partition = best_individual
        final_cost =self.calculate_cost(partition, circuit)
        return partition, final_cost#[ind.fitness.values[0] for ind in result_population]
    def calculate_cost(self, partition, circuit):
        partition = {value:key for key, values in partition.items() for value in values}
        cost = 0
        for command in circuit:
            type = command.op.type
            qubits = command.qubits
            if len(qubits) == 2 and type != OpType.Measure and type != OpType.Reset:

                if partition[qubits[0]] != partition[qubits[1]]:
                    cost += nx.shortest_path_length(self.network, partition[qubits[0]], partition[qubits[1]])

        return cost

    def connect_qpus(self, qpu1_id, qpu2_id):
        self.network.add_edge(qpu1_id, qpu2_id)

    def get_communities(self):
        server_nodes = [node for node in self.network.nodes() if
                        self.network.nodes[node]['type'] == 'qpu' and self.network.nodes[node][
                            'qpu'].available_qubits > 0]
        server_graph = self.network.subgraph(server_nodes)
        c = nx.community.greedy_modularity_communities(server_graph, weight='weight')
        return c

    def get_available_qubits(self):
        sum = 0
        for qpu in self.qpus:
            sum += qpu.available_qubits
        return sum

    def find_connected_qpus(self, required_qubits):
        """
        Find a connected set of QPUs with a total number of available qubits >= required_qubits.

        :param required_qubits: Number of qubits required for the circuit.
        :return: A set of QPU node IDs that can run the circuit or an empty set if no such set exists.
        """
        for start_node in self.network.nodes:
            if self.network.nodes[start_node]['type'] == 'qpu':
                visited, qpu_set, total_qubits = self._bfs_qpu_search(start_node, required_qubits)
                if total_qubits >= required_qubits:
                    return qpu_set
        return set()

    # test whether one partition is illegal or not
    def test_legal(self, partition):
        #     partition is a dict where the key is qpu and the value is a list of qubits
        for qpu in partition.keys():
            if len(partition[qpu]) > self.network.nodes[qpu]['qpu'].available_qubits:
                return False
        return True

    def _bfs_qpu_search(self, start_node, required_qubits):
        """
        Perform BFS to find a connected subgraph of QPUs starting from start_node.

        :param start_node: The starting QPU node ID for BFS.
        :param required_qubits: Number of qubits required for the circuit.
        :return: A tuple of (visited set of nodes, set of QPU nodes, total available qubits).
        """
        visited = set()
        queue = deque([start_node])
        qpu_set = set()
        total_qubits = 0

        while queue:
            node = queue.popleft()
            if node not in visited and self.network.nodes[node]['type'] == 'qpu':
                visited.add(node)
                qpu_set.add(node)
                total_qubits += len(self.network.nodes[node]['available_qubits'])

                if total_qubits >= required_qubits:
                    return visited, qpu_set, total_qubits

                for neighbor in self.network.neighbors(node):
                    if neighbor not in visited:
                        queue.append(neighbor)

        return visited, qpu_set, total_qubits

    def get_server_graphs(self):
        server_nodes = [node for node in self.network.nodes() if
                        self.network.nodes[node]['type'] == 'qpu' and self.network.nodes[node][
                            'qpu'].available_qubits > 0]
        self.server_subgraph = self.network.subgraph(server_nodes)

    # this method needs to be called each time the network status be updated
    def create_new_weight_graph(self):
        self.get_server_graphs()
        for u, v, data in self.server_subgraph.edges(data=True):
            qubits_u, qubits_v = self.server_subgraph.nodes[u]['qpu'].available_qubits, self.server_subgraph.nodes[v][
                'qpu'].available_qubits
            self.server_subgraph[u][v]['weight'] = (qubits_u + qubits_v) / 2

    def bfs_queue_gen(self, G, start_node):
        visited = set()
        queue = deque([start_node])
        bfs_order = []

        while queue:
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                bfs_order.append(node)
                neighbors = sorted(G.neighbors(node), key=lambda x: G.degree(x, weight='weight'), reverse=True)
                for neighbor in neighbors:
                    if neighbor not in visited:
                        queue.append(neighbor)

        return deque(bfs_order)

    def plot_graph(self, G):
        nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=15)
        #plt.show()

    def map_remote_wig_to_community_new_2(self, remote_wig, community, partition):
        count = Counter(partition)
        sub_graph = self.network.subgraph(community)
        center_p = nx.center(sub_graph)
        try:
            center_l = nx.center(remote_wig)
        except:
            print('no logical center')
        result = {}
        # plot the subgraph
        nx.draw(sub_graph, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=15)
        #plt.show()
        if center_l:
            center_l = max(center_l, key=lambda x: remote_wig.degree(x, weight='weight'))

        number_qubits = count[center_l]
        candidate_nodes = [node for node in center_p if sub_graph.nodes[node]['qpu'].available_qubits > number_qubits]
        if not candidate_nodes:
            return None
        if center_p:
            center_p = max((node for node in center_p if sub_graph.nodes[node]['qpu'].available_qubits > number_qubits),
                           key=lambda x: sub_graph.degree(x))

        result[center_l] = center_p
        queue = [center_l]
        visited = set()
        count = Counter(partition)

    def map_remote_wig_to_community_new(self, remote_wig, community, partition):

        count = Counter(partition)
        sub_graph = self.network.subgraph(community)
        center_p = nx.center(sub_graph)
        try:
            center_l = nx.center(remote_wig)
        except:
            print('no logical center')
            center_l = None
        result = {}
        # plot the subgraph
        nx.draw(sub_graph, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=15)
        #plt.show()
        
        if center_l:
            center_l = max(center_l, key=lambda x: remote_wig.degree(x, weight='weight'))

        number_qubits = count[center_l]
        candidate_nodes = [node for node in center_p if sub_graph.nodes[node]['qpu'].available_qubits >= number_qubits]
        if not candidate_nodes:
            return None
        if center_p:
            center_p = max((node for node in candidate_nodes if sub_graph.nodes[node]['qpu'].available_qubits >= number_qubits),
                           key=lambda x: sub_graph.degree(x))

        result[center_l] = center_p
        if count[center_l] > sub_graph.nodes[center_p]['qpu'].available_qubits:
            print("hhhh")
        queue = [center_l]
        visited = set()

        while queue:
            node = queue.pop(0)
            if node not in result and center_p not in result.values():
                if count[node] <= sub_graph.nodes[center_p]['qpu'].available_qubits:
                    result[node] = center_p
                    if count[node] > sub_graph.nodes[center_p]['qpu'].available_qubits:
                        print("hhhh")

            # check if the current node has been assigned
            if node not in result and center_p in result.values():
                #     find one location
                candidate = [node for node in sub_graph.nodes() if node not in result.values()]
                #     find closest candidate
                self.plot_graph(sub_graph)
                closest_candidate = min(candidate, key=lambda x: nx.shortest_path_length(sub_graph, center_p, x))
                if count[node] <= sub_graph.nodes[closest_candidate]['qpu'].available_qubits:
                    result[node] = closest_candidate
                    if count[node] > sub_graph.nodes[closest_candidate]['qpu'].available_qubits:
                        print("hhhh")
            neighbors = remote_wig.neighbors(node)
            neighbors = sorted(neighbors, key=lambda x: remote_wig.degree(x, weight='weight'), reverse=True)

            for neighbor in neighbors:
                if neighbor not in result and neighbor not in visited:
                    queue.append(neighbor)
                visited.add(neighbor)

            try:
                new_p_centers = list(sub_graph.neighbors(center_p))
            except KeyError:
                # if the center_p cannot be found in the sub_graph, return None for now
                return None, None
            # check how many number of nodes in new_p_centers are not in the result
            new_p_centers = [node for node in new_p_centers if node not in result.values()]

            for neighbor in neighbors:
                if neighbor in result:
                    continue
                for p_center in new_p_centers:
                    if p_center in result.values():
                        continue
                    if count[neighbor] <= sub_graph.nodes[p_center]['qpu'].available_qubits:
                        result[neighbor] = p_center
                        if count[neighbor] > sub_graph.nodes[p_center]['qpu'].available_qubits:
                            print("hhhh")
                        center_p = p_center
                        break

        #     logic to recheck if all nodes are assigned to a QPU
        unassigned_nodes = [node for node in remote_wig.nodes() if node not in result]
        if unassigned_nodes:
            #     plot the subgraph
            nx.draw(remote_wig, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000,
                    font_size=15)
            #plt.show()
            # print(unassigned_nodes)
            for node_x in unassigned_nodes:
                candidate_unassigned = [node for node in self.network.nodes() if
                             (node not in result.values() and self.network.nodes[node]['qpu'].available_qubits >= count[
                                 node_x])]
                if candidate_unassigned:
                    try:
                        closest_candidate = min(candidate_unassigned, key=lambda x: nx.shortest_path_length(self.network, center_p, x))
                    except:
                        nx.draw(self.network, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000,font_size=15)
                        #plt.show()

                    result[node_x] = closest_candidate
                    if count[node_x] > self.network.nodes[closest_candidate]['qpu'].available_qubits:
                        print("hhhh")
                else:
                    return None, None

        original_sum, weighted_length = self.compute_weighted_length(partition, remote_wig, result)
        return result, weighted_length

    def map_remote_wig_to_community_new_1(self, remote_wig, community, partition):
        from collections import Counter
        import networkx as nx

        count = Counter(partition)
        sub_graph = self.network.subgraph(community)
        center_p = nx.center(sub_graph)
        center_l = nx.center(remote_wig)
        result = {}

        # Ensure center nodes are identified
        if not center_l:
            return None
        center_l = max(center_l, key=lambda x: remote_wig.degree(x, weight='weight'))

        number_qubits = count[center_l]
        if len(center_p) > 1:
            center_p = max((node for node in center_p if sub_graph.nodes[node]['qpu'].available_qubits > number_qubits),
                           key=lambda x: sub_graph.degree(x))

        # Ensure at least one node is mapped
        if not center_p:
            return None

        result[center_l] = center_p
        queue = [center_l]
        visited = set()

        while queue:
            node = queue.pop(0)
            if node not in result:
                result[node] = center_p

            neighbors = list(remote_wig.neighbors(node))
            neighbors = sorted(neighbors, key=lambda x: remote_wig.degree(x, weight='weight'), reverse=True)

            for neighbor in neighbors:
                if neighbor not in result and neighbor not in visited:
                    queue.append(neighbor)
                visited.add(neighbor)

            try:
                new_p_centers = list(sub_graph.neighbors(center_p))
            except KeyError:
                # if the center_p cannot be found in the sub_graph, return the current result and weighted length
                original_sum, weighted_length = self.compute_weighted_length(partition, remote_wig, result)
                return result, weighted_length

            # Find the best mapping for each neighbor
            for neighbor in neighbors:
                if neighbor in result:
                    continue
                best_p_center = min(new_p_centers,
                                    key=lambda p_center: nx.shortest_path_length(sub_graph, center_p, p_center))
                if best_p_center in result.values():
                    continue
                result[neighbor] = best_p_center
                center_p = best_p_center

        original_sum, weighted_length = self.compute_weighted_length(partition, remote_wig, result)
        return result, weighted_length

    def map_remote_wig_to_community(self, remote_wig, community, partition):
        # input is a community, a partition, and a remote_wig
        # output is a dictionary, the key is the node in remote_wig, the value is the node in the community
        # community is a frozenset of nodes
        count = Counter(partition)
        sub_graph = self.network.subgraph(community)
        center_p = nx.center(sub_graph)
        center_l = nx.center(remote_wig)
        result = {}
        if (center_l):
            center_l = max(center_l, key=lambda x: remote_wig.degree(x, weight='weight'))
        number_qubits = count[center_l]
        node_list = [node for node in center_p if sub_graph.nodes[node]['qpu'].available_qubits >= number_qubits]
        if len(center_p) > 1:
            center_p = max(
                (node for node in center_p if sub_graph.nodes[node]['qpu'].available_qubits >= number_qubits),
                key=lambda x: sub_graph.degree(x))
        result[center_l] = center_p
        queue = [center_l]
        visited = set()
        while queue:
            node = queue.pop(0)
            if node not in result and center_p not in result.values():
                result[node] = center_p
            neighbors = remote_wig.neighbors(node)
            neighbors = sorted(neighbors, key=lambda x: remote_wig.degree(x, weight='weight'), reverse=True)
            sorted_neighbors_with_weights = [(x, remote_wig.degree(x, weight='weight')) for x in neighbors]
            sorted_neighbors_with_weights.sort(key=lambda item: item[1], reverse=True)
            for neighbor in neighbors:
                if neighbor not in result and neighbor not in visited:
                    queue.append(neighbor)
                visited.add(neighbor)
            try:
                new_p_centers = list(sub_graph.neighbors(center_p))
            #     if can not find the center_p in the sub_graph, just return non for now
            except:
                return None, None
            for neighbor in neighbors:
                if neighbor in result:
                    continue
                for p_center in new_p_centers:
                    if p_center in result.values():
                        continue
                    result[neighbor] = p_center
                    center_p = p_center
                    break
        # if len(result) != len(remote_wig.nodes):
        #     for node in remote_wig.nodes:
        #         if node not in result:
        #        find the node with the the highest degree with node in the community

        original_sum, weighted_lenghh = self.compute_weighted_length(partition, remote_wig, result)
        return result, weighted_lenghh

    def community_find_qpus_weight(self, partition, remote_wig):
        #nx.draw(remote_wig, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=15)
        ##plt.show()
        self.create_new_weight_graph()
        c = self.get_communities()
        total_qubits_list = [sum([self.network.nodes[node]['qpu'].available_qubits for node in community])
                             for community
                             in c]
        qubit_per_qpu = {qpu:self.network.nodes[qpu]['qpu'].available_qubits    for community in c for qpu in community}
        count = Counter(partition)
        result = {}
        # first determine whether single community can run the job, if yes: evalueate multiple possible qpu combinations
        # if not, merge two communities
        # determine which community can run the job
        single_community = list(filter(
            lambda x: sum([self.network.nodes[node]['qpu'].available_qubits for node in x]) >= sum(
                count.values()) and len(x) >= len(count), c))
        result = list(map(lambda x: self.map_remote_wig_to_community_new(remote_wig, x, partition), single_community))
        # filter none reult in the list
        result = list(filter(lambda x: x != (None, None) and x is not None, result))
        if result:
            #     return the best result
            try:
                return min(result, key=lambda x: x[1])
            except:
                print('not possible')

        # print("ss")
        result = {}
        # find all possible combinations of communities that may run the job
        possible_combinations = []
        for r in range(2, len(c) + 1):
            for combo in combinations(c, r):
                # test whether two combination of community is adjacent
                total_qubits_combo = sum(
                    sum(self.network.nodes[node]['qpu'].available_qubits for node in community) for community in combo)
                tmp1 = count.values()
                if sum(count.values()) > total_qubits_combo:
                    continue
                total_qpu_numbers = sum(len(community) for community in combo)
                if len(count) < total_qpu_numbers:
                    possible_combinations.append(combo)
        # print(possible_combinations)
        merged_sets = []
        for frozensets in possible_combinations:
            merged_frozenset = frozenset().union(*frozensets)
            merged_sets.append(merged_frozenset)
        # Step 2: Filter out frozensets that are supersets of any other frozenset
        minimal_sets = []
        for current_set in merged_sets:
            if all(not current_set.issuperset(other_set) for other_set in merged_sets if current_set != other_set):
                minimal_sets.append(current_set)
        # filter out the combination that is not connected
        minimal_sets = list(filter(lambda x: nx.is_connected(self.network.subgraph(x)), minimal_sets))
        #   for each combination of communities, find the best mapping
        result = list(map(lambda x: self.map_remote_wig_to_community_new(remote_wig, x, partition), minimal_sets))
        result = list(filter(lambda x: x != (None, None) and x is not None, result))
        if result:
            return min(result, key=lambda x: x[1])

    def check_availability(self, comb, wig):
        # comb is a dictionary, the key is number of qpus, the value is a list of list, each list elemement is one
        # possibility of placement
        all_available_qubits = self.get_available_qubits()
        for key, value in comb.items():
            for partition in value:
                # check if one placement is possible
                qubits_needed = len(partition)
                if qubits_needed > all_available_qubits:
                    return []
        # in result the first element is qpu_index, the second is the job partition
        res = {}
        res_community = {}
        length_community = {}
        res_enumerate = {}
        lenghth = {}
        for key, value in comb.items():
            res[key] = []
            res_community[key] = []
            res_enumerate[key] = []
            for partition in value:
                # check if one placement is possible
                # try:
                remote_wig = self.contract_wig(wig, partition)
                # except:
                #     print('not possible')
                count = Counter(partition)
                # print(partition)
                if any(element > self.qpu_qubit_num for element in count.values()):
                    continue
                else:
                    start_node = random.randint(0, len(self.qpus) - 1)
                    community_with_weight, lenghth_withweight = self.community_find_qpus_weight(partition, remote_wig)
                    available_qpus, weighted_length_bfs = self.bfs_find_qpus(start_node, partition, remote_wig)
                    aas = self.enumerate_find_qpus(partition)
                    bbb, weighted_length_community = self.find_qpu_community(partition, remote_wig)
                    res[key].append(available_qpus)
                    res_community[key].append(bbb)
                    lenghth[key] = weighted_length_bfs
                    length_community[key] = weighted_length_community
                    res_enumerate[key].append(aas)
        return res, res_community, res_enumerate

    def ramdom_find_placement(self, circuit):
        # find initial random placemnt for a circuit
        #   select a series of qps
        qubits_list = copy.deepcopy(circuit.qubits)
        nqubits = len(qubits_list)
        candidates_qpus = self.find_connected_qpus(nqubits)
        qpu_qubit_usage = {qpu: 0 for qpu in candidates_qpus}
        qubit_mapping = {}
        random.shuffle(qubits_list)
        for qubit in qubits_list:
            for qpu in candidates_qpus:
                if qpu_qubit_usage[qpu] + 1 <= self.qpu_qubit_num:
                    qubit_mapping[qubit] = qpu
                    qpu_qubit_usage[qpu] += 1
                    break
        # reverse mapping of qpu to qubits
        partition = {qpu: [] for qpu in candidates_qpus}
        for key, value in qubit_mapping.items():
            partition[value].append(key)
        return partition

    def sa_find_placement(self, circuit):
        #     find initial random placemnt for a circuit
        #     step1: find a set of connected qpus that can run the job
        #     step2: find the best placement for the job
        n_qubits = circuit.n_qubits
        qpu_list = self.find_connected_qpus(n_qubits)
        initial_placement = self.ramdom_find_placement(circuit)
        # partition = {}
        # for key, value in initial_placement.items():
        #     if value not in partition:
        #         partition[value] = [key]
        #     else:
        #         partition[value].append(key)
        annealing_solver = annealer(circuit, self)
        partition, history = annealing_solver.annealing(initial_placement, 1000, 100)
        return partition, history[-1]

    def find_placement_bfs(self, comb, wig):
        all_available_qubits = self.get_available_qubits()
        for key, value in comb.items():
            for partition in value:
                # check if one placement is possible
                qubits_needed = len(partition)
                if qubits_needed > all_available_qubits:
                    return []
        res = {}
        for key, value in comb.items():
            res[key] = []
            for partition in value:
                remote_wig = self.contract_wig(wig, partition)
                count = Counter(partition)
                # print(partition)
                if any(element > self.qpu_qubit_num for element in count.values()):
                    continue
                else:
                    start_node = random.randint(0, len(self.qpus) - 1)
                    result = self.bfs_find_qpus(start_node ,partition, remote_wig)
                    if result:
                        res[key].append(result[0])
                    else:
                        continue
        return res



    def find_placement(self, comb, wig):
        # comb is a dictionary, the key is number of qpus, the value is a list of list, each list elemement is one
        # possibility of placement
        all_available_qubits = self.get_available_qubits()
        for key, value in comb.items():
            for partition in value:
                # check if one placement is possible
                qubits_needed = len(partition)
                if qubits_needed > all_available_qubits:
                    return []
        res = {}
        for key, value in comb.items():
            res[key] = []
            for partition in value:
                remote_wig = self.contract_wig(wig, partition)
                count = Counter(partition)
                # print(partition)
                if any(element > self.qpu_qubit_num for element in count.values()):
                    continue
                else:
                    result = self.community_find_qpus_weight(partition, remote_wig)
                    if result:
                        res[key].append(result[0])
                    else:
                        continue
        return res

    def contract_wig(self, wig, partition):
        node_to_int = {node: i for i, node in enumerate(wig.nodes())}
        int_to_node = {i: node for node, i in node_to_int.items()}  # Reverse mapping
        count = Counter(partition)
        qubits_in_partitions = {i: [] for i in count.keys()}
        for node_index, partition_1 in enumerate(partition):

            original_node = int_to_node[node_index]
            try:
                qubits_in_partitions[partition_1].append(original_node)
            except:

                print('no partition')
        qubit_partition_dict = {}
        for key, value in qubits_in_partitions.items():
            for qubit in value:
                qubit_partition_dict[qubit] = key
        graph = nx.Graph()
        for edge in wig.edges():
            if qubit_partition_dict[edge[0]] != qubit_partition_dict[edge[1]]:
                if not graph.has_edge(qubit_partition_dict[edge[0]], qubit_partition_dict[edge[1]]):
                    graph.add_edge(qubit_partition_dict[edge[0]], qubit_partition_dict[edge[1]],
                                   weight=wig[edge[0]][edge[1]]['weight'])
                else:
                    graph[qubit_partition_dict[edge[0]]][qubit_partition_dict[edge[1]]]['weight'] += \
                        wig[edge[0]][edge[1]]['weight']

        # calculate the sum of the weight of the edges
        sum = 0
        for edge in graph.edges():
            sum += graph[edge[0]][edge[1]]['weight']
        # print(sum)

        return graph

    def test_sub_isomorphic(self, communities, remote_wig):
        for community in communities:
            # get maximum subgraph of remote_wig that is isomorphic to the subgraph of the community

            sub_graph = self.network.subgraph(community)
            matcher = isomorphism.GraphMatcher(sub_graph, remote_wig)
            if matcher.subgraph_is_isomorphic():
                print(matcher.mapping)
                print('isomorphic')
                return matcher.mapping

            else:
                print('not isomorphic')
        return

    def map_qpu_to_community(self, community, remote_wig):
        subgraph = self.network.subgraph(community)
        sorted_nodes = sorted(remote_wig.nodes(), key=lambda x: remote_wig.degree(x, weight='weight'), reverse=True)
        res = {}
        for node_1 in sorted_nodes:
            #     find most suitable node in the community to mao the node in remote_wig
            try:
                center_node = nx.center(subgraph)
                if len(center_node) > 1:
                    #         find the node with the highest degree in the center node
                    max_degree = 0
                    for node in center_node:
                        if subgraph.degree(node) > max_degree:
                            max_degree = subgraph.degree(node)
                            center_node = [node]
                            res[node_1] = center_node[0]
                else:
                    res[node_1] = center_node[0]


            except:
                center_node = list(subgraph.nodes())
                max_degree = 0
                for node in center_node:
                    if subgraph.degree(node) > max_degree:
                        max_degree = subgraph.degree(node)
                        center_node = [node]
                res[node_1] = center_node[0]
            rest_nodes = subgraph.nodes() - center_node[0:1]
            subgraph = nx.subgraph(subgraph, rest_nodes)
        return res

    def find_qpu_community(self, partition, remote_wig):
        self.create_new_weight_graph()
        c = self.get_communities()
        # check if any commubity can run the job
        # for each community, first calulate total qubits for each community
        total_qubits_list = [sum([self.network.nodes[node]['qpu'].available_qubits for node in community]) for community
                             in c]
        # find one community that has enough qubits to run the job
        # count is a dictionary, the key is the partition index, the value is the number of qubits
        count = Counter(partition)
        # check if one community can run the job, if such community doesn't exist, merge two communities
        result = {}
        res = self.test_sub_isomorphic(self.network, remote_wig)
        if res:
            # get reverse mapping of res
            res = {v: k for k, v in res.items()}
            _, weighted_lenghh = self.compute_weighted_length(partition, remote_wig, res)
            return res, weighted_lenghh
        for i in range(len(total_qubits_list)):
            if sum(count.values()) <= total_qubits_list[i] and len(c[i]) >= len(count):
                #         map the node of remote_wig to the node in community
                sub_graph = self.network.subgraph(c[i])
                # optimal_solution = self.map_qpu_to_community(sub_graph, remote_wig)
                #       sort nodes in remote_wig by degree*weight of edge
                sorted_nodes = sorted(remote_wig.nodes(), key=lambda x: remote_wig.degree(x, weight='weight'),
                                      reverse=True)
                nx.draw(sub_graph, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000,
                        font_size=15)
                ##plt.show()
                nx.draw(remote_wig, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000,
                        font_size=15)
                ##plt.show()
                fig, axes = plt.subplots(1, 2, figsize=(22, 12))  # Adjust the figsize as needed

                # check whether remote wig is subgraph-iso to the subgraph
                for node in sorted_nodes:
                    #     find most suitable node in the community to mao the node in remote_wig

                    try:
                        center_node = nx.center(sub_graph)
                        if len(center_node) > 1:
                            betweenness_centrality = nx.betweenness_centrality(sub_graph)
                            # Find the node with the highest Betweenness Centrality
                            max_betweenness_node = max(betweenness_centrality, key=betweenness_centrality.get)
                            center_node = [max_betweenness_node]

                    except:

                        center_node = list(sub_graph.nodes())
                    try:
                        result[node] = center_node[0]
                    except:
                        print(center_node)

                    rest_nodes = sub_graph.nodes() - center_node[0:1]
                    # print(rest_nodes)
                    sub_graph = nx.subgraph(sub_graph, rest_nodes)
                #        plot the subgraph

                res = self.map_qpu_to_community(c[i], remote_wig)
                original_sum_test, weighted_sum_test = self.compute_weighted_length(partition, remote_wig, res)
                # print(sorted_nodes)
                original_sum, weighted_lenghh = self.compute_weighted_length(partition, remote_wig, result)
                if weighted_sum_test < weighted_lenghh:
                    return res, weighted_sum_test
                return result, weighted_lenghh

        # merge more than one community
        for r in range(2, len(c) + 1):
            for combo in combinations(c, r):
                total_qubits_combo = sum(
                    sum(self.network.nodes[node]['qpu'].available_qubits for node in community) for community in combo)
                total_qpu_combo = sum(len(community) for community in combo)
                if sum(count.values()) <= total_qubits_combo and len(count) <= total_qpu_combo:
                    merged_community = [node for community in combo for node in community]
                    sub_graph = self.network.subgraph(merged_community)
                    sorted_nodes = sorted(remote_wig.nodes(), key=lambda x: remote_wig.degree(x, weight='weight'),
                                          reverse=True)
                    for node in sorted_nodes:
                        #     find most suitable node in the community to mao the node in remote_wig
                        try:
                            center_node = nx.center(sub_graph)
                            if len(center_node) > 1:
                                betweenness_centrality = nx.betweenness_centrality(sub_graph)
                                # Find the node with the highest Betweenness Centrality
                                max_betweenness_node = max(betweenness_centrality, key=betweenness_centrality.get)
                                center_node = [max_betweenness_node]
                        except:
                            betweenness_centrality = nx.betweenness_centrality(sub_graph)
                            # Find the node with the highest Betweenness Centrality
                            try:
                                max_betweenness_node = max(betweenness_centrality, key=betweenness_centrality.get)
                            except:
                                print("no betweenness centrality")
                            center_node = [max_betweenness_node]
                        result[node] = center_node[0]
                        rest_nodes = sub_graph.nodes() - center_node[0:1]
                        # print(rest_nodes)
                        sub_graph = nx.subgraph(sub_graph, rest_nodes)
                    original_sum, weighted_lenghh = self.compute_weighted_length(partition, remote_wig, result)
                    return result, weighted_lenghh

        # print(c)

    def map_qpu_to_community_LP(self, community, remote_wig):
        all_pairs_shortest_path_length = dict(nx.all_pairs_dijkstra_path_length(community))
        problem = pulp.LpProblem("Optimal_Node_Mapping", pulp.LpMinimize)

        # Nodes in remote_wig and community_graph
        remote_wig_nodes = list(remote_wig.nodes())
        community_nodes = list(community.nodes())

        n = len(remote_wig_nodes)  # Number of nodes in remote_wig
        m = len(community_nodes)  # Number of nodes in the community

        # Decision variables: x[i][j] = 1 if node i in 'remote_wig' is mapped to node j in 'community'
        x = pulp.LpVariable.dicts("x", (range(n), range(m)), cat=pulp.LpBinary)

        # Objective function
        objective = pulp.lpSum([
            remote_wig.edges[u, v]['weight'] * all_pairs_shortest_path_length[community_nodes[j]][community_nodes[l]] *
            x[i][j] * x[k][l]
            for i, u in enumerate(remote_wig_nodes)
            for j in range(m)
            for k, v in enumerate(remote_wig_nodes)
            for l in range(m)
            if u != v and (u, v) in remote_wig.edges
        ])

        problem += objective

        # Constraint: Each node in 'remote_wig' must be mapped to exactly one node in 'community'
        for i in range(n):
            problem += pulp.lpSum(x[i][j] for j in range(m)) == 1

        # Solve the problem
        problem.solve()

        # Check if the problem has an optimal solution
        if problem.status == pulp.LpStatusOptimal:
            # Constructing the final mapping dictionary
            final_mapping = {}
            for i in range(n):
                for j in range(m):
                    if pulp.value(x[i][j]) == 1:  # If the decision variable is part of the optimal solution
                        final_mapping[remote_wig_nodes[i]] = community_nodes[j]

            # print("Final Mapping:", final_mapping)
        else:
            print("No optimal solution found.")

        pass

    def compute_weighted_length(self, partition, wig, result):
        sum = 0
        orginal_sum = 0
        for edge in wig.edges():
            node_1, node_2 = edge
            try:
                qpu_1, qpu_2 = result[node_1], result[node_2]
            except:
                print('no result')
            distance = nx.shortest_path_length(self.network, qpu_1, qpu_2)
            sum += wig[edge[0]][edge[1]]['weight'] * distance
            orginal_sum += wig[edge[0]][edge[1]]['weight']

        return orginal_sum, sum

    def enumerate_find_qpus(self, partition):
        #     enumberate all qpus available qubits
        qpus_available_qubits = {qpu.qpuid: qpu.available_qubits for qpu in self.qpus}
        count = Counter(partition)
        res = {}
        max_available_qubits = max(qpus_available_qubits.values())
        if max_available_qubits < max(count.values()):
            return None
        used_qpus = set()
        for job_part, required_qubits in count.items():
            #     list all possible qpus that have more than required qubits and sort them by available qubits
            possible_qpus = [qpu for qpu in qpus_available_qubits.keys() if
                             qpus_available_qubits[qpu] >= required_qubits]
            possible_qpus.sort(key=lambda x: qpus_available_qubits[x], reverse=True)
            for qpu in possible_qpus:
                if qpu not in used_qpus:
                    res[job_part] = qpu
                    used_qpus.add(qpu)
                    break
        return res
        pass

    # another way to find possible qpus is needed, could return a set of possible qpus and with shortest path between each two is minimized
    def bfs_find_qpus(self, start_node, partition, remote_wig):
        # use bfs to find qpus that can run this job(consider greedy qpu usage)
        visited = set()
        count = Counter(partition)
        queue = [start_node]
        qpu_combination = {}
        distribution = list(count.values())
        distribution.sort(reverse=True)
        while queue:
            node = queue.pop(0)
            if node not in visited:
                visited.add(node)
                qpu = self.network.nodes[node]['qpu']
                available_qubits = qpu.available_qubits
                for key, value in count.items():
                    if value <= available_qubits:
                        qpu_combination[key] = node
                        count.pop(key)
                        distribution.remove(value)
                        if len(count.keys()) == 0:
                            original_sum, weighted_length = self.compute_weighted_length(partition, remote_wig,
                                                                                         qpu_combination)
                            return qpu_combination, weighted_length

                        break
                for neighbor in self.network.neighbors(node):
                    if neighbor not in visited:
                        queue.append(neighbor)

        return

    def set_collaboration_data(self):
        for i in range(len(self.qpus)):
            self.qpus[i].collaboration_data = self.generate_collboration_data(i)

    def generate_collboration_data(self, qpu_id):
        num_qubits = self.qpu_qubit_num
        distributions = {
            "T1": {"mean": 100, "std": 20},  # Replace with your distribution parameters
            "T2": {"mean": 150, "std": 30},
            "Frequency": {"mean": 5, "std": 0.1},
            "Anharmonicity": {"mean": -0.34, "std": 0.01},
            "Readout error": {"mean": 0.01, "std": 0.005},
            "CNOT error": {"mean": 0.005, "std": 0.001},
            "Gate time": {"mean": 500, "std": 50},
            # ... other distributions
        }
        data = {
            "Qubit": np.arange(num_qubits),
            "T1 (us)": np.random.normal(distributions["T1"]["mean"], distributions["T1"]["std"], num_qubits),
            "T2 (us)": np.random.normal(distributions["T2"]["mean"], distributions["T2"]["std"], num_qubits),
            "Frequency (GHz)": np.random.normal(distributions["Frequency"]["mean"], distributions["Frequency"]["std"],
                                                num_qubits),
            "Anharmonicity (GHz)": np.random.normal(distributions["Anharmonicity"]["mean"],
                                                    distributions["Anharmonicity"]["std"], num_qubits),
            "Readout assignment error": np.random.normal(distributions["Readout error"]["mean"],
                                                         distributions["Readout error"]["std"], num_qubits),
            # ... other single qubit parameters
        }
        cnot_errors = []
        gate_times = []
        for i in range(num_qubits):
            cnot_error = []
            gate_time = []
            for j in range(num_qubits):
                if i != j:  # Assuming no self-pairing
                    cnot_error.append(
                        f"{i}_{j}:{np.random.normal(distributions['CNOT error']['mean'], distributions['CNOT error']['std'])}")
                    gate_time.append(
                        f"{i}_{j}:{np.random.normal(distributions['Gate time']['mean'], distributions['Gate time']['std'])}")
            cnot_errors.append("; ".join(cnot_error))
            gate_times.append("; ".join(gate_time))

        data["CNOT error"] = cnot_errors
        data["Gate time (ns)"] = gate_times
        collboration_data = pd.DataFrame(data)


def create_random_topology(num_qpus, probability):
    return nx.erdos_renyi_graph(num_qpus, probability)


def main():
    num_qpus = 10
    probability = 0.5
    cloud = qCloud(num_qpus, create_random_topology, probability)

    def find_qpu_combinations(qpus, circuit_size, current_combination=[], current_index=0, current_qubits=0):
        solutions = []
        if current_qubits >= circuit_size:
            return [current_combination]

        for i in range(current_index, len(qpus)):
            updated_combination = current_combination + [qpus[i]]
            updated_qubits = current_qubits + qpus[i].ncp_qubits
            solutions.extend(find_qpu_combinations(qpus, circuit_size, updated_combination, i + 1, updated_qubits))

        return solutions

    # qpu_combinations = find_qpu_combinations(cloud.qpus, circuit_size)

    # Display the combinations
    # for combo in qpu_combinations:
    #     combo_ids = [qpu.qpuid for qpu in combo]
    #     print(f"Combination: {combo_ids}, Total Qubits: {sum(qpu.ncp_qubits for qpu in combo)}")
    nx.draw(cloud.network, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=15)
    a = cloud.network.nodes[0]['qpu']
    # Show the plot
    #plt.show()
    # print(len(cloud.network.nodes[0]['available_qubits']))
    # print(cloud.network.nodes[0]['qpu'].ncm_qubits)


# Your main code goes here


if __name__ == "__main__":
    main()
