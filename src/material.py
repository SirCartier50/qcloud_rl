class flow_scheduler:
    def __init__(self, des, qcloud, job, placement, EPR_p, name):
        self.des = des
        self.qcloud = qcloud
        self.job = job
        self.placement = placement
        self.remote_dag = placement.remote_dag
        self.EPR_p = EPR_p
        self.name = name

    # once one job is scheduled and find mapping, register it to the flow scheudler, and wait for flow-scheduling process

    # get remote DAG representation of a quantum circuit based on its
    #    based on job and placement, simulate running of one quantum circuit
    def simulate_run_dag_without_allocation(self):
        # define a table to record whther each gate is ready to be executed
        current_step = 0
        # to_go_table ={node: False for node in self.remote_dag.nodes()}
        graph_to_process = copy.deepcopy(self.remote_dag)
        priority = self._compute_priority_1(graph_to_process)
        input_layer = [node for node in self.remote_dag.nodes() if
                       self.remote_dag.nodes._nodes[node]['desc'] == 'Input']
        # status table records when each node is finished
        status_table = {node: 0 for node in self.remote_dag.nodes()}
        for node in input_layer:
            # to_go_table[node] = True
            graph_to_process.remove_node(node)
        # initialize the front layer,at this step  graph_to_process only doesn't contain input layer
        front_layer = [node for node in graph_to_process.nodes() if
                       graph_to_process.in_degree(node) == 0 and graph_to_process.out_degree(node) != 0]
        aa = [nx.ancestors(self.remote_dag, node) for node in front_layer]
        output_layer = [node for node in graph_to_process.nodes() if
                        self.remote_dag.nodes._nodes[node]['desc'] == 'Output']
        for node in output_layer:
            graph_to_process.remove_node(node)
        to_go_table = {node: 'unready' for node in self.remote_dag.nodes()}
        # generate all possible combination of two elements
        remoete_combo = list(combinations(Counter(self.placement.partition).keys(), 2))
        for node in input_layer:
            to_go_table[node] = 'finished'
        while len(graph_to_process.nodes()) != 0:
            # nx.draw(graph_to_process, with_labels=True)
            # plt.show()
            current_step += 1
            node_to_delet = []
            node_to_add = []
            #       while loop, each iteration node can run one step(row dice for onece)
            for node in front_layer:
                # check if the node is ready to go
                seed = random.uniform(0, 1)
                # node successfully executed
                # this step needs modified later, currentlt only fix a probability , but later needs resource allocation
                if seed >= 1 - self.EPR_p:
                    status_table[node] = current_step
                    node_to_delet.append(node)
                    to_go_table[node] = 'finished'
                    #                     check the status of its successor
                    for succ in graph_to_process.succ[node]:
                        #                 if all its predecessors are finished, then it is ready to go
                        if all([to_go_table[pred] == 'finished' for pred in graph_to_process.pred[succ]]):
                            to_go_table[succ] = 'ready'
                            if node not in output_layer:
                                node_to_add.append(succ)

                else:
                    pass
            for node in node_to_delet:
                front_layer.remove(node)
                graph_to_process.remove_node(node)
            for node in node_to_add:
                front_layer.append(node)
        curren_time = self.des.current_time + current_step
        # given partition and qpu mapping, compute the qpumapping
        count = Counter(self.placement.partition)
        qpu_mapping = self.placement.qpu_mapping[0]
        used_qpu = {value: count[key] for key, value in qpu_mapping.items()}
        # job scheduler shouldn't put event in the des directly, it should return the event and let des handle it
        finished_event = FinishedJob(curren_time, self.job, used_qpu, str(self.job.id) + ' finished')
        self.des.schedule_event(finished_event)

    def find_two_farthest_parents(self, graph, target_node):
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

    def simulate_run_dag_with_allocation_complex(self):
        # define a table to record whther each gate is ready to be executed
        # change key of dict from qubit to str of it
        node_to_int = {str(node): i for i, node in enumerate(self.placement.wig.nodes())}
        current_step = 0
        # to_go_table ={node: False for node in self.remote_dag.nodes()}
        graph_to_process = self.remote_dag.copy()
        print("finished deepcopy")
        priority = self._compute_priority_1(graph_to_process)
        input_layer = [node for node in self.remote_dag.nodes() if
                       self.remote_dag.nodes._nodes[node]['desc'] == 'Input']
        # status table records when each node is finished
        status_table = {node: 0 for node in self.remote_dag.nodes()}
        for node in input_layer:
            # to_go_table[node] = True
            graph_to_process.remove_node(node)
        # initialize the front layer,at this step  graph_to_process only doesn't contain input layer
        front_layer = [node for node in graph_to_process.nodes() if
                       graph_to_process.in_degree(node) == 0 and graph_to_process.out_degree(node) != 0]
        output_layer = [node for node in graph_to_process.nodes() if
                        self.remote_dag.nodes._nodes[node]['desc'] == 'Output']
        for node in output_layer:
            graph_to_process.remove_node(node)
        to_go_table = {node: 'unready' for node in self.remote_dag.nodes()}
        remote_combo = list(combinations(Counter(self.placement.partition).keys(), 2))
        for node in input_layer:
            to_go_table[node] = 'finished'
        parents = {node: [] for node in graph_to_process.nodes()}
        # first get the parent node for each node,
        # here the index of parent is the index in the dag
        # parents hold the partion for each node
        # {23 : [0,1]} means node 23 has two parents, one is in partition 0, the other is in partition 1
        for node in graph_to_process.nodes():
            parents[node] = self.find_two_farthest_parents(self.remote_dag, node)
        for node in parents.keys():
            # get the qubit index for each parent in the circuit
            # eg  0  -----> q[0]
            qubit_index = [self.placement.modified_circuit._dag_data[6][node] for node in parents[node]]
            parents[node] = [self.placement.partition[node_to_int[node]] for node in qubit_index]
        while len(graph_to_process.nodes()) != 0:
            current_step += 1
            node_to_delet = []
            node_to_add = []
            to_go_pool = {combo: [] for combo in remote_combo}
            # resources {partition_id : #number of communication qubits}
            # resources denotes the number of communication qubits for each partition，
            # qpu[1] is the partition id, qpu[0] is the qpu id
            resources = {qpu[1]: self.qcloud.network.nodes[qpu[0]]['qpu'].ncm_qubits for qpu in
                         self.placement.qpu_mapping[0]}
            allocation_result = {node: 0 for node in front_layer}
            competing_sets = {}
            for node in front_layer:
                #     check which queue should node go to
                for combo in to_go_pool.keys():
                    if parents[node] == list(combo) or parents[node] == list(combo)[::-1]:
                        to_go_pool[combo].append(node)
            # identify competition between different sets
            for tasks in to_go_pool:
                for task in tasks:
                    competing_sets.setdefault(task, set()).add(tasks)
            to_go_pool = sorted(to_go_pool.items(), key=lambda x: len(x[1]), reverse=True)
            to_go_pool = {x[0]: x[1] for x in to_go_pool}
            to_go_pool = {combo: 0 for combo in remote_combo}
            # sort partition key based on the number of nodes in the queue

            # allocate resources within machine sets
            for comb, nodes in to_go_pool.items():
                if nodes:
                    nodes.sort(key=lambda x: priority[x], reverse=True)
                    competing_keys = [key for key in to_go_pool.keys() if
                                      key != comb and set(key).intersection(set(comb)) and len(to_go_pool[key]) > 0]
                    available_resources = [min(resources[m] for m in comb) for comb in competing_keys]
                    #       find the key in to_go_pool that comptetes with the current key

                    resources_among_competition = {key: 0 for key in competing_keys}
                    pass

            for node in front_layer:
                seed = random.uniform(0, 1)
                if seed >= 0.3:
                    status_table[node] = current_step
                    node_to_delet.append(node)
                    to_go_table[node] = 'finished'
                    for succ in graph_to_process.succ[node]:
                        if all([to_go_table[pred] == 'finished' for pred in graph_to_process.pred[succ]]):
                            to_go_table[succ] = 'ready'
                            if node not in output_layer:
                                node_to_add.append(succ)
                else:
                    pass
            for node in node_to_delet:
                front_layer.remove(node)
                graph_to_process.remove_node(node)
            for node in node_to_add:
                front_layer.append(node)
        return current_step

    def simulate_run_dag_with_allocation_simple(self):
        # define a table to record whther each gate is ready to be executed
        # change key of dict from qubit to str of it
        node_to_int = {str(node): i for i, node in enumerate(self.placement.wig.nodes())}
        current_step = 0
        # to_go_table ={node: False for node in self.remote_dag.nodes()}
        graph_to_process = copy.deepcopy(self.remote_dag)
        print("finished deepcopy")
        priority = self._compute_priority_1(self.remote_dag)
        input_layer = [node for node in self.remote_dag.nodes() if
                       self.remote_dag.nodes._nodes[node]['desc'] == 'Input']
        # status table records when each node is finished
        status_table = {node: 0 for node in self.remote_dag.nodes()}
        for node in input_layer:
            # to_go_table[node] = True
            graph_to_process.remove_node(node)
        # initialize the front layer,at this step  graph_to_process only doesn't contain input layer
        front_layer = [node for node in graph_to_process.nodes() if
                       graph_to_process.in_degree(node) == 0 and graph_to_process.out_degree(node) != 0]
        output_layer = [node for node in self.remote_dag.nodes() if
                        self.remote_dag.nodes._nodes[node]['desc'] == 'Output']
        for node in output_layer:
            graph_to_process.remove_node(node)
        to_go_table = {node: 'unready' for node in self.remote_dag.nodes()}
        remote_combo = list(combinations(Counter(self.placement.partition).keys(), 2))
        for node in input_layer:
            to_go_table[node] = 'finished'
        parents = {node: [] for node in graph_to_process.nodes()}
        # first get the parent node for each node,
        # here the index of parent is the index in the dag
        # parents hold the partion for each node
        # {23 : [0,1]} means node 23 has two parents, one is in partition 0, the other is in partition 1
        for node in graph_to_process.nodes():
            parents[node] = self.find_two_farthest_parents(self.remote_dag, node)
        for node in parents.keys():
            # get the qubit index for each parent in the circuit
            # eg  0  -----> q[0]
            qubit_index = [self.placement.modified_circuit._dag_data[6][node] for node in parents[node]]
            parents[node] = [self.placement.partition[node_to_int[node]] for node in qubit_index]
        while len(graph_to_process.nodes()) != 0:
            current_step += 1
            node_to_delet = []
            node_to_add = []
            to_go_pool = {combo: [] for combo in remote_combo}
            # resources {partition_id : #number of communication qubits}
            # resources denotes the number of communication qubits for each partition，
            # qpu[1] is the partition id, qpu[0] is the qpu id
            resources = {par_idx: self.qcloud.network.nodes[qpu_idx]['qpu'].ncm_qubits for par_idx, qpu_idx in
                         self.placement.qpu_mapping[0].items()}
            allocation_result = {node: 0 for node in front_layer}
            competing_sets = {}
            for node in front_layer:
                #     check which queue should node go to
                for combo in to_go_pool.keys():
                    if parents[node] == list(combo) or parents[node] == list(combo)[::-1]:
                        to_go_pool[combo].append(node)
            # get inter-set allocation
            inter_sets_allocations = self.allocate_resources_between_sets_priority(to_go_pool, resources, priority)
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

            # print("hh")
            for node in front_layer:
                parent_nodes = parents[node]
                qpu_parent_nodes = [self.placement.qpu_mapping[0][par] for par in parent_nodes]
                distance = [nx.shortest_path_length(self.qcloud.network, qpu_parent_nodes[0], qpu_parent_nodes[1])]
                seed = random.uniform(0, 1)
                p = compute_probability(allocation_result[node], self.EPR_p)
                p_topo = compute_probability_with_distance(allocation_result[node], self.EPR_p, distance)
                if seed >= 1 - p_topo:
                    status_table[node] = current_step
                    node_to_delet.append(node)
                    to_go_table[node] = 'finished'
                    for succ in graph_to_process.succ[node]:
                        if all([to_go_table[pred] == 'finished' for pred in graph_to_process.pred[succ]]):
                            to_go_table[succ] = 'ready'
                            if node not in output_layer:
                                node_to_add.append(succ)
                else:
                    pass
            for node in node_to_delet:
                front_layer.remove(node)
                graph_to_process.remove_node(node)
            for node in node_to_add:
                front_layer.append(node)
        curren_time = self.des.current_time + current_step
        # given partition and qpu mapping, compute the qpumapping
        count = Counter(self.placement.partition)
        qpu_mapping = self.placement.qpu_mapping[0]
        used_qpu = {value: count[key] for key, value in qpu_mapping.items()}
        # job scheduler shouldn't put event in the des directly, it should return the event and let des handle it
        finished_event = FinishedJob(curren_time, self.job, used_qpu,
                                     str(self.job.id) + ' finished+ withallocation' + self.name)
        self.des.schedule_event(finished_event)

    # remove deepcopy of remote dag

    def simulate_run_dag_with_allocation_simple_1(self):
        # define a table to record whther each gate is ready to be executed
        # change key of dict from qubit to str of it
        node_to_int = {str(node): i for i, node in enumerate(self.placement.wig.nodes())}
        current_step = 0
        removed_nodes = set()
        # to_go_table ={node: False for node in self.remote_dag.nodes()}

        print("finished deepcopy")
        priority = self._compute_priority_1(self.remote_dag)
        print("finished priority")
        input_layer = [node for node in self.remote_dag.nodes() if
                       self.remote_dag.nodes._nodes[node]['desc'] == 'Input']
        # status table records when each node is finished
        status_table = {node: 0 for node in self.remote_dag.nodes()}

        def compute_front_layer():
            return [node for node in self.remote_dag.nodes() if node not in removed_nodes and
                    all(pred in removed_nodes for pred in self.remote_dag.predecessors(node)) and
                    self.remote_dag.out_degree(node) != 0]

        for node in input_layer:
            # to_go_table[node] = True
            removed_nodes.add(node)
            status_table[node] = 'finished'
        # initialize the front layer,at this step  graph_to_process only doesn't contain input layer
        front_layer = compute_front_layer()
        output_layer = [node for node in self.remote_dag.nodes() if
                        self.remote_dag.nodes._nodes[node]['desc'] == 'Output']
        for node in output_layer:
            removed_nodes.add(node)
        to_go_table = {node: 'unready' for node in self.remote_dag.nodes()}
        remote_combo = list(combinations(Counter(self.placement.partition).keys(), 2))
        for node in input_layer:
            to_go_table[node] = 'finished'
        print("finished input layer")

        def parallel_find_parents(node):
            # This will be your parallel task
            return node, self.find_two_farthest_parents(self.remote_dag, node)

        parents = {node: [] for node in self.remote_dag.nodes() if node not in removed_nodes}
        # first get the parent node for each node,
        # here the index of parent is the index in the dag
        # parents hold the partion for each node
        # {23 : [0,1]} means node 23 has two parents, one is in partition 0, the other is in partition 1

        # original code to get parents
        for node in parents:
            print(node)
            parents[node] = self.find_two_farthest_parents(self.remote_dag, node)

        print("finished find two farthest parents")
        # original code to get parents
        for node in parents.keys():
            print(node)
            qubit_index = [self.placement.modified_circuit._dag_data[6][node] for node in parents[node]]
            parents[node] = [self.placement.partition[node_to_int[node]] for node in qubit_index]

        # qubit_to_partition = {node: self.placement.partition[node_to_int[node]] for node in
        #                       self.placement.modified_circuit._dag_data[6]}
        #
        # for node in parents.keys():
        #     qubit_index = [self.placement.modified_circuit._dag_data[6][node] for node in parents[node]]
        #     parents[node] = [qubit_to_partition[node] for node in qubit_index]
        print("finish initiliazastion")
        while len(removed_nodes) < len(self.remote_dag.nodes()):
            print(current_step)
            current_step += 1
            node_to_delet = []
            node_to_add = []
            to_go_pool = {combo: [] for combo in remote_combo}
            # resources {partition_id : #number of communication qubits}
            # resources denotes the number of communication qubits for each partition，
            # qpu[1] is the partition id, qpu[0] is the qpu id
            resources = {par_idx: self.qcloud.network.nodes[qpu_idx]['qpu'].ncm_qubits for par_idx, qpu_idx in
                         self.placement.qpu_mapping[0].items()}
            allocation_result = {node: 0 for node in front_layer}
            for node in front_layer:
                #     check which queue should node go to
                for combo in to_go_pool.keys():
                    if parents[node] == list(combo) or parents[node] == list(combo)[::-1]:
                        to_go_pool[combo].append(node)
            # get inter-set allocation
            inter_sets_allocations = self.allocate_resources_between_sets_priority(to_go_pool, resources, priority)
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

            for node in front_layer:
                parent_nodes = parents[node]
                qpu_parent_nodes = [self.placement.qpu_mapping[0][par] for par in parent_nodes]
                distance = [nx.shortest_path_length(self.qcloud.network, source=qpu_parent_nodes[0],
                                                    target=qpu_parent_nodes[1])]
                seed = random.uniform(0, 1)
                p = compute_probability(allocation_result[node], self.EPR_p)
                p_topo = compute_probability_with_distance(allocation_result[node], self.EPR_p, distance)
                if seed >= 1 - p_topo:
                    status_table[node] = current_step
                    node_to_delet.append(node)
                    to_go_table[node] = 'finished'
                    for succ in self.remote_dag.successors(node):
                        if all(to_go_table[pred] == 'finished' for pred in self.remote_dag.predecessors(succ)):
                            to_go_table[succ] = 'ready'
                            if succ not in output_layer:
                                node_to_add.append(succ)

            for node in node_to_delet:
                removed_nodes.add(node)
                if node in front_layer:
                    front_layer.remove(node)

            for node in node_to_add:
                if node not in front_layer:
                    front_layer.append(node)
        curren_time = self.des.current_time + current_step
        # given partition and qpu mapping, compute the qpumapping
        count = Counter(self.placement.partition)
        qpu_mapping = self.placement.qpu_mapping[0]
        used_qpu = {value: count[key] for key, value in qpu_mapping.items()}
        # job scheduler shouldn't put event in the des directly, it should return the event and let des handle it
        finished_event = FinishedJob(curren_time, self.job, used_qpu,
                                     " " + str(self.job.id) + ' finished+ withallocation' + self.name)
        self.des.schedule_event(finished_event)

    def simulate_run_dag_with_allocation_simple_non_topo(self):
        # define a table to record whther each gate is ready to be executed
        # change key of dict from qubit to str of it
        node_to_int = {str(node): i for i, node in enumerate(self.placement.wig.nodes())}
        current_step = 0
        # to_go_table ={node: False for node in self.remote_dag.nodes()}
        graph_to_process = copy.deepcopy(self.remote_dag)
        priority = self._compute_priority_1(graph_to_process)
        input_layer = [node for node in self.remote_dag.nodes() if
                       self.remote_dag.nodes._nodes[node]['desc'] == 'Input']
        # status table records when each node is finished
        status_table = {node: 0 for node in self.remote_dag.nodes()}
        for node in input_layer:
            # to_go_table[node] = True
            graph_to_process.remove_node(node)
        # initialize the front layer,at this step  graph_to_process only doesn't contain input layer
        front_layer = [node for node in graph_to_process.nodes() if
                       graph_to_process.in_degree(node) == 0 and graph_to_process.out_degree(node) != 0]
        output_layer = [node for node in graph_to_process.nodes() if
                        self.remote_dag.nodes._nodes[node]['desc'] == 'Output']
        for node in output_layer:
            graph_to_process.remove_node(node)
        to_go_table = {node: 'unready' for node in self.remote_dag.nodes()}
        remote_combo = list(combinations(Counter(self.placement.partition).keys(), 2))
        for node in input_layer:
            to_go_table[node] = 'finished'
        parents = {node: [] for node in graph_to_process.nodes()}
        # first get the parent node for each node,
        # here the index of parent is the index in the dag
        # parents hold the partion for each node
        # {23 : [0,1]} means node 23 has two parents, one is in partition 0, the other is in partition 1
        for node in graph_to_process.nodes():
            parents[node] = self.find_two_farthest_parents(self.remote_dag, node)
        for node in parents.keys():
            # get the qubit index for each parent in the circuit
            # eg  0  -----> q[0]
            qubit_index = [self.placement.modified_circuit._dag_data[6][node] for node in parents[node]]
            parents[node] = [self.placement.partition[node_to_int[node]] for node in qubit_index]
        while len(graph_to_process.nodes()) != 0:
            current_step += 1
            node_to_delet = []
            node_to_add = []
            to_go_pool = {combo: [] for combo in remote_combo}
            # resources {partition_id : #number of communication qubits}
            # resources denotes the number of communication qubits for each partition，
            # qpu[1] is the partition id, qpu[0] is the qpu id
            resources = {par_idx: self.qcloud.network.nodes[qpu_idx]['qpu'].ncm_qubits for par_idx, qpu_idx in
                         self.placement.qpu_mapping[0].items()}
            allocation_result = {node: 0 for node in front_layer}
            competing_sets = {}
            for node in front_layer:
                #     check which queue should node go to
                for combo in to_go_pool.keys():
                    if parents[node] == list(combo) or parents[node] == list(combo)[::-1]:
                        to_go_pool[combo].append(node)
            # get inter-set allocation
            inter_sets_allocations = self.allocate_resources_between_sets_priority(to_go_pool, resources, priority)
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

            # print("hh")
            for node in front_layer:
                parent_nodes = parents[node]
                qpu_parent_nodes = [self.placement.qpu_mapping[0][par] for par in parent_nodes]
                distance = [nx.shortest_path_length(self.qcloud.network, qpu_parent_nodes[0], qpu_parent_nodes[1])]
                seed = random.uniform(0, 1)
                p = compute_probability(allocation_result[node], self.EPR_p)
                if seed >= 1 - p:
                    status_table[node] = current_step
                    node_to_delet.append(node)
                    to_go_table[node] = 'finished'
                    for succ in graph_to_process.succ[node]:
                        if all([to_go_table[pred] == 'finished' for pred in graph_to_process.pred[succ]]):
                            to_go_table[succ] = 'ready'
                            if node not in output_layer:
                                node_to_add.append(succ)
                else:
                    pass
            for node in node_to_delet:
                front_layer.remove(node)
                graph_to_process.remove_node(node)
            for node in node_to_add:
                front_layer.append(node)
        curren_time = self.des.current_time + current_step
        # given partition and qpu mapping, compute the qpumapping
        count = Counter(self.placement.partition)
        qpu_mapping = self.placement.qpu_mapping[0]
        used_qpu = {value: count[key] for key, value in qpu_mapping.items()}
        # job scheduler shouldn't put event in the des directly, it should return the event and let des handle it
        finished_event = FinishedJob(curren_time, self.job, used_qpu, str(self.job.id) + ' finished+ withallocation')
        self.des.schedule_event(finished_event)

    def simulate_run_dag_with_allocation_average(self):
        # define a table to record whther each gate is ready to be executed
        node_to_int = {str(node): i for i, node in enumerate(self.placement.wig.nodes())}
        current_step = 0
        # to_go_table ={node: False for node in self.remote_dag.nodes()}
        graph_to_process = copy.deepcopy(self.remote_dag)
        priority = self._compute_priority_1(graph_to_process)
        input_layer = [node for node in self.remote_dag.nodes() if
                       self.remote_dag.nodes._nodes[node]['desc'] == 'Input']
        # status table records when each node is finished
        status_table = {node: 0 for node in self.remote_dag.nodes()}
        for node in input_layer:
            # to_go_table[node] = True
            graph_to_process.remove_node(node)
        # initialize the front layer,at this step  graph_to_process only doesn't contain input layer
        front_layer = [node for node in graph_to_process.nodes() if
                       graph_to_process.in_degree(node) == 0 and graph_to_process.out_degree(node) != 0]
        aa = [nx.ancestors(self.remote_dag, node) for node in front_layer]
        output_layer = [node for node in graph_to_process.nodes() if
                        self.remote_dag.nodes._nodes[node]['desc'] == 'Output']
        for node in output_layer:
            graph_to_process.remove_node(node)
        to_go_table = {node: 'unready' for node in self.remote_dag.nodes()}
        # generate all possible combination of two elements
        remote_combo = list(combinations(Counter(self.placement.partition).keys(), 2))
        for node in input_layer:
            to_go_table[node] = 'finished'
        parents = {node: [] for node in graph_to_process.nodes()}
        for node in graph_to_process.nodes():
            parents[node] = self.find_two_farthest_parents(self.remote_dag, node)
        for node in parents.keys():
            # get the qubit index for each parent in the circuit
            # eg  0  -----> q[0]
            qubit_index = [self.placement.modified_circuit._dag_data[6][node] for node in parents[node]]
            parents[node] = [self.placement.partition[node_to_int[node]] for node in qubit_index]
        while len(graph_to_process.nodes()) != 0:
            current_step += 1
            node_to_delet = []
            node_to_add = []
            to_go_pool = {combo: [] for combo in remote_combo}
            for node in front_layer:
                #     check which queue should node go to
                for combo in to_go_pool.keys():
                    if parents[node] == list(combo) or parents[node] == list(combo)[::-1]:
                        to_go_pool[combo].append(node)
            # resources {partition_id : #number of communication qubits}
            # resources denotes the number of communication qubits for each partition，
            # qpu[1] is the partition id, qpu[0] is the qpu id
            resources = {par_idx: self.qcloud.network.nodes[qpu_idx]['qpu'].ncm_qubits for par_idx, qpu_idx in
                         self.placement.qpu_mapping[0].items()}
            allocation_result = {node: 0 for node in front_layer}
            inter_sets_allocations = self.allocate_resources_between_sets(to_go_pool, resources)
            #       while loop, each iteration node can run one step(row dice for onece)
            for comb, nodes in to_go_pool.items():
                if nodes:
                    total_resources = inter_sets_allocations[comb]
                    share = math.floor(total_resources / len(nodes))
                    remaining_resources = total_resources
                    for node in nodes:
                        if remaining_resources > share:
                            allocation_result[node] = share
                            remaining_resources -= share
                        else:
                            allocation_result[node] = remaining_resources
                            remaining_resources = 0
            # print("hh")
            for node in front_layer:
                # check if the node is ready to go
                seed = random.uniform(0, 1)
                # node successfully executed
                # this step needs modified later, currentlt only fix a probability , but later needs resource allocation
                if seed >= 1 - compute_probability(allocation_result[node], self.EPR_p):
                    status_table[node] = current_step
                    node_to_delet.append(node)
                    to_go_table[node] = 'finished'
                    #                     check the status of its successor
                    for succ in graph_to_process.succ[node]:
                        #                 if all its predecessors are finished, then it is ready to go
                        if all([to_go_table[pred] == 'finished' for pred in graph_to_process.pred[succ]]):
                            to_go_table[succ] = 'ready'
                            if node not in output_layer:
                                node_to_add.append(succ)

                else:
                    pass
            for node in node_to_delet:
                front_layer.remove(node)
                graph_to_process.remove_node(node)
            for node in node_to_add:
                front_layer.append(node)
        # only test for now
        # curren_time = self.des.current_time + current_step
        # given partition and qpu mapping, compute the qpumapping
        count = Counter(self.placement.partition)
        qpu_mapping = self.placement.qpu_mapping[0]
        used_qpu = {value: count[key] for key, value in qpu_mapping.items()}
        # job scheduler shouldn't put event in the des directly, it should return the event and let des handle it
        # finished_event = FinishedJob(curren_time, self.job, used_qpu, str(self.job.id) + ' finished +average')
        # only test for now
        # self.des.schedule_event(finished_event)
        return current_step
    def allocate_resources_between_sets(self, tasks, machine_resources):
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

    #
    def simulate_run_dag_with_allocation_simple_fast(self):
        # this fast version only retrieve dag data once and thus faster
        # define a table to record whther each gate is ready to be executed
        # change key of dict from qubit to str of it
        node_to_int = {str(node): i for i, node in enumerate(self.placement.wig.nodes())}
        current_step = 0
        # to_go_table ={node: False for node in self.remote_dag.nodes()}
        graph_to_process = copy.deepcopy(self.remote_dag)
        print("finished deepcopy")
        priority = self._compute_priority_1(self.remote_dag)
        print("finished priority")
        input_layer = [node for node in self.remote_dag.nodes() if
                       self.remote_dag.nodes._nodes[node]['desc'] == 'Input']
        # status table records when each node is finished
        status_table = {node: 0 for node in self.remote_dag.nodes()}
        for node in input_layer:
            # to_go_table[node] = True
            graph_to_process.remove_node(node)
        # initialize the front layer,at this step  graph_to_process only doesn't contain input layer
        front_layer = [node for node in graph_to_process.nodes() if
                       graph_to_process.in_degree(node) == 0 and graph_to_process.out_degree(node) != 0]
        output_layer = [node for node in self.remote_dag.nodes() if
                        self.remote_dag.nodes._nodes[node]['desc'] == 'Output']
        for node in output_layer:
            graph_to_process.remove_node(node)
        to_go_table = {node: 'unready' for node in self.remote_dag.nodes()}
        remote_combo = list(combinations(Counter(self.placement.partition).keys(), 2))
        for node in input_layer:
            to_go_table[node] = 'finished'
        parents = {node: [] for node in graph_to_process.nodes()}
        # first get the parent node for each node,
        # here the index of parent is the index in the dag
        # parents hold the partion for each node
        # {23 : [0,1]} means node 23 has two parents, one is in partition 0, the other is in partition 1
        for node in graph_to_process.nodes():
            parents[node] = self.find_two_farthest_parents(self.remote_dag, node)
        print("finished parent step 1")
        data_type = self.placement.modified_circuit._dag_data[6]
        for node in parents.keys():
            # get the qubit index for each parent in the circuit
            # eg  0  -----> q[0]
            qubit_index = [data_type[node] for node in parents[node]]
            parents[node] = [self.placement.partition[node_to_int[node]] for node in qubit_index]
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
            resources = {par_idx: self.qcloud.network.nodes[qpu_idx]['qpu'].ncm_qubits for par_idx, qpu_idx in
                         self.placement.qpu_mapping[0].items()}
            allocation_result = {node: 0 for node in front_layer}
            competing_sets = {}
            for node in front_layer:
                #     check which queue should node go to
                for combo in to_go_pool.keys():
                    if parents[node] == list(combo) or parents[node] == list(combo)[::-1]:
                        to_go_pool[combo].append(node)
            # get inter-set allocation
            inter_sets_allocations = self.allocate_resources_between_sets_priority(to_go_pool, resources, priority)
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

            # print("hh")
            for node in front_layer:
                print(self.remote_dag.nodes[node]['desc'])
                parent_nodes = set(parents[node])
                if len(parent_nodes) == 2:
                    qpu_parent_nodes = [self.placement.qpu_mapping[0][par] for par in parent_nodes]
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
                    p_topo = 1
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
        # only for test for now
        # curren_time = self.des.current_time + current_step
        # given partition and qpu mapping, compute the qpumapping
        count = Counter(self.placement.partition)
        qpu_mapping = self.placement.qpu_mapping[0]
        used_qpu = {value: count[key] for key, value in qpu_mapping.items()}
        # job scheduler shouldn't put event in the des directly, it should return the event and let des handle it
        # finished_event = FinishedJob(curren_time, self.job, used_qpu,
        #                              str(self.job.id) + ' finished+ withallocation' + self.name)
        # only test for now
        # self.des.schedule_event(finished_event)
        return current_step

    def _compute_priority(self, dag):
        priority = {}
        for node in dag.nodes():
            priority[node] = self.find_paths_to_leaves(dag, node)
        return priority

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

    def _compute_priority_2(self, dag):
        leaf_nodes = [node for node in dag.nodes() if dag.out_degree(node) == 0]

        # Use memoization to avoid redundant calculations
        @lru_cache(maxsize=None)
        def sum_paths_to_leaves(start_node):
            if start_node in leaf_nodes:
                return 1  # Length of the path from a leaf to itself is 1

            total_length = 0
            for neighbor in dag.successors(start_node):
                total_length += 1 + sum_paths_to_leaves(neighbor)
            return total_length

        return {node: sum_paths_to_leaves(node) for node in dag.nodes()}

    def find_paths_to_leaves(self, dag, start_node):
        # Identify all leaf nodes
        leaf_nodes = [node for node in dag.nodes() if dag.out_degree(node) == 0]

        # Find all paths from start_node to each leaf node
        all_paths = []
        for leaf in leaf_nodes:
            paths = list(nx.all_simple_paths(dag, start_node, leaf))
            all_paths.extend(paths)
        if not all_paths:
            return 0
        return max([len(path) for path in all_paths])
