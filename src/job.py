# job define the job submitted to the cloud
import os
import random
import numpy as np
from pytket import Circuit, OpType, qasm
import multiprocessing

global current_job_id
current_job_id = 0


class job_generator:
    def __init__(self):
        self.circuit_paths = {
            'small': self.get_qasm_files("/Users/mignotmesele/Downloads/qcloud/circuit/small"),
            'medium': self.get_qasm_files("/Users/mignotmesele/Downloads/qcloud/circuit/medium"),
            'large': self.get_qasm_files("/Users/mignotmesele/Downloads/qcloud/circuit/large"),
            'fixed': self.get_qasm_files("/Users/mignotmesele/Downloads/qcloud/circuit/test_whole"),
            'all_pool': self.get_qasm_files("/Users/mignotmesele/Downloads/qcloud/circuit/pool_2"),
            'qft': self.get_qasm_files("/Users/mignotmesele/Downloads/qcloud/circuit/pool_qft"),
            'qugan': self.get_qasm_files("/Users/mignotmesele/Downloads/qcloud/circuit/pool_qugan"),
            'arith': self.get_qasm_files("/Users/mignotmesele/Downloads/qcloud/circuit/pool_arith")
        }
    def generate_circuit_arith_pool(self):
        picked_circuit = self.select_random_circuit('arith')
        name = picked_circuit.split('/')[-1]
        print(name)
        circuit = None
        try:
            circuit = qasm.circuit_from_qasm(picked_circuit)
        except:
            print(picked_circuit)
        shots = random.uniform(50, 1000)
        # time = random.uniform(0, 10)
        # set time to 0 for now
        time = 0
        return name, circuit, shots, time
    def generate_circuit_qft_pool(self):
        picked_circuit = self.select_random_circuit('qft')
        name = picked_circuit.split('/')[-1]
        print(name)
        circuit = None
        try:
            circuit = qasm.circuit_from_qasm(picked_circuit)
        except:
            print(picked_circuit)
        shots = random.uniform(50, 1000)
        # time = random.uniform(0, 10)
        # set time to 0 for now
        time = 0
        return name, circuit, shots, time
    def generate_circuit_qugan_pool(self):
        picked_circuit = self.select_random_circuit('qugan')
        name = picked_circuit.split('/')[-1]
        print(name)
        circuit = None
        try:
            circuit = qasm.circuit_from_qasm(picked_circuit)
        except:
            print(picked_circuit)
        shots = random.uniform(50, 1000)
        # time = random.uniform(0, 10)
        # set time to 0 for now
        time = 0
        return name, circuit, shots, time

    def generate_circuit_from_dirctory(self, directory):
        qasm_files = self.circuit_paths[directory]
        job_list = []
        for file in qasm_files:
            try:
                circuit = qasm.circuit_from_qasm(file, maxwidth=1000)
            except Exception as e:
                print(f"Skipping invalid job: {e}")
                continue
            shots = random.uniform(50, 1000)
            time = random.uniform(0, 0)
            name = file.split('/')[-1]
            job_list.append(job(name, circuit, shots, time))
        return job_list
         

    def get_qasm_files(self, directory):
        qasm_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.qasm'):
                    qasm_files.append(os.path.join(root, file))
        return qasm_files

    def select_random_circuit(self, directory):
        # only returns the file ends with
        return random.choice(self.circuit_paths[directory]) if self.circuit_paths[directory] else None

    def generate_fixed_job(self, path):
        name = path.split('/')[-1]
        circuit = qasm.circuit_from_qasm(path)
        shots = 1000
        time = 0
        return job(name, circuit, shots, time)

    def generate_circuit_fixed_pool(self):
        picked_circuit = self.select_random_circuit('fixed')
        #print("this is the picked_circuit: ", picked_circuit)
        name = picked_circuit.split('/')[-1]
        circuit = None
        try:
            circuit = qasm.circuit_from_qasm(picked_circuit)
        except:
            #print(picked_circuit)
            pass
        shots = random.uniform(50, 1000)
        # time = random.uniform(0, 10)
        # set time to 0 for now
        time = 0
        return name, circuit, shots, time

    def generate_circuit(self, probability):
        # Ensure the sum of the probabilities is 1
        total_prob = sum(probability)
        if total_prob != 1:
            probability = [p / total_prob for p in probability]

        # Generate a random number using a uniform distribution
        distribution = np.random.uniform(0, 1)

        # Decide which circuit to use based on the probabilities
        if distribution < probability[0]:  # Probability for small circuit
            circuit = qasm.circuit_from_qasm(self.select_random_circuit('small'))
        elif distribution < probability[0] + probability[1]:  # Probability for medium circuit
            circuit = qasm.circuit_from_qasm(self.select_random_circuit('medium'))
        else:  # Remaining probability for large circuit
            circuit = qasm.circuit_from_qasm(self.select_random_circuit('large'))

        # Generate a random number of shots
        shots = random.uniform(50, 1000)
        # minimum time unit is ns,jobs are generated each 10 seconds(for now)
        # here the time is the time job has been generated, unit is s
        time = random.uniform(0, 10)
        return circuit, shots, time

    # generate large circuit just for testing
    def generate_large_circuit_job(self):
        circuit = qasm.circuit_from_qasm(self.select_random_circuit(self.large_circuits_path))
        shots = random.uniform(50, 1000)
        time = random.uniform(0, 10)
        return job(circuit, shots, time)

    def get_file_number(self, directory):
        return len(self.get_qasm_files(directory))

    # setp defines how many step it's generating
    # time_frame define the time_frame for each step
    # probability defines the probability of generating the size of a job
    def generate_job(self, n, time_frame, step, probability):
        jobs = []
        for i in range(n):
            circuit, shots, time = self.generate_circuit(probability)
            jobs.append(job(circuit, shots, time + time_frame * step))
        return jobs

    def generate_job_fixed_pool(self, n, time_frame, step):
        jobs = []
        for i in range(n):
            name, circuit, shots, time = self.generate_circuit_fixed_pool()
            jobs.append(job(name, circuit, shots, time + time_frame * step))
        return jobs
    def generate_job_qft_pool(self, n, time_frame, step):
        jobs = []
        for i in range(n):
            name, circuit, shots, time = self.generate_circuit_qft_pool()
            jobs.append(job(name, circuit, shots, time + time_frame * step))
        return jobs

    def generate_job_qugan_pool(self, n, time_frame, step):
        jobs = []
        for i in range(n):
            name, circuit, shots, time = self.generate_circuit_qugan_pool()
            jobs.append(job(name, circuit, shots, time + time_frame * step))
        return jobs
    
    def generate_job_arith_pool(self, n, time_frame, step):
        jobs = []
        for i in range(n):
            name, circuit, shots, time = self.generate_circuit_arith_pool()
            jobs.append(job(name, circuit, shots, time + time_frame * step))
        return jobs

    def generate_job_sequence(self, i, path):
        circuit_queue = []
        all_qasm_files = self.get_qasm_files(path)

        circuit = qasm.circuit_from_qasm(all_qasm_files[i])
        shots = random.uniform(50, 1000)
        time = random.uniform(0, 10)
        name = all_qasm_files[i].split('/')[-1]
        circuit_queue.append(job(name, circuit, 0, 0))
        return circuit_queue

    def generate_all_jobs(self):
        circuit_queue = []
        all_qasm_files = self.get_qasm_files("/Users/mignotmesele/Downloads/qcloud/circuit/pool_2")
        for i in range(len(all_qasm_files)):
            circuit = qasm.circuit_from_qasm(all_qasm_files[i])
            shots = random.uniform(50, 1000)
            time = random.uniform(0, 10)
            name = all_qasm_files[i].split('/')[-1]
            circuit_queue.append(job(name, circuit, 0, 0))
        return circuit_queue

    def generate_single_job(self, time_frame, step, probability):
        circuit, shots, time = self.generate_circuit(probability)
        return job(circuit, shots, time + time_frame * step)

    def generate_job_new(self, n, time_frame, step, probability):
        # Create a tuple of arguments for each job
        args = [(time_frame, step, probability) for _ in range(n)]

        # Create a pool of processes and generate jobs
        with multiprocessing.Pool() as pool:
            jobs = pool.starmap(self.generate_single_job, args)

        return jobs
    def nrandom_circuit(self, num_qubits, depth):
        circ = Circuit(num_qubits)
        for _ in range(depth):
            q = random.randint(0, num_qubits-1)
            gate = random.choice([OpType.H, OpType.X, OpType.Y, OpType.Z, OpType.CX])
            if gate == OpType.CX:
                target = random.randint(0, num_qubits-1)
                while target == q:
                    target = random.randint(0, num_qubits-1)
                circ.add_gate(gate, [q, target])
            else:
                circ.add_gate(gate, [q])
        circ.measure_all()
        return circ
    def generate_random_job(self, num_qubits=3, depth=5):
        jobs = []
        for _ in range(5):
            circuit = self.nrandom_circuit(num_qubits, depth)
            shots = random.randint(50, 1000)
            time = 0
            name = f"random_{np.random.randint(10000)}"
            jobs.append(job(name, circuit, shots, time))
        return jobs



class job:
    def __init__(self, name, circuit, shots, time):
        self.name = name
        self.circuit = circuit
        self.shots = shots
        self.time = time
        self.placement = None
        self.status = None
        global current_job_id
        self.id = current_job_id
        current_job_id += 1


class job_queue:
    def __init__(self, job_list):
        self.queue = job_list
        self.queue.sort(key=lambda x: x.time, reverse=False)


def main():
    # Your main code goes here
    print(random.choice([1, 2, 3]))
    a = job_generator()
    a.generate_circuit()


if __name__ == "__main__":
    main()
