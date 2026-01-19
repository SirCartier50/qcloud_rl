# simple des define how the cloud works, and event class defines what could happen in the cloud
import heapq
from abc import ABC, abstractmethod
from abc import ABC, abstractmethod

class Event(ABC):
    def __init__(self, time):
        self.time = time
    @abstractmethod
    def execute(self, simulation):
        pass

    def __lt__(self, other):
        if isinstance(other, Event):
            return self.time < other.time
        return NotImplemented
#
class FinishedJob(Event):
    #FinisheJob is an event that denotes one job is finished
    # possible side effect: may be possible to scheule another job in the queue that hasn't been scheduled
    # thus it needs to check each element in the queue to see whether it is possible schedule another evennt
    def __init__(self, time, job, qpu, log, placement):
        super().__init__(time)
        self.time = time
        self.job= job
        self.qpu = qpu
        self.log = log
        self.placement = placement

    def execute(self, des):
        # when one job is finished, it will first free the resoruces it used on qpu
        # then it needs to check whether there is any job in the queue that can be scheduled
        for qpu_id in self.qpu.keys():
            qpu = des.cloud.network.nodes[qpu_id]['qpu']
            nqubits = self.qpu[qpu_id]
            qpu.free_qubits( nqubits,self.job)
        des.scheduler.job_queue.extend( des.scheduler.unscheduled_job)
        des.scheduler.unscheduled_job = []
        # check whether there is any job in the queue that can be scheduled
        if des.scheduler.job_queue:
            des.scheduler.schedule()
        pass
        
#
class generatingJob(Event):
    # generatingJob is an event that denotes that at each one fixed time, the cloud will receive a batch of jobs
    # it assume each batch of job arrives at the start of each time frame
    # incoming jobs will be added to the queue
    # possible side effect: job queue will be changed
    #                       need to schedule more jobs.
    def __init__(self, time,job_list):
        super().__init__(time)
        self.job_list = job_list

    def execute(self, des):
        des.scheduler.job_queue.extend(self.job_list)
        des.scheduler.schedule_choice()


class DES:
    # the input to DES should be
    def __init__(self,cloud,scheduler, logger = None):
        self.current_time = 0
        self.event_queue = []
        self.cloud = cloud
        self.scheduler = scheduler
        if not self.scheduler.job_queue:
            self.scheduler.job_queue=[]
        self.scheduler.des = self
        self.unpushed_event = []
        self.finished_job = []
        self.logger = logger

    def schedule_event(self, event):
        heapq.heappush(self.event_queue, event)

        print('hh')
    def run(self):
        while self.event_queue:
            event = heapq.heappop(self.event_queue)
            # check whether is a finshed job event
            if isinstance(event, FinishedJob):
                self.finished_job.append(event)
                print(str(event.time))
                print("finished job event"+str(event.time)+event.log)
                available_qubits = self.cloud.get_available_qubits()
                job_completion_time = event.time - event.placement.start_time
                self.logger.log(event.job.name, log=event.log, time=event.time,original_time = event.placement.dag_longest_path_length, qubits = available_qubits,start_time = event.placement.start_time, jct = job_completion_time)

            self.current_time = event.time
            event.execute(self)
