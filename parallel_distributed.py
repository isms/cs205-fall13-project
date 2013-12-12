"""
NOTE: This implementation is deprecated and strictly dominated in performance
      by `parallel_naive.py` and `parallel_master_slave.py`.
"""

from __future__ import division
from Queue import Queue
import random
from threading import Thread
import time

from mpi4py import MPI
import numpy as np

from serial import find_solution, write_results
from settings import ABSOLUTE_MIN, ABSOLUTE_MAX

# set up parameters
MAX_REJECTIONS = 15
MIN_REBALANCING_SIZE = 6
PAUSE_TIME = 2

# set up empty arrays that will be used for Irecv templates
LOOKING_FOR_WORK = np.array([101], dtype=np.float64)
REJECTION = np.array([102, 102], dtype=np.float64)
TERMINATION_INFO = np.array([103, 103], dtype=np.float64)

# set up variables for tags
LOOKING_FOR_WORK_TAG = 1
SENDING_WORK_BACK_TAG = 2
COLLECTING_INFO_TAG = 3
TERMINATION_TAG = 4


def check_for_termination_signal(comm, rank):
    """
    checks for a signal from processor 0 that lets the thread know
    that it should stop
    """
    terminate = np.empty_like(TERMINATION_INFO)
    req = comm.Irecv(terminate, source=0, tag=TERMINATION_TAG)
    status = MPI.Status()
    status.Set_source(0)

    if req.Test(status):
        req.Wait(status)
        print "terminating", rank
        return True

    req.Cancel()
    return False


def check_for_work_requests(comm, processor_list, q_end, min_value, end, work_left, always_reject=False):
    """
    Checks for requests for work from all other processors. If the
    always_reject variable is True, then it always tells the other core
    that it does not have work. If the always_reject variable is False,
    then if checks if the processor has an interval greater than
    MIN_REBALANCING_SIZE remaining. If so, then it splits off a chunk
    and sends it to the other processor. Otherwise, it tells the other
    processor that it has no work to give.
    """

    # the value of temp variable won't be used... just created to 
    # have something for Irecv to receive
    irecv_buffer = np.empty_like(LOOKING_FOR_WORK)
    req = comm.Irecv(irecv_buffer, source=MPI.ANY_SOURCE, tag=LOOKING_FOR_WORK_TAG)
    status = MPI.Status()

    if req.Test(status):
        signal_source = status.Get_source()

        if always_reject or work_left <= MIN_REBALANCING_SIZE:
            comm.Send(REJECTION, dest=signal_source, tag=SENDING_WORK_BACK_TAG)
        else:
            new_end = end - int((end - min_value) / 2) - 1
            q_end.put(new_end)
            other_interval_start = new_end + 1
            other_interval_end = end
            end = new_end
            work_left = int(end - min_value)
            data = np.array([other_interval_start, other_interval_end], dtype=np.float64)
            comm.Send(data, dest=signal_source, tag=SENDING_WORK_BACK_TAG)
    else:
        req.Cancel()

    return end, work_left


def listen_for_signals(comm, q_end, q_work_left, q_looking_for_work, q_start_end):
    rank = comm.Get_rank()
    size = comm.Get_size()

    processor_list = range(size)
    processor_list.pop(rank) # take current processor out of list
    continue_on = True

    # get information from the work thread on how much workload this
    # processor has before doing anything else
    work_left = 0
    while q_work_left.qsize < 1:
        time.sleep(1)

    min_value, end = q_work_left.get()

    while continue_on:
        # sleep for a little bit between each loop so that this thread
        # doesn't take up too much processing power
        time.sleep(PAUSE_TIME)

        # check if processor 0 sent a message to terminate
        if check_for_termination_signal(comm, rank):
            break

        # check the q_work_left queue for updates from the work thread
        if q_work_left.qsize() > 0:
            temp = q_work_left.get()
            if type(temp) == list:
                min_value = temp[0]
            else:
                min_value = temp

        work_left = int(end - min_value)

        # check if work requests came in from any other processors
        end, work_left = check_for_work_requests(comm, processor_list, q_end, min_value, end, work_left, False)

        # check the q_looking_for_work queue to see if the work thread
        # is done and needs more work
        if q_looking_for_work.qsize() > 0:
            q_looking_for_work.get()  # remove the element from the queue
            rejections = 0
            random_int = random.randint(0, size - 2)
            # Pick another processor at random and check if it has work
            # to give. If so, then take that work and give it to the
            # work thread. If not, then check the next processor. If you
            # get MAX_REJECTIONS (or size-2, whichever is less), then
            # stop asking.
            while rejections < min(MAX_REJECTIONS, size - 1):

                comm.Isend(LOOKING_FOR_WORK,
                           dest=processor_list[random_int],
                           tag=LOOKING_FOR_WORK_TAG)
                data = np.empty_like(REJECTION)
                req = comm.Irecv(data,
                                 source=processor_list[random_int],
                                 tag=SENDING_WORK_BACK_TAG)
                status = MPI.Status()
                status.Set_source(processor_list[random_int])

                # while waiting for a reply, check for incoming work
                # requests and tell all of them that you have no work
                # to give (since the last parameter in the
                # check_for_work_requests function is set to True). Also
                # check for termination requests from processor zero.
                while not req.Test(status):
                    check_for_work_requests(comm, processor_list, q_end, min_value,
                                            end, work_left, True)

                    if check_for_termination_signal(comm, rank):
                        continue_on == False
                        break

                req.Wait(status)
                if np.array_equal(data, REJECTION):
                    rejections += 1
                else:
                    min_value, end = data[0], data[1]
                    q_start_end.put([min_value, end])
                    break

                random_int = (random_int + 1) % (size - 1)

            # if you get enough rejections, tell the worker thread to
            # stop doing work
            if rejections == min(MAX_REJECTIONS, size - 1):
                q_start_end.put("Rejection")


def process_interval(comm, start, end, q_end, q_work_left, q_looking_for_work,
                     q_start_end, top_level=True, results=[]):
    """
    This process iterates over the given interval and runs
    the function for finding solutions from serial.py
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    processor_list = range(size)
    processor_list.pop(rank) # take current processor out of list
    min_value = start
    if top_level:
        q_work_left.put([min_value, end])  # let the other thread know the start and end

    # iterate through given interval... end may change if communication
    # thread receives work request
    while min_value <= end:
        linear_value, row = find_solution(min_value)
        print rank, 'found solutions'
        #~ print rank, min_value, end, end-min_value
        min_value = linear_value + 1
        if linear_value <= end:
            results.append(row)

        # put the min_value in this queue to let the other thread know
        # that the interval was updated 
        q_work_left.put(min_value)

        # check q_end to see if the other thread updated the interval's
        # end
        if q_end.qsize() > 0:
            end = q_end.get()

    # let the other thread know that this thread is done and is looking 
    # for more work
    q_looking_for_work.put(1)
    response = q_start_end.get()
    if response <> "Rejection":
        start, end = response
        results = process_interval(comm, start, end, q_end, q_work_left, q_looking_for_work,
                                   q_start_end, False, results)

    return results


if __name__ == '__main__':

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # create various queues that will be used to communicate between 
    # threads
    q_end = Queue() # used to tell the worker thread that it should use a new end point
    q_looking_for_work = Queue() # used to tell communication that work is needed
    q_work_left = Queue() # used to tell the comm thread how much work is left
    q_start_end = Queue() # used to deliver initial start and end of interval

    #setup and start the communication thread
    t_communication = Thread(target=listen_for_signals,
                             args=(comm, q_end, q_work_left,
                                   q_looking_for_work, q_start_end))
    t_communication.start()

    # send out intervals to each processor
    work = np.arange(ABSOLUTE_MIN, ABSOLUTE_MAX + 1)
    local_work = np.array_split(work, size)[rank]

    # get start and end intervals for core 0
    start = local_work.min()
    end = local_work.max()

    if rank == 0:
        print "Starting with ", size, "processors"
        status_string = "MAX_REJECTIONS, MIN_REBALANCING_SIZE, PAUSE_TIME:"
        print status_string, MAX_REJECTIONS, MIN_REBALANCING_SIZE, PAUSE_TIME, "with initial sleep adjustment"
        start_time = MPI.Wtime()

    # do the Integer Linear Programming processes on interval on core 0
    results = process_interval(comm, start, end, q_end, q_work_left,
                               q_looking_for_work, q_start_end)

    if rank == 0:
        # collect results from other processors
        for i in range(1, size):
            results += comm.recv(source=i, tag=COLLECTING_INFO_TAG)
            print "Got results from", i

        # send out messages to other processors to terminate the t_communication thread
        for i in range(size):
            comm.Isend(TERMINATION_INFO, dest=i, tag=TERMINATION_TAG)

        total_time = MPI.Wtime() - start_time
        write_results(results, total_time, processors=size)
        print "total time", total_time
    else:
        comm.send(results, dest=0, tag=COLLECTING_INFO_TAG)