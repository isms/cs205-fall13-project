from __future__ import division
import optparse

from mpi4py import MPI
import numpy as np

from serial import find_solution, write_results
from settings import SEARCH_SPACE


def refine_work(work, p, new_min, verbose=False, split=False, split_min=4):
    """
    Filter remaining work based on newly found solution and split up
    remaining work if necessary to give processes that finish more
    quickly work from other processes.
    """
    # remove the minimum value numbers that have been skipped over
    work[p] = work[p][work[p] > new_min]

    # if option selected, divvy up work to processes that have run out
    if split and not work[p].size:

        # select the biggest queue remaining and split its work
        big_p, big_arr = sorted(work.iteritems(),
                                reverse=True,
                                key=lambda x: x[1].size)[0]

        # only take the other process's work if there are more than the min
        # number of values (this prevents each empty "stealing" the last
        # element in turn)
        if big_arr.size >= split_min:

            # print out status if desired
            if verbose:
                print('taking from process {} ({} left) to give '
                      'to process {}'.format(big_p, big_arr.size, p))

            # split it between the two processes
            work[big_p], work[p] = np.array_split(big_arr, 2)

    return work


def calculate_progress(work):
    """
    Calculate a % progress based on how much of the search space has been
    covered.
    """
    numerator = SEARCH_SPACE - queue_left(work)
    return numerator / SEARCH_SPACE


def queue_left(work):
    """
    Figure out how much work is left to in the queue.
    """
    return sum([arr.size for arr in work.values()])


def master(comm, min, max, verbose=True, split=False, split_min=4):
    """
    Assign work as appropriate to slave processes. When they run out of
    work in their range, take work from another process. When nobody has
    any work left to do, collect up all the leftovers and write out.
    """
    size = comm.Get_size()
    status = MPI.Status()
    start = MPI.Wtime()

    # placeholder for our output data
    data_list = []
    solved_so_far = 0

    # get all possible minimum values and split them up among the non-root processes
    process_min_ranges = np.array_split(np.arange(min, max), size - 1)
    work = {p + 1: work for p, work in enumerate(process_min_ranges)}

    # keep track of which slave processes are still alive
    processes_working = range(1, size)

    # setup: seed all processes with some work to start with
    for p in processes_working:
        comm.send(work[p].min(), dest=p)

    # collect work as it comes in and assign new minima as processes finish
    while True:
        # get the result of the slave process
        row = comm.recv(None, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        p = status.Get_source()
        linear_value = status.Get_tag()

        # update counters
        data_list.append(row)
        solved_so_far += 1

        # update work queues based on p's found linear value
        work = refine_work(work, p, linear_value,
                           verbose=verbose,
                           split=split,
                           split_min=split_min)

        # if there's no work left for process p, remove it from the working pool
        # (otherwise we'll mistakenly try to collect work from it at the end)
        if not work[p].size:
            processes_working.remove(p)
            print('master killing process {}, ({} left)'.
                  format(p, len(processes_working)))
            comm.send(-1, dest=p)
            # if nobody has any work left, break out
            if not queue_left(work):
                print('no work left; master breaking out of while loop, '
                      '{} processes still working'.format(len(processes_working)))
                break
        # ... otherwise, if there is work left for p, send it
        else:
            # pop the lowest value from process p's work queue and send it out
            new_min = work[p].min()
            comm.send(new_min, dest=p)
            work[p] = work[p][work[p] > new_min]

        if verbose or solved_so_far % 10 == 0:
            print('{:0.2f}s: solved {} so far ({:0.2%} of search space)'.
                  format(MPI.Wtime() - start, solved_so_far, calculate_progress(work)))

    # get the last few
    for _ in processes_working:
        row = comm.recv(None, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        data_list.append(row)
        solved_so_far += 1

    # kill the other processes
    for p in processes_working:
        comm.send(-1, dest=p)

    return data_list


def slave(comm):
    """
    Take work from master process until given the signal to shut down.
    """

    status = MPI.Status()
    rank = comm.Get_rank()

    print('slave {} working'.format(rank))
    while True:
        # get work from the master
        min_value = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)

        # die if sent a negative value
        if min_value < 0:
            break

        # do the actual computation for row i of the image
        linear_value, row = find_solution(min_value)
        comm.send(row, dest=0, tag=linear_value)


if __name__ == '__main__':

    # parse command line options
    parser = optparse.OptionParser()
    parser.add_option("-v", dest="verbose", action="store_true",
                      default=False,
                      help="verbose (print every solution instead of every 10)")
    parser.add_option("-s", dest="split", action="store_true",
                      default=False,
                      help="split work queues",)
    parser.add_option("--split-min", dest="split_min", type="int",
                      default=2,
                      help="threshold for splitting work queues")
    parser.add_option("--min", dest="min", type="float",
                      default=9927.0,
                      help="minimum objective value")
    parser.add_option("--max", dest="max", type="float",
                      default=11534.0,
                      help="maximum objective value")
    options, args = parser.parse_args()

    # get MPI data
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # if master, take change of work, otherwise prepare to do work
    if rank == 0:
        print 'master: started with size', size
        start_time = MPI.Wtime()

        results = master(comm, options.min, options.max,
                         verbose=options.verbose,
                         split=options.split,
                         split_min=options.split_min)

        end_time = MPI.Wtime()
        total_time = end_time - start_time
        write_results(results, total_time, processors=size)
        print 'total time', total_time
    else:
        slave(comm)
