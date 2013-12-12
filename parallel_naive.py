from __future__ import division
import optparse

from mpi4py import MPI
import numpy as np

from serial import find_solution, write_results


def process_interval(rank, work, verbose=False):
    """
    Iterate over the given interval and runs the function for finding solutions
    from `serial.py`.
    """

    print('process {} working on {} to {}\n'.format(rank, work.min(), work.max()))

    results = []
    i = 0
    while work.size:
        linear_value, row = find_solution(work.min())

        if not linear_value:
            break

        results.append(row)
        work = work[work > linear_value]
        i += 1

        if verbose:
            print('process {} just finished solution {}, objective function '
                  'value {}'.format(rank, i, linear_value))

    print('process {} finished\n'.format(rank))
    return results


if __name__ == '__main__':
    # parse command line options
    parser = optparse.OptionParser()
    parser.add_option("-v", dest="verbose", action="store_true",
                      default=False,
                      help="verbose (print every solution instead of every 10)")
    parser.add_option("--min", dest="min", type="float",
                      default=9927.0,
                      help="minimum objective value")
    parser.add_option("--max", dest="max", type="float",
                      default=11534.0,
                      help="maximum objective value")
    options, args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    work = np.arange(options.min, options.max+1)
    local_work = np.array_split(work, size)[rank]

    if rank == 0:
        start_time = MPI.Wtime()

        # do the Integer Linear Programming process on interval
        results = process_interval(rank, local_work, verbose=options.verbose)

        # collect results from other processors
        for i in range(1, size):
            results += comm.recv(source=i)

        total_time = start_time - MPI.Wtime()
        write_results(results, total_time, processors=size)
    else:
        results = process_interval(rank, local_work, verbose=options.verbose)
        comm.send(results, dest=0)
        
    
        
    



