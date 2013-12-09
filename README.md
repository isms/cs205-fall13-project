Parallelizing an ILP Assignment Problem with MPI
================================================


Background
----------

This repo has the code for a Fall 2013 [CS 205](http://iacs-courses.seas.harvard.edu/courses/cs205/) final project, where we model an Integer Linear Program (ILP) assignment problem, and then parallelize with MPI using several different schemes.

For more information about the project including background information and results, see [the site](http://isms.github.io/cs205-fall13-project/).


Requirements
------------

The requirements for this repo are in `requirements.txt`. In a virtualenv, you should be able to run:

    pip install -r requirements.txt

Prior to installing [MPI](http://en.wikipedia.org/wiki/Message_Passing_Interface), you need to have the libraries for an implementation of MPI on your system. We used [Open MPI](http://www.open-mpi.org/).


Implementations
---------------

   * Normal, non-parallel version

     - File: `serial.py`
     - Usage: `python serial.py`

   * Simple implementation with no load balancing
     - File: `parallel_naive.py`
     - Usage: `mpiexec -n [num. processors] python parallel_naive.py`

   * Simple implementation with load balancing
     - File: `parallel_distributed.py`
     - Usage: `mpiexec -n [num. processors] python parallel_distributed.py`

   * Master/slave implementation with optional load balancing
     - File: `parallel_master_slave.py`
     - Usage: `mpiexec -n [num. processors] python parallel_master_slave.py [options]`
     - Notes: Options include verbose printing and minimum threshold for redistributing work. Run `python parallel_master_slave.py --help` to see all of the options.


Input
-----

The (anonymized) inputs from the school are text files in `data/`. The data in these files is parsed in and accessible through `settings.py`, which also contains other global settings for setting up the ILP.

Output
------

The output from running any of the implementations listed above will be a semicolon-delimited CSV with the following naming convention:

    output_2013.12.02-12.14_p256_t161.1719.csv

Contained in this filename is the date, the number of processors (256), and the total running time (161.1719 seconds). The columns of this file are as follows: 

   * `min_value_constraint`: the minimum value that was passed in to the model to be used as a constraint for the objective function.
   * `setup_time`: the amount of time spent in setting up the `pulp` LP model object.
   * `solve_time`: the amount of time spent finding a feasible solution for the model.
   * `eval_time`: the amount of time spent evaluating the non-linear fitness score of the ILP solution.
   * `total_time`: the total time in finding this solution.
   * `linear_obj`: the LP's objective function for this solution.
   * `nonlinear_obj`: the non-linear fitness score of the ILP solution.
   * `solution`: the actual solution found, with students' class placements separated by commas in the order of students (i.e., `3,1,2...` means that student 1 is in class 3, student 2 is in class 1, student 3 is in class 2, and so forth).