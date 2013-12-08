import time
import csv
from datetime import datetime

from pulp import *

from nonlinear import evaluate_placement
from settings import *


def create_model(min_value):
    """
    Generate the `pulp` model of this ILP assignment problem.

    See generally: https://projects.coin-or.org/PuLP
    """

    # Initialize ILP and decision variables
    if MINIMIZE:
        prob = LpProblem("StudentClass", LpMinimize)
    else:
        prob = LpProblem("StudentClass", LpMaximize)

    # set up decision variables
    assign_vars = LpVariable.dicts("InClass", [(s, c) for s in STUDENTS
                                               for c in CLASSES], 0, 1, LpBinary)
    # add objective function
    prob += lpSum(assign_vars[(s, c)] * (c + 1) * (s + 1)
                  for s in STUDENTS for c in CLASSES)
    # add objective function constraint
    prob += lpSum(assign_vars[(s, c)] * (c + 1) * (s + 1)
                  for s in STUDENTS for c in CLASSES) >= min_value
    # add constraints for class size
    for c in CLASSES:
        prob += lpSum(assign_vars[(s, c)] for s in STUDENTS) >= CLASS_CAPACITY_MIN
        prob += lpSum(assign_vars[(s, c)] for s in STUDENTS) <= CLASS_CAPACITY_MAX

    # add constraints that assign each student to one class
    for s in STUDENTS:
        prob += lpSum(assign_vars[(s, c)] for c in CLASSES) == 1

    # add constraints for students who need to be in specific classes
    for s in STUDENTS:
        for c in CLASSES:
            # if INCLUDE_LOW_PRIORITY_CLASS_OKS false, only add in "high priority" cases
            if INCLUDE_LOW_PRIORITY_CLASS_OKS or ok_classes[s][5] == 1:
                prob += assign_vars[(s, c)] <= ok_classes[s][c + 1]

    # add constraints that guarantee that each student gets at least one requested friend
    for s in STUDENTS:
        for c in CLASSES:
            sum_friends = 0
            i = 1
            while i < 5 and not np.isnan(parent_requests[s][i]):
                sum_friends += assign_vars[(parent_requests[s][i] - 1, c)]
                i += 1
            prob += assign_vars[(s, c)] <= sum_friends

    # add constraints that guarantee that no students are placed with enemies
    for s in STUDENTS:
        for c in CLASSES:
            for e in xrange(NUM_ENEMIES_ALLOWED):
                if INCLUDE_LOW_PRIORITY_ENEMIES or enemies[s][e * 2 + 2] == 2:
                    if not np.isnan(enemies[s][e * 2 + 1]):
                        prob += assign_vars[(s, c)] + assign_vars[(enemies[s][e * 2 + 1] - 1, c)] <= 1

    # add constraints that guarantee that all recommended students are together
    for s in STUDENTS:
        for c in CLASSES:
            for r in xrange(NUM_RECOMMENDATIONS_ALLOWED):
                if INCLUDE_LOW_PRIORITY_RECOMMENDATIONS:
                    if not np.isnan(teacher_recs[s][r * 2 + 1]):
                        prob += assign_vars[(s, c)] <= assign_vars[(teacher_recs[s][r * 2 + 1] - 1, c)]
                else:
                    if teacher_recs[s][r * 2 + 2] == 2:
                        if not np.isnan(teacher_recs[s][r * 2 + 1]):
                            prob += assign_vars[(s, c)] <= assign_vars[(teacher_recs[s][r * 2 + 1] - 1, c)]

    # add constraints for making sure not too many troublemakers are together
    for c in CLASSES:
        prob += lpSum(assign_vars[(s, c)]
                      for s in STUDENTS
                      if (troublemakers[s][1] == 1)) <= MAX_NUM_TROUBLEMAKERS_PER_CLASS

    # add constraints for balancing out gender ratios
    for c in CLASSES:
        prob += lpSum(assign_vars[(s, c)] for s in STUDENTS
                      if (gender[s][1] == "M")) / MIN_MALE_PERCENTAGE >= \
                lpSum(assign_vars[(s, c)] for s in STUDENTS)
        prob += lpSum(assign_vars[(s, c)] for s in STUDENTS
                      if (gender[s][1] == "M")) / MAX_MALE_PERCENTAGE <= \
                lpSum(assign_vars[(s, c)] for s in STUDENTS)

    return assign_vars, prob


def find_solution(min_value=0):
    """
    Solve the ILP given a minimum objective function value.
    """

    # set start time
    start_iter_time = time.time()

    # get the model from our ILP factory
    assign_vars, prob = create_model(min_value)

    # end time counter for setup
    setup_time = time.time() - start_iter_time

    # solve the LP
    solve_start_time = time.time()
    solution_found = prob.solve()

    solve_time = time.time() - solve_start_time

    # if we didn't find a solution, return a row with just the time taken
    if solution_found != 1:
        return False, (min_value, setup_time, solve_time,
                       np.nan, np.nan, np.nan,
                       np.nan, (np.nan,))

    # evaluate solution based on nonlinear criteria
    eval_start_time = time.time()

    # sort solution array
    assignments = np.array([p for p in assign_vars.keys()
                            if assign_vars[p].varValue == 1])
    assignments = np.array(sorted(assignments, key=lambda entry: entry[0]))

    # adjust to 1-based numbers for easier human readability
    assignments += 1
    solution = ','.join(assignments[:, 1].astype(str))
    nonlinear_obj_value = evaluate_placement(assignments)

    # calculate time used
    eval_time = time.time() - eval_start_time
    total_time = setup_time + solve_time

    # append results of this iteration
    linear_value = value(prob.objective)
    return linear_value, (min_value, setup_time, solve_time,
                          eval_time, total_time, linear_value,
                          nonlinear_obj_value, solution)


def write_results(data_list, total_time, processors=1):
    """
    Write the resulting rows to a semicolon delimited .csv
    """

    # format the filename
    dt = datetime.now().strftime('%Y.%m.%d-%H.%M')
    filename = 'output_{}_p{:03d}_t{:0.4f}.csv'.format(dt, processors, total_time)

    # write out to informatively named csv file
    with open(filename, "w") as f:
        header = ';'.join(('min_value_constraint', 'setup_time', 'solve_time',
                           'eval_time', 'total_time', 'linear_obj', 'nonlinear_obj',
                           'solution'))
        f.write(header + '\n')
        writer = csv.writer(f, delimiter=';')
        writer.writerows(data_list)


if __name__ == '__main__':

    start = time.time()
    linear_value = -1.0
    data_list = []

    i = 0
    while linear_value:

        # find a new solution
        linear_value, row = find_solution(linear_value + 1)

        # add the current solution and print status
        if linear_value:
            data_list.append(row)
            print('{:0.2f}s: solved {} so far, current objective '
                  'function {}'.format(time.time()-start, i, linear_value))

        i += 1

    total_time = time.time() - start
    write_results(data_list, total_time, processors=1)
    "total time: ", total_time
