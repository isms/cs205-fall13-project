import numpy as np

parent_requests = np.genfromtxt("data/parent_requests.txt", delimiter='\t', skip_header=1)
enemies = np.genfromtxt("data/enemies.txt", delimiter='\t', skip_header=1)
ok_classes = np.genfromtxt("data/teacher_requests.txt", delimiter='\t', skip_header=1)
gender = np.genfromtxt("data/gender.txt", delimiter='\t', dtype=str, skip_header=1)
teacher_recs = np.genfromtxt("data/teacher_recommendations.txt", delimiter='\t', skip_header=1)
locations = np.genfromtxt("data/location.txt", delimiter='\t', dtype=str, skip_header=1)
troublemakers = np.genfromtxt("data/troublemakers.txt", delimiter='\t', skip_header=1)

# constants
NUM_ENEMIES_ALLOWED = 4
NUM_RECOMMENDATIONS_ALLOWED = 5
MAX_NUM_TROUBLEMAKERS_PER_CLASS = 4
MIN_MALE_PERCENTAGE = 0.4
MAX_MALE_PERCENTAGE = 0.67
INCLUDE_LOW_PRIORITY_CLASS_OKS = True
INCLUDE_LOW_PRIORITY_ENEMIES = True
INCLUDE_LOW_PRIORITY_RECOMMENDATIONS = False
NUMBER_OF_STUDENTS = 91
CLASSES = [0, 1, 2, 3]
NUM_CLASSES = len(CLASSES)
STUDENTS = range(NUMBER_OF_STUDENTS)
CLASS_CAPACITY_MAX = 23
CLASS_CAPACITY_MIN = 22
MINIMIZE = True

# search space for objective function
ABSOLUTE_MIN = 9927.0
ABSOLUTE_MAX = 11534.0
SEARCH_SPACE = ABSOLUTE_MAX - ABSOLUTE_MIN