from ortools.sat.python import cp_model

# Initiate
model = cp_model.CpModel()

'''
task   product
1       A
2       A
'''

# 1. Data

tasks = {1, 2}
products = {'A'}
task_to_product = {1: 'A', 2: 'A'}

# product -> people mode -> time duration
processing_time = {
    'A': {
        2: 3,
        3: 2
    }
}

max_time = 20

# (task, people_mode)