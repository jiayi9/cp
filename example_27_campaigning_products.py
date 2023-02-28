from ortools.sat.python import cp_model
from time import time
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import string

model = cp_model.CpModel()


def generate_task_data(num_of_products, num_of_tasks_per_product):
    """ Generate tasks of products (no more than 26 products) """
    products = string.ascii_uppercase[0:num_of_products]
    total_num_of_tasks = num_of_tasks_per_product*num_of_products
    tasks = {x for x in range(total_num_of_tasks)}
    task_to_product = {}
    for product_idx, product in enumerate(products):
        task_to_product.update({
            product_idx*num_of_tasks_per_product+task_idx: product for task_idx in range(num_of_tasks_per_product)
        })
    return tasks, task_to_product


