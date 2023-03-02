# def generation_combinations_helper(lst, group_size, start_index):
#     if group_size == 0 or start_index == len(lst):
#         return [[]]
#
#     result = []
#
#     for i in range(start_index, len(lst)):
#         for combo in generation_combinations_helper(lst, group_size-1, i + 1):
#             result.append([lst[i]] + combo)
#     return result
#
#
# def generate_combinations(lst, max_group_size):
#     result=[]
#     for i in range(1,max_group_size+1):
#         result += generation_combinations_helper(lst, i, 0)
#
#     return result
#
#
# n = 3
# tasks = [i for i in range(n)]
# max_group_size = 2
#
# combinations = generate_combinations(tasks, max_group_size)
# for combo in combinations:
#     print(combo)

##################################################################################################################


def group_elements(lst, max_group_size):
    if not lst:
        return [[]]

    result = []
    for group_size in range(1, min(max_group_size, len(lst)) + 1):
        for i in range(len(lst) - group_size + 1):
            current_group = lst[i:i+group_size]
            for remaining_groups in group_elements(lst[i+group_size:], max_group_size):
                result.append([current_group] + remaining_groups)
    return result

# Example 1
lst = [1, 2, 3]
max_group_size = 2
x = group_elements(lst, max_group_size)
x[0][0]
x[0][0][0]
x[0][0][1]
x[0][0][2]


# Example 2
lst = [1, 2, 3, 4]
max_group_size = 2
group_elements(lst, max_group_size)
