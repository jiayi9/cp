import pandas as pd


def flatten(lst):
    """ flatten a list  """
    return [item for sublist in lst for item in sublist]


def group_elements(lst, max_group_size):
    """ the recursive subgroup generator """
    if not lst:
        return [[]]

    result = []
    for group_size in range(1, min(max_group_size, len(lst)) + 1):
        for i in range(len(lst) - group_size + 1):
            current_group = lst[i:i+group_size]
            for remaining_groups in group_elements(lst[i+group_size:], max_group_size):
                result.append([current_group] + remaining_groups)

    return result


def generate_campaigns(lst, max_group_size):
    """ the campaign pattern generator assuming the sequence of the tasks is fixed """

    result = group_elements(lst, max_group_size)
    output = []
    for sublist in result:
        if len(flatten(sublist)) == len(lst):
            output.append(sublist)
    return output


def run_example():
    """ a test """
    lst = [1, 2, 3, 4]
    max_group_size = 2
    campaign_patterns = generate_campaigns(lst, max_group_size)
    print(f"This is an example\n"
          f"Tasks: {lst}, max campaign size: {max_group_size} \n"
          f"All possible campaign patterns:")
    for campaign_pattern in campaign_patterns:
        print(campaign_pattern)


if __name__ == "__main__":

    run_example()

    task_sizes = list(range(2, 20))
    campaign_sizes = list(range(2, 7))

    L = []
    for task_size in task_sizes:
        print(task_size)
        for campaign_size in campaign_sizes:
            if task_size >= campaign_size:
                tasks = list(range(task_size))
                groups = generate_campaigns(tasks, campaign_size)
                tmp = [task_size, campaign_size, len(groups)]
                L.append(tmp)

    df = pd.DataFrame(L)
    df.columns = ['task_size', 'campaign_size', 'possible_grouping']
    print(df)

    df.to_csv("C:/Temp/campaigning_possibilities.csv", index=False)
