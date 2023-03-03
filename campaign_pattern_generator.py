import pandas as pd


def flatten(l):
    return [item for sublist in l for item in sublist]


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


def generate_campaigns(lst, max_group_size):

    result = group_elements(lst, max_group_size)

    output = []
    for sublist in result:
        if len(flatten(sublist)) == len(lst):
            output.append(sublist)
    return output


# Example 1
lst = [1, 2, 3, 4]
max_group_size = 2
output = generate_campaigns(lst, max_group_size)

if __name__ == "__main__":

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
