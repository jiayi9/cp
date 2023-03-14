import pandas as pd
import plotly.express as px

df = pd.DataFrame({
    'task': ['task1', 'task2', 'task3'],
    'desp': ['xxxx', 'yyyy', 'zzzz'],
    'type': ['A', 'A', 'B'],
    'start': ['2021-05-01', '2021-05-01', '2021-05-02'],
    'end': ['2021-05-02', '2021-05-03', '2021-05-04'],
})

df['start'] = df['start'].astype('datetime64')
df['end'] = df['end'].astype('datetime64')


colors = {}
colors['A'] =  'rgb(29, 133, 60)' #specify the color for the 'planned' schedule bars
colors['B'] = 'rgb(245, 148, 22)'  #specify the color for the 'actual' schedule bars

fig = px.timeline(
    df,
    x_start="start",
    x_end="end",
    y="task",
    color='type',
    color_discrete_map = colors,
    hover_name="desp"
    )

fig.update_yaxes(autorange="reversed")          #if not specified as 'reversed', the tasks will be listed from bottom up
fig.data[1].width=0.5 # update the width of the 'Actual' schedule bars (the second trace of the figure)
fig.show()

#################################################################################