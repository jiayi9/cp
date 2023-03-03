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

df['end'].values[0]

colors = {}
colors['A'] = 'rgb(29, 133, 60)' #specify the color for the 'planned' schedule bars
colors['B'] = 'rgb(245, 148, 22)'  #specify the color for the 'actual' schedule bars


fig = px.timeline(
    df,
    x_start="start",
    x_end="end",
    y="task",
    color='type',
    color_discrete_map = colors,
    hover_name="desp",
    range_x=['2021-05-01 00:00:00', '2021-05-06 24:00:00']
    )

#fig.add_vline(x=df['end'].values[0])
#fig.add_hline(y=df['end'].values[0])
fig.update_layout(
    shapes=[
        dict(
            type='line',
            yref='paper',
            y0=0, y1=1,
            xref='x',
            x0='2021-05-02 01:00:00', x1='2021-05-02 01:00:00',
            line=dict(
                color="MediumPurple",
                width=3,
                dash="dot"
            )
        ),
        dict(
            type='line',
            yref='paper',
            y0=0, y1=1,
            xref='x',
            x0='2021-05-03', x1='2021-05-03',
            line=dict(
                color="MediumPurple",
                width=3,
                dash="solid"
            )
        )
    ]
)

fig.add_vrect(x0="2021-05-01", x1="2021-05-02",
              annotation_text="Unofficial<br>Summertime<br>in USA<br>(Memorial Day to<br>Labor Day)", annotation_position="top right",
              annotation_font_size=11,
              annotation_font_color="Green",
              fillcolor="yellow", opacity=0.25, line_width=0)
# Make a horizontal highlight section
fig.add_hrect(y0=0, y1=0.5,
              annotation_text="Observed data<br>of interest", annotation_position="top right",
              annotation_font_size=11,
              annotation_font_color="Black",
              fillcolor="red", opacity=0.25, line_width=0)

fig.update_layout(
                title='Project Plan Gantt Chart',
                bargap=0.1,
                # width=2000,
                # height=1000,
                xaxis_title="time",
                yaxis_title="tasks",
                title_x=0.5,
                legend_title="this is a legend",
                legend = dict(orientation = 'v', xanchor = "center", x = 0.92, y= 0.98), #Adjust legend position
            )

fig.update_xaxes(tickangle=-5,side ="top",  tickfont=dict(family='Rockwell', color='blue', size=15))


fig.update_yaxes(autorange="reversed")          #if not specified as 'reversed', the tasks will be listed from bottom up
fig.data[1].width=0.5 # update the width of the 'Actual' schedule bars (the second trace of the figure)
fig.show()

