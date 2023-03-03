import pandas as pd
import plotly.express as px

df = pd.DataFrame({
    'order': ['001', '001', '002', '002', '003', '003'],
    'process': ['FBD', 'Compression', 'FBD', 'Compression', 'FBD', 'Compression'],
    'machine': ['FBD-1', 'TAB-1', 'FBD-1', 'TAB-1', 'FBD-2', 'TAB-1'],
    'task': ['A-1', 'A-2', 'B-1', 'B-2', 'C-1', 'C-2'],
    'product': ['M1', 'M1', 'M2', 'M2', 'M1', 'M1'],
    'start': ['2022-05-01 07:00:00', '2022-05-01 12:00:00', '2022-05-01 12:00:00', '2022-05-01 16:00:00', '2022-05-01 07:00:00', '2022-05-01 22:00:00'],
    'end': ['2022-05-01 12:00:00', '2022-05-01 16:00:00', '2022-05-01 16:00:00', '2022-05-01 22:00:00', '2022-05-01 11:00:00', '2022-05-02 03:00:00'],
})
df = df.assign(desp=df['product']+' '+df['order'])

df = df.sort_values(['machine'])


fig = px.timeline(
    df,
    x_start="start",
    x_end="end",
    y="machine",
    color='order',
    hover_name="order",
    text="desp",
    #insidetextanchor='middle', #['end', 'middle', 'start']
    #range_x=['2021-05-01 00:00:00', '2021-05-06 24:00:00']
    range_x=['2022-04-30', '2022-05-05'],
    )

fig.update_yaxes(categoryorder='array', categoryarray=['FBD-1', 'FBD-2', 'TAB-1'])

# For facetting only
# fig.update_yaxes(matches=None)
# fig.for_each_yaxis(lambda yaxis: yaxis.update(showticklabels=True))


# add process area
fig.add_hrect(
    y0=-0.5, y1=1.5,
    annotation_text="FBD",
    annotation_position="outside top right",
    annotation_font_size=11,
    annotation_font_color="Black",
    layer="below",
    fillcolor="gray",
    opacity=0.1,
    line_width=0
)

fig.update_layout(
    shapes=[
        dict(
            type='line',
            yref='paper',
            y0=0, y1=1,
            xref='x',
            x0='2022-05-01 07:00:00', x1='2022-05-01 07:00:00',
            line=dict(
                color="MediumPurple",
                width=5,
                dash="dot"
            ),
            layer="above",
        ),
        dict(
            type='line',
            yref='paper',
            y0=0, y1=1,
            xref='x',
            x0='2022-05-01 07:00:00', x1='2022-05-01 07:00:00',
            line=dict(
                color="MediumPurple",
                width=3,
                dash="dot"
            ),
            layer="above",
        ),
        dict(
            type='line',
            yref='paper',
            y0=0, y1=1,
            xref='x',
            x0='2022-05-02 07:00:00', x1='2022-05-02 07:00:00',
            line=dict(
                color="MediumPurple",
                width=3,
                dash="dot"
            ),
            layer="above",
        ),
        dict(
            type='line',
            yref='paper',
            y0=0, y1=1,
            xref='x',
            x0='2022-05-03 07:00:00', x1='2022-05-03 07:00:00',
            line=dict(
                color="MediumPurple",
                width=3,
                dash="dot"
            ),
            layer="above",
        ),
    ]
)


fig.update_layout(
        title='Schedule',
        bargap=0.1,
        width=2000,
        height=400,
        xaxis_title="time",
        yaxis_title="",
        title_x=0.5,
        legend_title="this is a legend",
        # legend = dict(orientation = 'v', xanchor = "center", x = 0.92, y= 0.98)
    )

# This is relative size
# for x in fig.data:
#     x.width=0.5
#fig.data[1].width=0.5
#fig.update_layout(barmode="group")


fig.update_yaxes(autorange="reversed")
fig.update_layout(dragmode='select')
fig.show()
