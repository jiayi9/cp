
########################################################

# import plotly.express as px
#
# # Creating the Figure instance
# fig = px.line(x=[1, 2, 3], y=[1, 2, 3])
#
# # showing the plot
# fig.show()

########################################################


from plotly import optional_imports

import datetime as dt

import pandas as pd
import plotly.express as px
import plotly.graph_objs

def daterange(date1: dt.datetime, date2: dt.datetime):
    for n in range(int((date2 - date1).days) + 1):
        yield date1 + dt.timedelta(n)



def shade_weekends(fig: plotly.graph_objs.Figure, data):
    for weekend_start, weekend_end in data.weekends:
        fig.add_vrect(
            x0=weekend_start,
            x1=weekend_end,
            fillcolor="LightSalmon",
            opacity=0.3,
            layer="below",
            line_width=0,
        )
    return fig


def _update_late_trace(trace):
    """Check if a trace contains a late brand. If it does outline in red."""
    try:
        brand = trace.customdata[0][0]
    except (IndexError, TypeError):
        return
    if " Late" in brand:
        trace.update(marker={"line": {"width": 2, "color": "red"}})


category_orders = {
        "Brand": [
            "Brilinta",
            "Xigduo",
            "Brilinta_ODT",
            "Atacand",
            "Xigduo (optional batch)",
            "Brilinta Late",
            "Xigduo Late",
            "Brilinta_ODT Late",
            "Atacand Late",
            "Xigduo (optional batch) Late",
            "Campaign clean",
            "A-cleaning",
            "B-cleaning",
            "C-cleaning",
            "Factory Closure",
            "Machine Downtime",
        ],
        "Task Stage": [
            "Solution Granulation",
            "Granulation",
            "Mixing",
            "Mixing+compression",
            "Compression",
            "Solution Drage",
            "Coating",
        ],
}

color_discrete_sequence = [
        "#35155d",
        "#db6318",
        "#787878",
        "#069923",
        "#FDB44E",
        "#35155d",
        "#db6318",
        "#787878",
        "#069923",
        "#FDB44E",
        "#09bbe3",
        "#0981e3",
        "#0959e3",
        "#1009e3",
        "#b80606",
        "#b80606",  # Same colour for closures and downtime
]

df["Task Stage"] = df["Task Stage"].str.split("_").str[-1]

hover_data = None
if hover_data is None:
    if settings.plotly_show_full_hover:
            hover_data = list(df.columns)
    else:
            hover_data = list(
                [
                    "Brand",
                    "Task Stage",
                    "Start",
                    "End",
                    "Machine",
                    "Product",
                    "Batch Index",
                    "Batch ID",
                    "Due",
                    "Batch Start",
                    "Duration",
                ]
            )
df = df.sort_values(
    by=["Task Stage", y, facet_row], ascending=[False, False, True]
).reset_index(drop=True)
    # colours_list = list({*px.colors.qualitative.Light24, *px.colors.qualitative.Plotly, *px.colors.qualitative.Dark24})
fig = px.timeline(
        df,
        hover_name=hover_name,
        hover_data=hover_data,
        x_start=x_start,
        x_end=x_end,
        y=y,
        color=color,
        facet_row=facet_row,
        category_orders=category_orders,
        color_discrete_sequence=color_discrete_sequence,
)
fig.update_xaxes(dtick="d1", tickformat="%d-%b\n %Y-W%V")
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
fig.for_each_trace(_update_late_trace)
fig.update_yaxes(type="category")
fig.update_yaxes(matches=None)
fig = shade_weekends(fig, data)

fig.show()