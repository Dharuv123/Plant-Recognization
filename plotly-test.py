import plotly.graph_objects as go
import numpy as np

colorscale = [[0, '#FAEE1C'], [0.33, '#F3558E'], [0.66, '#9C1DE7'], [1, '#581B98']]
scatter = go.Scatter(
    y=np.random.randn(500),
    mode='markers',
    marker=dict(
        size=16,
        color=np.random.randn(500),
        colorscale=colorscale,
        showscale=True
    )
)

fig = go.Figure(data=[scatter])
fig.show()
