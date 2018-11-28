import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from CorrelationProject.Dahs_test.data_loader import *
from CorrelationProject.Dahs_test.tools import *

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly import tools


xl = pd.read_excel("Base Dados_final.xlsx", "Base de dados")
xl2 = pd.read_excel("Base Dados_final.xlsx", "Sheet2")

intensity = xl2[[intense for intense in xl2.keys() if "Intensidade" in intense]]
psychosocial = xl2.iloc[:, 21:40]
scores1 = xl2[[scores for scores in xl2.keys() if ("score" in scores or "Score" in scores or "P_" in scores)]]

intense_nrm = normalize_df(intensity)
psychsl = normalize_df(psychosocial)
scores1 = normalize_df(scores1)
#norm xl2
# xl2_nrm = (xl2 - xl2.mean())/xl2.std()

SelectorGraph = select_type_column(xl)

#datatest
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/iris-data.csv')
classes=np.unique(df['class'].values).tolist()
class_code={classes[k]: k for k in range(3)}
color_vals=[class_code[cl] for cl in df['class']]
text=[df.loc[ k, 'class'] for k in range(len(df))]

pl_colorscale=[[0.0, '#19d3f3'],
               [0.333, '#19d3f3'],
               [0.333, '#e763fa'],
               [0.666, '#e763fa'],
               [0.666, '#636efa'],
               [1, '#636efa']]


trace1 = go.Splom(dimensions=[dict(label='sepal length',
                                 values=df['sepal length']),
                            dict(label='sepal width',
                                 values=df['sepal width']),
                            dict(label='petal length',
                                 values=df['petal length'])],
                text=text,
                #default axes name assignment :
                #xaxes= ['x1','x2',  'x3'],
                #yaxes=  ['y1', 'y2', 'y3'],
                marker=dict(color=color_vals,
                            size=7,
                            colorscale=pl_colorscale,
                            showscale=False,
                            line=dict(width=0.5,
                                      color='rgb(230,230,230)'))
                )

axis = dict(showline=True,
          zeroline=False,
          gridcolor='#fff',
          ticklen=4)

layout = go.Layout(
    title='Iris Data set',
    dragmode='select',
    width=600,
    height=600,
    autosize=False,
    hovermode='closest',
    plot_bgcolor='rgba(240,240,240, 0.95)',
    xaxis1=dict(axis),
    xaxis2=dict(axis),
    xaxis3=dict(axis),
    yaxis1=dict(axis),
    yaxis2=dict(axis),
    yaxis3=dict(axis)
)

fig1 = dict(data=[trace1], layout=layout)

app = dash.Dash()

app.layout = html.Div([
	html.Div(id="header_row", children=
	[
		html.H1(children="Dashboard for Data Analysis")
	]),

	html.Div(id="graph1", children=
	[
		dcc.Graph(
			id='selector_graph'),
		dcc.Dropdown(
			id="dropdown1",
			options=[
				{'label': 'Gender', 'value': 'gender'},
				{'label': 'Age', 'value': 'age'},
				{'label': 'Seniority', 'value': 'seniority'}
			],
			value = 'gender'
		)
	]),
html.Div(id="graph2", children=
	[
		dcc.Graph(
			id='selector_graph2',
			figure=fig1),
		dcc.Dropdown(
			id="dropdown2",
			options=[
				{'label': 'Gender', 'value': 'gender'},
				{'label': 'Age', 'value': 'age'},
				{'label': 'Seniority', 'value': 'seniority'}
			],
			value = 'gender'
		)
	]),
html.Div(id="graph_corr", children=
         [
	        dcc.Graph(
		        id='selector_graph_corr',
		        style={"width": 2000, "height":2000}
	        ),
	        dcc.Dropdown(
		        id="dropdown3",
		        options=[
					{'label': 'Male', 'value': 'male'},
					{'label': 'Female', 'value': 'female'}
		        ],
				value = 'male'
	        ),
			dcc.Dropdown(
		        id="dropdown4",
		        options=[
					{'label': 'Pain VS Psych', 'value': 'pp'},
					{'label': 'Pain VS Scores', 'value': 'ps1'},
					{'label': 'Psych VS Scores', 'value': 'ps2'},
		        ],
				value = 'male'
	        ),
         ])
])

def createHist(value, key):
	return go.Histogram(
		histfunc="sum",
		y=SelectorGraph[value][key]["arr7"]["freq"]["y"],
		x=SelectorGraph[value][key]["arr7"]["freq"]["x"],
		name=key,
	)

@app.callback(
    dash.dependencies.Output('selector_graph', 'figure'),
    [dash.dependencies.Input('dropdown1', 'value')])
def update_output(value):
	graphs = [createHist(value, key) for key in SelectorGraph[value].keys()]
	return {
		'data': graphs
	}

@app.callback(
	dash.dependencies.Output("selector_graph_corr", 'figure'),
	[dash.dependencies.Input("dropdown3", "value"), dash.dependencies.Input("dropdown4", "value")]
)
def update_corr_graph(value1, value2):
	if(value1=="male"):
		scores = scores1.loc[xl["Genero"] == 2]
		pain = intense_nrm.loc[xl["Genero"] == 2]
		psych_s = psychsl.loc[xl["Genero"] == 2]

	else:
		scores = scores1.loc[xl["Genero"] == 1]
		pain = intense_nrm.loc[xl["Genero"] == 1]
		psych_s = psychsl.loc[xl["Genero"] == 1]

	if(value2 =="pp"):
		z = abs(pain.corr()).values.tolist()
		z_text = np.around(z, decimals=2)

		fig = ff.create_annotated_heatmap(z=z, annotation_text=z_text, colorscale="Blues", showscale=True,
	                                  xgap=1.5, ygap=1.5)

	# data = [go.Heatmap(z=abs(df.corr()).values.tolist(), colorscale="Blues_r")]
	# data = fig

	return fig


if __name__ == '__main__':
    app.run_server(debug=True)