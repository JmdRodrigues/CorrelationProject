from Dahs_test.data_loader import *
from Dahs_test.tools import *
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly import tools

pd.options.mode.chained_assignment = None

xl = pd.read_excel("Base Dados_final.xlsx", "Base de dados")
xl2 = pd.read_excel("Base Dados_final2.xlsx", "Sheet2")

intensity = xl2[[intense for intense in xl2.keys() if "Intensidade" in intense]]
psychosocial = xl2.iloc[:, 21:40]
scores1 = xl2[[scores for scores in xl2.keys() if ("score" in scores or "Score" in scores or "P_" in scores)]]

intense_nrm = normalize_df(intensity)
psychsl = normalize_df(psychosocial)
scores1 = normalize_df(scores1)

sum_pain_7d = pain_no_pain(xl)

SelectorGraph = select_type_column(xl)

app = dash.Dash()
app.css.config.serve_locally = True
app.scripts.config.serve_locally = True

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
				{'label': 'Seniority', 'value': 'seniority'},
				{'label': 'URQ', 'value': 'urq'},
				{'label': 'Zone', 'value': 'zone'},
				{'label': 'Weight', 'value': 'wght'},
				{'label': 'Height', 'value': 'hght'},
				{'label': 'IMC', 'value': 'imc'},
			],
			value = 'gender'
		)
	]),

	html.Div(id="graph_scatter_Corr", children=
		 [
			 dcc.Graph(
				 id='scatterMatrix',
				 style={"width": 1250, "height":1250}
			 ),
			dcc.Dropdown(
				id="dropdown2",
				options=[
					{'label': 'Gender', 'value': 'gender'},
					{'label': 'Age', 'value': 'age'},
					{'label': 'Seniority', 'value': 'seniority'},
					{'label': 'URQ', 'value':'urq'}
				],
				value = 'gender'
			)
	 	]),
		html.Div(id="graph_matrix_corr", children=
         [
	        dcc.Graph(
		        id='selector_graph_corr',
		        style={"width": 1500, "height":1500}
	        ),
	        dcc.Dropdown(
		        id="dropdown3",
		        options=[
					{'label': 'Male', 'value': 'male'},
					{'label': 'Female', 'value': 'female'},
					{'label': 'age over 39', 'value': '59'},
					{'label': 'age 39', 'value': '39'},
					{'label': 'seniority > 11', 'value': '23'},
					{'label': 'seniority < 11', 'value': '11'},
					{'label': 'multisite pain', 'value': 'mpain'},
					{'label': 'pain', 'value': 'pain'},
					{'label': 'no pain', 'value': 'npain'}
					# {'label': 'weight', 'value': 'weight'},
					# {'label': 'height', 'value': 'height'},
					# {'label': 'imc', 'value': 'IMC'},
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
				value = 'pp'
	        ),
		])
])

def createHist(value, key):
	return go.Histogram(
		histfunc="sum",
		y=np.divide(SelectorGraph[value][key]["arr7"]["freq"]["y"], SelectorGraph[value][key]["size"]),
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
	dash.dependencies.Output('scatterMatrix', 'figure'),
    [dash.dependencies.Input('dropdown2', 'value')])
def update_output2(value):

	color_vals, pl_colorscale, dim = Dimension_Scatter_Matrix(value, xl)

	trace1 = go.Splom(dimensions=dim,
					  # default axes name assignment :
					  # xaxes= ['x1','x2',  'x3'],
					  # yaxes=  ['y1', 'y2', 'y3'],
					  marker=dict(color=color_vals,
								  size=7,
								  colorscale=pl_colorscale,
								  showscale=False,
								  line=dict(width=0.5,
											color='rgb(230,230,230)'))
					  )
	layout = go.Layout(
		title='Iris Data set',
		dragmode='select',
		autosize=False,
		hovermode='closest',
		plot_bgcolor='rgba(240,240,240, 0.95)',
		xaxis=dict(
			tickangle=0,
			tickfont=dict(
				family='Arial, sans-serif',
				size=18,
				color='grey'
			)),
		yaxis=dict(
			tickangle=90,
			tickfont=dict(
				family='Arial, sans-serif',
				size=18,
				color='grey'
			),
		)
	)
	axisd = dict(showline=False,
				 zeroline=False,
				 gridcolor='#fff',
				 ticklen=4,
				 titlefont=dict(size=13))
	for i in range(len(dim)):
		layout["xaxis"+str(i+1)] = axisd
		layout["yaxis"+str(i+1)] = axisd

	return {
		'data': [trace1],
		'layout': layout
	}

@app.callback(
	dash.dependencies.Output("selector_graph_corr", 'figure'),
	[dash.dependencies.Input("dropdown3", "value"), dash.dependencies.Input("dropdown4", "value")]
)
def update_corr_graph(value1, value2):
	#pain vs psych
	if (value2 == "pp"):
		xl_temp = pd.concat([intense_nrm, psychsl], axis=1)
	#pain vs scores
	elif(value2 == "ps1"):
		xl_temp = pd.concat([intense_nrm, scores1], axis=1)
	#psych vs scores
	else:
		xl_temp = pd.concat([psychsl, scores1], axis=1)

	if(value1=="male"):
		df = xl_temp.loc[xl["Genero"] == 2]
	elif(value1=="female"):
		df = xl_temp.loc[xl["Genero"] == 1]
	elif(value1=="59"):
		df = xl_temp.loc[xl["Idade"] > 39]
	elif(value1=="39"):
		df = xl_temp.loc[xl["Idade"] > 18 and xl["Idade"] <= 39]
	elif(value1=="23"):
		df = xl_temp.loc[xl["Antoguidade"] >= 11]
	elif(value1=="11"):
		df = xl_temp.loc[xl["Antiguidade"] < 11]
	elif(value1=="mpain"):
		df = xl_temp.loc[sum_pain_7d > 1]
	elif(value1=="pain"):
		df = xl_temp.loc[sum_pain_7d > 0]
	elif(value1=="no_pain"):
		df = xl_temp.loc[sum_pain_7d == 0]

	x = [key for key in df.keys()]
	y = [key for key in df.keys()]

	z = abs(df.corr()).values.tolist()
	z_text = np.around(z, decimals=2)

	fig = ff.create_annotated_heatmap(z=z, x=x, y=y, annotation_text=z_text, reversescale=True, colorscale="Blues", showscale=True,
	                                  xgap=1.5, ygap=1.5)
	# data = [go.Heatmap(z=abs(df.corr()).values.tolist(), colorscale="Blues_r")]
	# data = fig

	return fig

if __name__ == '__main__':
    app.run_server(debug=False)