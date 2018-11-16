import pandas as pd
import numpy as np

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
from plotly import tools
from Dahs_test.tools import gradient, findcloser_log

xl = pd.read_excel("Base Dados_final.xlsx")

def readPain(datafrm):
	pain_dict7 = {"total": {"F":0, "I":0},
	              "neck": {"F":0, "I":0},
	              "shoulder": {"total": {"F":0, "I":0}, "left":{"F":0, "I":0}, "right":{"F":0, "I":0}},
	              "elbow": {"total": {"F":0, "I":0}, "left":{"F":0, "I":0}, "right":{"F":0, "I":0}},
	              "hand": {"total": {"F":0, "I":0}, "left":{"F":0, "I":0}, "right":{"F":0, "I":0}},
	              "trunk": {"F":0, "I":0},
	              "lombar": {"F":0, "I":0}}
	pain_dict12 = {"total": {"F": 0, "I": 0},
	               "neck": {"F": 0, "I": 0},
	               "shoulder": {"total": {"F": 0, "I": 0}, "left": {"F": 0, "I": 0}, "right": {"F": 0, "I": 0}},
	               "elbow": {"total": {"F": 0, "I": 0}, "left": {"F": 0, "I": 0}, "right": {"F": 0, "I": 0}},
	               "hand": {"total": {"F": 0, "I": 0}, "left": {"F": 0, "I": 0}, "right": {"F": 0, "I": 0}},
	               "trunk": {"F": 0, "I": 0},
	               "lombar":{"F":0, "I":0}}

	pain_array7 = {"freq":{"x":[], 'y':[]}, "int":{"x":[], 'y':[]}}
	pain_array12 = {"x":[], 'y':[]}

	#intensity
	blue_1 = (218, 232, 245)
	blue_10 = (11, 85, 159)
	green = (60, 179, 113)
	red = (178, 34, 34)

	intensity_color = np.linspace(0, 10, 100000)
	color = gradient(red, blue_10, 100000)

	m12 = [key for key in datafrm.keys() if ('12m' in key and 'Impedimento' not in key)]
	m7 = [key for key in datafrm.keys() if ('7dias' in key and 'Impedimento' not in key and "exausto" not in key)]

	for index, (key12, key7) in enumerate(zip(m12, m7)):
		if("Pescoço" in key12):
			pain_dict7["neck"]["F"] = datafrm[key7].sum()
			intensity = [x for x in datafrm["Pescoço_Intensidade"] if x > 0]
			pain_dict7["neck"]["I"] = color[findcloser_log(intensity_color, np.mean(intensity))]
			pain_dict12["neck"]["F"] = datafrm[key12].sum()

			pain_array7["freq"]["x"].append("neck")
			pain_array7["int"]["x"].append("neck")
			pain_array7["freq"]["y"].append(pain_dict7["neck"]["F"])
			pain_array7["int"]["y"].append(pain_dict7["neck"]["I"])

			pain_array12["y"].append(pain_dict12["neck"]["F"])
			pain_array12["x"].append("neck")

		elif ("Ombro" in key12):
			pain_dict7["shoulder"]["left"]["F"] = np.sum([1 for x in datafrm[key7] if(x == 2 or x == 3)])
			pain_dict7["shoulder"]["left"]["I"] = datafrm["Ombro_Intensidade_esquerdo"].mean()
			pain_dict7["shoulder"]["right"]["F"] = np.sum([1 for x in datafrm[key7] if(x == 1 or x == 3)])
			pain_dict7["shoulder"]["right"]["I"] = datafrm["Ombro_Intensidade_direito"].mean()
			pain_dict7["shoulder"]["total"]["F"] =pain_dict12["shoulder"]["total"]["F"] = np.sum([1 if(x > 0 and x < 3) else 2 if(x==3) else 0 for x in datafrm[key7]])
			intensity_right = [x for x in datafrm["Ombro_Intensidade_direito"] if x>0]
			intensity_left = [x for x in datafrm["Ombro_Intensidade_esquerdo"] if x>0]
			intensity = (np.sum(intensity_right) + np.sum(intensity_left))/(len(intensity_left)+len(intensity_right))
			pain_dict7["shoulder"]["total"]["I"] = color[findcloser_log(intensity_color, np.mean(intensity))]

			pain_dict12["shoulder"]["left"]["F"] = np.sum([1 for x in datafrm[key12] if(x == 2 or x == 3)])
			pain_dict12["shoulder"]["right"]["F"] = np.sum([1 for x in datafrm[key12] if(x == 1 or x == 3)])
			pain_dict12["shoulder"]["total"]["F"] = np.sum([1 if(x > 0 and x < 3) else 2 if(x==3) else 0 for x in datafrm[key12]])

			pain_array7["freq"]["x"].append("shoulder")
			pain_array7["int"]["x"].append("shoulder")
			pain_array7["freq"]["y"].append(pain_dict7["shoulder"]["total"]["F"])
			pain_array7["int"]["y"].append(pain_dict7["shoulder"]["total"]["I"])

			pain_array12["y"].append(pain_dict12["shoulder"]["total"]["F"])
			pain_array12["x"].append("shoulder")

		elif("Cotovelo" in key12):
			pain_dict7["elbow"]["left"]["F"] = np.sum([1 for x in datafrm[key7] if (x == 2 or x == 3)])
			pain_dict7["elbow"]["left"]["I"] = datafrm["Cotovelo_Intensidade_esquerdo"].mean()
			pain_dict7["elbow"]["right"]["F"] = np.sum([1 for x in datafrm[key7] if (x == 1 or x == 3)])
			pain_dict7["elbow"]["right"]["I"] = datafrm["Cotovelo_Intensidade_direito"].mean()
			pain_dict7["elbow"]["total"]["F"] = pain_dict12["elbow"]["total"]["F"] = np.sum(
				[1 if (x > 0 and x < 3) else 2 if (x == 3) else 0 for x in datafrm[key7]])
			intensity_right = [x for x in datafrm["Cotovelo_Intensidade_direito"] if x > 0]
			intensity_left = [x for x in datafrm["Cotovelo_Intensidade_esquerdo"] if x > 0]
			intensity = (np.sum(intensity_right) + np.sum(intensity_left)) / (len(intensity_left) + len(intensity_right))
			pain_dict7["elbow"]["total"]["I"] = color[findcloser_log(intensity_color, np.mean(intensity))]

			pain_dict12["elbow"]["left"]["F"] = np.sum([1 for x in datafrm[key12] if (x == 2 or x == 3)])
			pain_dict12["elbow"]["right"]["F"] = np.sum([1 for x in datafrm[key12] if (x == 1 or x == 3)])
			pain_dict12["elbow"]["total"]["F"] = np.sum(
				[1 if (x > 0 and x < 3) else 2 if (x == 3) else 0 for x in datafrm[key12]])

			pain_array7["freq"]["x"].append("elbow")
			pain_array7["int"]["x"].append("elbow")
			pain_array7["freq"]["y"].append(pain_dict7["elbow"]["total"]["F"])
			pain_array7["int"]["y"].append(pain_dict7["elbow"]["total"]["I"])

			pain_array12["y"].append(pain_dict12["elbow"]["total"]["F"])
			pain_array12["x"].append("elbow")

		elif("PunhoMao" in key12):
			pain_dict7["hand"]["left"]["F"] = np.sum([1 for x in datafrm[key7] if (x == 2 or x == 3)])
			pain_dict7["hand"]["left"]["I"] = datafrm["PunhoMao_Intensidade_esquerda"].mean()
			pain_dict7["hand"]["right"]["F"] = np.sum([1 for x in datafrm[key7] if (x == 1 or x == 3)])
			pain_dict7["hand"]["right"]["I"] = datafrm["PunhoMao_Intensidade_direita"].mean()
			pain_dict7["hand"]["total"]["F"] = pain_dict12["elbow"]["total"]["F"] = np.sum(
				[1 if (x > 0 and x < 3) else 2 if (x == 3) else 0 for x in datafrm[key7]])
			intensity_right = [x for x in datafrm["PunhoMao_Intensidade_direita"] if x > 0]
			intensity_left = [x for x in datafrm["PunhoMao_Intensidade_esquerda"] if x > 0]
			intensity = (np.sum(intensity_right) + np.sum(intensity_left)) / (len(intensity_left) + len(intensity_right))
			pain_dict7["hand"]["total"]["I"] = color[findcloser_log(intensity_color, np.mean(intensity))]

			pain_dict12["hand"]["left"]["F"] = np.sum([1 for x in datafrm[key12] if (x == 2 or x == 3)])
			pain_dict12["hand"]["right"]["F"] = np.sum([1 for x in datafrm[key12] if (x == 1 or x == 3)])
			pain_dict12["hand"]["total"]["F"] = np.sum(
				[1 if (x > 0 and x < 3) else 2 if (x == 3) else 0 for x in datafrm[key12]])

			pain_array7["freq"]["x"].append("hand")
			pain_array7["int"]["x"].append("hand")
			pain_array7["freq"]["y"].append(pain_dict7["hand"]["total"]["F"])
			pain_array7["int"]["y"].append(pain_dict7["hand"]["total"]["I"])

			pain_array12["y"].append(pain_dict12["hand"]["total"]["F"])
			pain_array12["x"].append("hand")

		elif("Toracica" in key12):
			pain_dict7["trunk"]["F"] = datafrm[key7].sum()
			intensity = [x for x in datafrm["Toracica_Intensidade"] if x > 0]
			pain_dict7["trunk"]["I"] = color[findcloser_log(intensity_color, np.mean(intensity))]
			pain_dict12["trunk"]["F"] = datafrm[key12].sum()

			pain_array7["freq"]["x"].append("trunk")
			pain_array7["int"]["x"].append("trunk")
			pain_array7["freq"]["y"].append(pain_dict7["trunk"]["F"])
			pain_array7["int"]["y"].append(pain_dict7["trunk"]["I"])

			pain_array12["y"].append(pain_dict12["trunk"]["F"])
			pain_array12["x"].append("trunk")

		elif ("Lombar" in key12):
			pain_dict7["lombar"]["F"] = datafrm[key7].sum()
			intensity = [x for x in datafrm["Lombar_Intensidade"] if x > 0]
			pain_dict7["lombar"]["I"] = color[findcloser_log(intensity_color, np.mean(intensity))]
			pain_dict12["lombar"]["F"] = datafrm[key12].sum()

			pain_array7["freq"]["x"].append("lombar")
			pain_array7["int"]["x"].append("lombar")
			pain_array7["freq"]["y"].append(pain_dict7["lombar"]["F"])
			pain_array7["int"]["y"].append(pain_dict7["lombar"]["I"])

			pain_array12["y"].append(pain_dict12["lombar"]["F"])
			pain_array12["x"].append("lombar")


	# #impedimento
	# m12 = [key for key in datafrm.keys() if ('12m' in key and 'Impedimento' not in key)]
	# m7 = [key for key in datafrm.keys() if ('7dias' in key and 'Impedimento' not in key)]
	pain_dict = {"pain7": pain_dict7, "pain12": pain_dict12}

	return pain_dict, pain_array7, pain_array12

def select_type_column(datafrm):

	selector_Graph1 = {"gender": {"male": 0, "female": 0}, "seniority": {"a[0-11]": 0, "a[12-23]": 0}, "age": {"a[18-38]": 0, "a[39-59]": 0}, "score": 0}

	for key in selector_Graph1.keys():
		if(key == "gender"):
			#Female
			df_female = datafrm.loc[datafrm['Genero'] == 1]
			female_pain_dict, female_arr7, female_arr12 = readPain(df_female)
			#Male
			df_male = datafrm.loc[datafrm['Genero'] == 2]
			male_pain_dict, male_arr7, male_arr12= readPain(df_male)

			selector_Graph1["gender"]["male"] = {"total":male_pain_dict, "arr7": male_arr7, "arr12":male_arr12, "size":len(df_male)}
			selector_Graph1["gender"]["female"] = {"total": female_pain_dict, "arr7": female_arr7, "arr12": female_arr12, "size":len(df_female)}

		elif(key == "seniority"):
			data = datafrm["Antiguidade"]
			# [0-11]
			df_0_11 = datafrm.loc[np.logical_and(datafrm['Antiguidade'] > 0, datafrm['Antiguidade'] < 12)]
			d011_pain_dict, d011_arr7, d011_arr12 = readPain(df_0_11)
			# [12-23]
			df_12_23 = datafrm.loc[datafrm['Antiguidade'] > 11]
			d1223_pain_dict, d1223_arr7, d1223_arr12 = readPain(df_12_23)

			selector_Graph1["seniority"]["a[0-11]"] = {"total": d011_pain_dict, "arr7": d011_arr7, "arr12": d011_arr12, "size":len(df_0_11)}
			selector_Graph1["seniority"]["a[12-23]"] = {"total": d1223_pain_dict, "arr7": d1223_arr7,
			                                       "arr12": d1223_arr12, "size":len(df_12_23)}

		elif (key == "age"):

			data = datafrm["Idade"]
			# [18-38]
			df_18_38 = datafrm.loc[np.logical_and(datafrm['Idade'] > 18, datafrm['Idade'] < 38)]
			d1838_pain_dict, d1838_arr7, d1838_arr12 = readPain(df_18_38)
			# [39-58]
			df_39_59 = datafrm.loc[datafrm['Idade'] > 38]
			d3959_pain_dict, d3959_arr7, d3959_arr12 = readPain(df_39_59)

			selector_Graph1["age"]["a[18-38]"] = {"total": d1838_pain_dict, "arr7": d1838_arr7,
			                                           "arr12": d1838_arr12, "size":len(df_18_38)}
			selector_Graph1["age"]["a[39-59]"] = {"total": d3959_pain_dict, "arr7": d3959_arr7,
			                                           "arr12": d3959_arr12, "size":len(df_39_59)}

		elif (key == "score"):
			data = datafrm["Score"]


	return selector_Graph1


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
				{'label': 'Seniority', 'value': 'seniority'}
			],
			value = 'gender'
		)
	]),

	html.Div(id="graphMatrix_Corr", children=
		 [
			 dcc.Graph(
				 id='regressionMatrix'
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
	dash.dependencies.Output('regressionMatrix', 'figure'),
    [dash.dependencies.Input('dropdown2', 'value')])
def update_output2(value):
	if(value == 'gender'):
		print("Gender!!!")
		class_code = {"male":0, "female":1}
		color_vals = [class_code["male"] if cl==0 else class_code["female"] for cl in xl['Genero']]
		pl_colorscale = [[0.0, '#19d3f3'],
						 [0.333, '#19d3f3'],
						 [0.333, '#e763fa'],
						 [0.666, '#e763fa'],
						 [0.666, '#636efa'],
						 [1, '#636efa']]
		# text = [df.loc[k, 'class'] for k in range(len(xl))]
	elif(value == 'seniority'):
		print("seniority!!!")
		class_code = {"a[0-11]": 0, "a[12-23]": 1}
		color_vals = [class_code["a[0-11]"] if cl <= 11 else class_code["a[12-23]"] for cl in xl['Antiguidade']]
		pl_colorscale = [[0.0, '#19d3f3'],
						 [0.333, '#19d3f3'],
						 [0.333, '#e763fa'],
						 [0.666, '#e763fa'],
						 [0.666, '#636efa'],
						 [1, '#636efa']]
	elif(value == 'age'):
		print("age!!!")
		class_code = {"a[18-38]": 0, "a[39-59]": 1}
		color_vals = [class_code["a[18-38]"] if (cl >= 18 and cl <=38) else class_code["a[39-59]"] for cl in xl['Idade']]
		pl_colorscale = [[0.0, '#19d3f3'],
						 [0.333, '#19d3f3'],
						 [0.333, '#e763fa'],
						 [0.666, '#e763fa'],
						 [0.666, '#636efa'],
						 [1, '#636efa']]
	elif(value == 'urq'):
		classes = np.unique(xl['URQ']).tolist()
		class_code = {classes[k]: k for k in range(len(classes))}
		color_vals = [class_code[cl] for cl in xl['URQ']]
		pl_colorscale = [[0.0, '#19d3f3'],
						 [0.333, '#19d3f3'],
						 [0.333, '#e763fa'],
						 [0.666, '#e763fa'],
						 [0.666, '#636efa'],
						 [1, '#636efa']]
	#removed nan dataframe
	df = pd.DataFrame([xl[lbl] for lbl in xl.keys() if("Intensidade" in lbl)]).T
	print(df)
	df = df.dropna()
	print(df)

	dim = [dict(label=lbl, values=df[lbl]) for lbl in df.keys()]
	# dim = [dict(label='ombro', values =xl[])]
	axis = dict(showline=True,
				zeroline=False,
				gridcolor='#fff',
				ticklen=4)

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
		width=600,
		height=600,
		autosize=False,
		hovermode='closest',
		plot_bgcolor='rgba(240,240,240, 0.95)'
	)

	return {
		'data': [trace1],
		'layout': layout
	}


if __name__ == '__main__':
    app.run_server(debug=False)