from Dahs_test.data_loader import *
from Dahs_test.tools import *
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import plotly.figure_factory as ff
from openpyxl import load_workbook
from plotly import tools
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None

print("loading excel file...")
xl = pd.read_excel("Base Dados_final.xlsx", "Base de dados")
print("loaded main database...")
xl2 = pd.read_excel("Base Dados_final2.xlsx", "Sheet2")
print("loaded main data...")
stations_risk = pd.read_excel(
    r"C:\Users\rjoao\Documents\PhD\Ana_Data_Correlation\CorrelationProject\Dahs_test\APErgo_Fatores de risco_1basedados.xlsx")
# Risk factors
elbowRisk = stations_risk[["P_60_range", "P_80_range", "P_100_range"]].sum(axis=1)
shoulderNeck_risk = stations_risk[["P_Stand_shoulderheight", "P_Stand_abovehead"]].sum(axis=1)
trunkRisk = stations_risk[["P_Stand_bent", "P_Stand_Strongly_bent"]].sum(axis=1)
scoreRisk = stations_risk[[key for key in stations_risk.keys() if "Score" in key or "score" in key]]

# personalfactors


print("loaded risk factors...")

print("extracting main data from database...")
stations_per_wkr = load_stations(xl)
print("got stations that each worker performed...")
wkr_risks = load_risk_coefs_per_wkr(stations_risk, stations_per_wkr)
print("loaded mean, max and min risk factors for each worker")
normalize = False
if (normalize == True):
    wkr_mean = wkr_risks[0].apply(lambda x: normalize_df2(x))
    wkr_max = wkr_risks[1].apply(lambda x: normalize_df2(x))
    wkr_min = wkr_risks[2].apply(lambda x: normalize_df2(x))
    wkr_sum = wkr_risks[3].apply(lambda x: normalize_df2(x))
else:
    wkr_mean = wkr_risks[0]
    wkr_max = wkr_risks[1]
    wkr_min = wkr_risks[2]
    wkr_sum = wkr_risks[3]

print("normalizing...")

sum_pain_7d = pain_no_pain(xl)

msiteZona = xl["Zona"].loc[sum_pain_7d >= 1]
msite_index = list(np.where(sum_pain_7d >= 1)[0])

# check stations that people with reported pain have done
msite_stations = np.unique([item for index in msite_index for item in stations_per_wkr[index]])

intensity = xl2[[intense for intense in xl2.keys() if "Intensidade" in intense]]
psychosocial = xl2.iloc[:, 21:40]
scores_mean = wkr_mean[
    [scores for scores in wkr_mean.keys() if ("score" in scores or "Score" in scores or "P_" in scores)]]
scores_max = wkr_max[
    [scores for scores in wkr_max.keys() if ("score" in scores or "Score" in scores or "P_" in scores)]]
scores_min = wkr_min[
    [scores for scores in wkr_min.keys() if ("score" in scores or "Score" in scores or "P_" in scores)]]
scores_sum = wkr_sum[
    [scores for scores in wkr_sum.keys() if ("score" in scores or "Score" in scores or "P_" in scores)]]

intense_nrm = normalize_df(intensity)
psychsl = normalize_df(psychosocial)

print("starting dash app...")

SelectorGraph = select_type_column(xl)

def run_corr_excel():
    value2 = "ps1"
    values1 = ["all", "male", "female", "59", "39", "23", "11", "imc1", "imc2", "mpain", "pain"]
    writer = pd.ExcelWriter('Pain_Score.xlsx', engine="xlsxwriter")
    for value1 in values1:
        # pain vs psych
        if (value2 == "pp"):
            xl_temp = pd.concat([intense_nrm, psychsl], axis=1)
            wdt = 1250
            hgt = 1000
            tag = "Pain_Psych"
        # pain vs scores
        elif (value2 == "ps1"):
            xl_temp = pd.concat([intense_nrm, scores_mean], axis=1)
            wdt = 1250
            hgt = 1000
            tag = "Pain_Score"
        # psych vs scores
        else:
            xl_temp = pd.concat([psychsl, scores_mean], axis=1)
            wdt = 1500
            hgt = 1500
            tag = "Psych_Score"

        if (value1 == "all"):
            df = xl_temp
            sheet = "All"
        elif (value1 == "male"):
            df = xl_temp.loc[xl["Genero"] == 2]
            sheet = "Men"
        elif (value1 == "female"):
            df = xl_temp.loc[xl["Genero"] == 1]
            sheet = "Women"
        elif (value1 == "59"):
            df = xl_temp.loc[xl["Idade"] > 39]
            sheet = "Age>39"
        elif (value1 == "39"):
            df = xl_temp.loc[xl["Idade"] <= 39]
            sheet = "Age<39"
        elif (value1 == "23"):
            df = xl_temp.loc[xl["Antiguidade"] >= 11]
            sheet = "Seniority>11"
        elif (value1 == "11"):
            df = xl_temp.loc[xl["Antiguidade"] < 11]
            sheet = "Seniority<11"
        elif (value1 == "imc1"):
            df = xl_temp.loc[xl["IMC"] < 24.9]
            sheet = "IMC<24.9"
        elif (value1 == "imc2"):
            df = xl_temp.loc[xl["IMC"] >= 24.9]
            sheet = "IMC>24.9"
        elif (value1 == "mpain"):
            df = xl_temp.loc[sum_pain_7d > 1]
            sheet = "MultiSitePain"
        elif (value1 == "pain"):
            df = xl_temp.loc[sum_pain_7d == 1]
            sheet = "OneSitePain"
        elif (value1 == "npain"):
            df = xl_temp.loc[sum_pain_7d == 0]
            sheet = "NoPain"

        x = [key for key in df.keys()]
        y = [key for key in df.keys()]

        z_corr = abs(df.corr())

        # save_Excel
        var1 = []
        var2 = []
        val = []
        keys = z_corr.keys()
        for key in keys:
            corr_array = z_corr[key]
            for index, corr in enumerate(corr_array):
                if (corr > 0.3):
                    var1.append(key)
                    var2.append(keys[index])
                    val.append(corr)
        df = pd.DataFrame({"var1": var1, 'var2': var2, 'val': val})
        df.to_excel(writer, sheet_name=sheet)
    writer.save()
    writer.close()

# run_corr_excel()

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
                {'label': 'Multisite-pain', 'value': 'm_pain'},
                {'label': 'Onesite-pain', 'value': 'o_pain'},
            ],
            value='gender'
        )
    ]),
    html.Div(id="Zone_header", children=
    [
        html.H3(children="Zone Analysis")
    ]),
    html.Div(id="radar_div", children=
    [
        dcc.Graph(
            id='radar_graph',
            style={"width": 400, "padding-top": -200, "display": "inline-block"}),
        html.Div(
            id="hist_containers",
            style={"width": 800, "display": "inline-block"},
            children=[
                dcc.Graph(
                    id='hist_personal',
                    style={"width": 800, "height": 300, "display": "inline-block"}),
                dcc.Graph(
                    id='hist_pain',
                    style={"width": 800, "height": 300, "display": "inline-block"})
            ]
        ),
        dcc.Dropdown(
            id="dropdown_zone",
            options=
            [{"label": zone, "value": zone} for zone in get_zone_list(xl)])
    ]),

    # html.Div(id="graph_scatter_Corr", children=
    # 	 [
    # 		 dcc.Graph(
    # 			 id='scatterMatrix',
    # 			 style={"width": 1250, "height":1250, "padding-top":100}
    # 		 ),
    # 		dcc.Dropdown(
    # 			id="dropdown2",
    # 			options=[
    # 				{'label': 'Gender', 'value': 'gender'},
    # 				{'label': 'Age', 'value': 'age'},
    # 				{'label': 'Seniority', 'value': 'seniority'},
    # 				{'label': 'URQ', 'value':'urq'}
    # 			],
    # 			value = 'gender'
    # 		)
    #  	]),
    html.Div(id="graph_matrix_corr", children=
    [
        dcc.Graph(
            id='selector_graph_corr',
            style={"width": 1500, "height": 1500, "padding-top": 100, "padding-bottom": 50, "padding-left": 75}
        ),
        dcc.Dropdown(
            id="dropdown3",
            options=[
                {'label': 'All', 'value': 'all'},
                {'label': 'Male', 'value': 'male'},
                {'label': 'Female', 'value': 'female'},
                {'label': 'age over 39', 'value': '59'},
                {'label': 'age 39', 'value': '39'},
                {'label': 'seniority > 11', 'value': '23'},
                {'label': 'seniority < 11', 'value': '11'},
                {'label': 'IMC < 24.9', 'value': 'imc1'},
                {'label': 'IMC > 24.9', 'value': 'imc2'},
                {'label': 'multisite pain', 'value': 'mpain'},
                {'label': 'pain', 'value': 'pain'},
                {'label': 'no pain', 'value': 'npain'}
                # {'label': 'weight', 'value': 'weight'},
                # {'label': 'height', 'value': 'height'},
                # {'label': 'imc', 'value': 'IMC'},
            ],
            value='male'
        ),
        dcc.Dropdown(
            id="dropdown4",
            options=[
                {'label': 'Pain VS Psych', 'value': 'pp'},
                {'label': 'Pain VS Scores', 'value': 'ps1'},
                {'label': 'Psych VS Scores', 'value': 'ps2'},
            ],
            value='pp'
        ),
    ])
])


def createHist(x, y, name):
    return go.Histogram(
        histfunc="sum",
        y=y,
        x=x,
        name=name,
    )


@app.callback(
    dash.dependencies.Output('selector_graph', 'figure'),
    [dash.dependencies.Input('dropdown1', 'value')])
def update_output(value):
    graphs = [createHist(SelectorGraph[value][key]["arr7"]["freq"]["x"],
                         np.divide(SelectorGraph[value][key]["arr7"]["freq"]["y"], SelectorGraph[value][key]["size"]),
                         key + " - " + str(SelectorGraph[value][key]["size"])) for key in SelectorGraph[value].keys()]
    return {
        'data': graphs
    }


"""
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
	}"""




def createRadar(r, theta, zone):
    data = [
        go.Scatterpolar(
            r=r,
            theta=theta,
            fill='toself',
            name=zone
        )
    ]
    return data


@app.callback(
    dash.dependencies.Output("radar_graph", "figure"),
    [dash.dependencies.Input("dropdown_zone", "value")]
)
def update_radarplot(value):
    # value will be the zone
    r = np.array(SelectorGraph["zone"][value]["arr7"]["freq"]["y"])
    theta = np.array(SelectorGraph["zone"][value]["arr7"]["freq"]["x"])
    nr = np.delete(r, np.where(theta == "total")[0])
    ntheta = np.delete(theta, np.where(theta == "total")[0])
    graph = createRadar(nr, ntheta, value)
    return {
        'data': graph
    }


def createNormalHist(x):
    return go.Histogram(
        x=x,
        opacity=0.75,
        nbinsx=50)


@app.callback(
    dash.dependencies.Output("hist_pain", "figure"),
    [dash.dependencies.Input("dropdown_zone", "value")]
)
def update_radarHist(value):
    # elbow_hist = createNormalHist(elbowRisk.loc[stations_risk["zona"]==value])
    # shoulderNeck_hist = createNormalHist(shoulderNeck_risk.loc[stations_risk["zona"]==value])
    # trunk_hist = createNormalHist(trunkRisk.loc[stations_risk["zona"]==value])
    # zone = msiteZona.loc[msiteZona==value]

    stations_ofzone = [station for station in msite_stations if station[0] == value]

    where_stations = [index for index, zone in enumerate(stations_risk["URQ"] + stations_risk["Estacao"].map(str)) if
                      (zone in stations_ofzone)]

    x = ["P_range", "P_abovesh&head", "P_stand_bent"]
    y = [elbowRisk[where_stations].mean(),
         shoulderNeck_risk[where_stations].mean(),
         trunkRisk[where_stations].mean()]
    y2 = [elbowRisk.mean(), shoulderNeck_risk.mean(), trunkRisk.mean()]
    score_factors = list(scoreRisk.loc[where_stations].mean().values)
    score_factors2 = list(scoreRisk.mean().values)
    x_factors = list(scoreRisk.loc[where_stations].mean().keys())
    y = y + score_factors
    y2 = y2 + score_factors2

    x = x + x_factors

    # y = [elbowRisk.loc[stations_risk["zona"]==value].mean(),
    #      shoulderNeck_risk.loc[stations_risk["zona"] == value].mean(),
    #      trunkRisk.loc[stations_risk["zona"] == value]]

    data = createHist(x, y, "Pain population")
    data2 = createHist(x, y2, "Normal population")
    # data = [elbow_hist, shoulderNeck_hist, trunk_hist]

    return {
        "data": [data, data2],
        "layout": go.Layout(barmode="stack")
    }


@app.callback(
    dash.dependencies.Output("hist_personal", "figure"),
    [dash.dependencies.Input("dropdown_zone", "value")]
)
def update_radarHist(value):
    # elbow_hist = createNormalHist(elbowRisk.loc[stations_risk["zona"]==value])
    # shoulderNeck_hist = createNormalHist(shoulderNeck_risk.loc[stations_risk["zona"]==value])
    # trunk_hist = createNormalHist(trunkRisk.loc[stations_risk["zona"]==value])
    # zone = msiteZona.loc[msiteZona==value]

    stations_ofzone = [station for station in msite_stations if station[0] == value]

    where_stations = [index for index, zone in enumerate(stations_risk["URQ"] + stations_risk["Estacao"].map(str)) if
                      (zone in stations_ofzone)]

    x = ["IMC", "Idade", "Antiguidade", "Altura", "Genero"]

    y = [xl["IMC"][where_stations].mean(),
         xl["Idade"][where_stations].mean(),
         xl["Antiguidade"][where_stations].mean(),
         xl["Altura"][where_stations].mean(),
         xl["Genero"][where_stations].mean()]
    y2 = [xl["IMC"].mean(),
          xl["Idade"].mean(),
          xl["Antiguidade"].mean(),
          xl["Altura"].mean(),
          xl["Genero"].mean()]

    x_factors = list(scoreRisk.loc[where_stations].mean().keys())

    data = createHist(x, y, "Pain population")
    data2 = createHist(x, y2, "Normal population")

    return {
        "data": [data, data2],
        "layout": go.Layout(barmode="stack")
    }


@app.callback(
    dash.dependencies.Output("selector_graph_corr", 'figure'),
    [dash.dependencies.Input("dropdown3", "value"), dash.dependencies.Input("dropdown4", "value")]
)
def update_corr_graph(value1, value2):
    # pain vs psych
    if (value2 == "pp"):
        xl_temp = pd.concat([intense_nrm, psychsl], axis=1)
        wdt = 1250
        hgt = 1000
        tag = "Pain_Psych"
    # pain vs scores
    elif (value2 == "ps1"):
        xl_temp = pd.concat([intense_nrm, scores_mean], axis=1)
        wdt = 1250
        hgt = 1000
        tag = "Pain_Score"
    # psych vs scores
    else:
        xl_temp = pd.concat([psychsl, scores_mean], axis=1)
        wdt = 1500
        hgt = 1500
        tag = "Psych_Score"

    if (value1 == "all"):
        df = xl_temp
        sheet = "All"
    elif (value1 == "male"):
        df = xl_temp.loc[xl["Genero"] == 2]
        sheet = "Men"
    elif (value1 == "female"):
        df = xl_temp.loc[xl["Genero"] == 1]
        sheet = "Women"
    elif (value1 == "59"):
        df = xl_temp.loc[xl["Idade"] > 39]
        sheet = "Age>39"
    elif (value1 == "39"):
        df = xl_temp.loc[xl["Idade"] > 18] and xl_temp.loc[xl["Idade"] <= 39]
        sheet = "Age<39"
    elif (value1 == "23"):
        df = xl_temp.loc[xl["Antiguidade"] >= 11]
        sheet = "Seniority>11"
    elif (value1 == "11"):
        df = xl_temp.loc[xl["Antiguidade"] < 11]
        sheet = "Seniority<11"
    elif (value1 == "imc1"):
        df = xl_temp.loc[xl["IMC"] < 24.9]
        sheet = "IMC<24.9"
    elif (value1 == "imc2"):
        df = xl_temp.loc[xl["IMC"] >= 24.9]
        sheet = "IMC>24.9"
    elif (value1 == "mpain"):
        df = xl_temp.loc[sum_pain_7d > 1]
        sheet = "MultiSitePain"
    elif (value1 == "pain"):
        df = xl_temp.loc[sum_pain_7d == 1]
        sheet = "OneSitePain"
    elif (value1 == "npain"):
        df = xl_temp.loc[sum_pain_7d == 0]
        sheet = "NoPain"

    x = [key for key in df.keys()]
    y = [key for key in df.keys()]

    z_corr = abs(df.corr())
    z_corr[z_corr < 0.3] = 0

    z = z_corr.values.tolist()
    z_text = np.around(z, decimals=2)

    fig = ff.create_annotated_heatmap(z=z, x=x, y=y, annotation_text=z_text, reversescale=True, colorscale="Blues",
                                      showscale=True,
                                      xgap=1.5, ygap=1.5)

    fig.layout["title"] = "Correlation Matrix - " + str(len(df))
    fig['layout']['xaxis'].update(side='bottom')
    fig['layout']["width"] = wdt
    fig['layout']["height"] = hgt

    return fig


@app.callback(
    dash.dependencies.Output("selector_graph_corr", 'style'),
    [dash.dependencies.Input("dropdown4", "value")]
)
def update_style_corr(value):
    if (value == "pp"):
        wdt = 1250
        hgt = 1000
    # pain vs scores
    elif (value == "ps1"):
        wdt = 1250
        hgt = 1000
    # psych vs scores
    else:
        wdt = 1500
        hgt = 1500

    return {'width': wdt, 'height': hgt, 'padding-top': 100, 'padding-bottom': 50, 'padding-left': 75}





if __name__ == '__main__':
    app.run_server(debug=False)

