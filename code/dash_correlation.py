import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html


app = dash.Dash(__name__)

app.layout = html.Div(
	[
		html.Div(
			[
				html.H1(
					'Correlation Analysis for Symptoms and Biomechanical Risk Factors',
					className='title'
				),

			]
		)
	]
)