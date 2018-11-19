import numpy as np
import pandas as pd

def make_rgb_transparent(rgb, bg_rgb, alpha):
	return [alpha * c1 + (1 - alpha) * c2
			for (c1, c2) in zip(rgb, bg_rgb)]

def arrayMultiply(array, c):
	return [element*c for element in array]

def arraySum(a, b):
	return map(sum, zip(a,b))

def intermediate(a, b, ratio):
	aComponent = arrayMultiply(a, ratio)
	bComponent = arrayMultiply(b, 1-ratio)
	sumS = [x + y for x, y in zip(aComponent, bComponent)]
	return "rgba("+str(sumS[0]/255)+","+str(sumS[1]/255)+","+ str(sumS[2]/255)+")"

def gradient(a, b, steps):
	steps = [n/float(steps) for n in range(steps)]
	colormap = []
	for step in steps:
		colormap.append(intermediate(a, b, step))
	return colormap

def findcloser(array, value):
	idx = (np.abs(array-value)).argmin()
	return idx

def findcloser_log(array, value):
	idx = (np.abs(array - value)).argmin()
	return idx

def createCorrMatrix(dataFrame, method='pearson'):
	return dataFrame.corr(method=method)

def normalize_df(dataFrame, norm='gauss'):
	if(norm == 'gauss'):
		return (dataFrame-dataFrame.mean())/dataFrame.std()
	else:
		return (dataFrame - dataFrame.mean())/(dataFrame.max - dataFrame.min)

def Dimension_Scatter_Matrix(selector_tag, dataframe):
	if (selector_tag == 'gender'):
		print("Gender!!!")
		class_code = {"male": 0, "female": 1}
		color_vals = [class_code["male"] if cl == 0 else class_code["female"] for cl in dataframe['Genero']]
		pl_colorscale = [[0.0, '#19d3f3'],
						 [0.333, '#19d3f3'],
						 [0.333, '#e763fa'],
						 [0.666, '#e763fa'],
						 [0.666, '#636efa'],
						 [1, '#636efa']]
	# text = [df.loc[k, 'class'] for k in range(len(xl))]
	elif (selector_tag == 'seniority'):
		print("seniority!!!")
		class_code = {"a[0-11]": 0, "a[12-23]": 1}
		color_vals = [class_code["a[0-11]"] if cl <= 11 else class_code["a[12-23]"] for cl in dataframe['Antiguidade']]
		pl_colorscale = [[0.0, '#19d3f3'],
						 [0.333, '#19d3f3'],
						 [0.333, '#e763fa'],
						 [0.666, '#e763fa'],
						 [0.666, '#636efa'],
						 [1, '#636efa']]
	elif (selector_tag == 'age'):
		print("age!!!")
		class_code = {"a[18-38]": 0, "a[39-59]": 1}
		color_vals = [class_code["a[18-38]"] if (cl >= 18 and cl <= 38) else class_code["a[39-59]"] for cl in
					  dataframe['Idade']]
		pl_colorscale = [[0.0, '#19d3f3'],
						 [0.333, '#19d3f3'],
						 [0.333, '#e763fa'],
						 [0.666, '#e763fa'],
						 [0.666, '#636efa'],
						 [1, '#636efa']]
	elif (selector_tag == 'urq'):
		classes = np.unique(dataframe['URQ']).tolist()
		class_code = {classes[k]: k for k in range(len(classes))}
		color_vals = [class_code[cl] for cl in dataframe['URQ']]
		pl_colorscale = [[0.0, '#19d3f3'],
						 [0.333, '#19d3f3'],
						 [0.333, '#e763fa'],
						 [0.666, '#e763fa'],
						 [0.666, '#636efa'],
						 [1, '#636efa']]
	# removed nan dataframe
	df = pd.DataFrame([dataframe[lbl] for lbl in dataframe.keys() if ("Intensidade" in lbl)]).T
	print(df)
	df = df.dropna()
	print(df)

	dim = [dict(label=lbl, values=df[lbl]) for lbl in df.keys()]

	return color_vals, pl_colorscale, dim

