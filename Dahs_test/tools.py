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