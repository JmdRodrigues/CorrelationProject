import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import pandas
import itertools
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.figure import SubplotParams


def plotClusters(y_pred, XPCA, n_clusters, data, colors2):
	"""
	:param y_pred: Matrix with Classification values
	:param OSignal: Original Signal
	:param time: Time array
	:param XPCA: Matrix after PCA. XPCA[0] - feature 1 and XPCA[1] - feature 2
	:param n_clusters: Number of clusters
	:return: plot object of clusters
	"""
	#Specify plot parameters
	# color
	face_color_r = 248 / 255.0
	face_color_g = 247 / 255.0
	face_color_b = 249 / 255.0

	# pars
	left = 0.05  # the left side of the subplots of the figure
	right = 0.95  # the right side of the subplots of the figure
	bottom = 0.05  # the bottom of the subplots of the figure
	top = 0.92  # the top of the subplots of the figure
	wspace = 0.5  # the amount of width reserved for blank space between subplots
	hspace = 0.4  # the amount of height reserved for white space between subplots

	pars = SubplotParams(left, bottom, right, top, wspace, hspace)

	#specify Font properties with fontmanager---------------------------------------------------
	font0 = FontProperties()
	font0.set_weight('medium')
	font0.set_family('monospace')
	#Specify Font properties of Legends
	font1 = FontProperties()
	font1.set_weight('normal')
	font1.set_family('sans-serif')
	font1.set_style('italic')
	font1.set_size(17)
	#Specify font properties of Titles
	font2 = FontProperties()
	font2.set_size(20)
	font2.set_family('sans-serif')
	font2.set_weight('medium')
	font2.set_style('italic')

	#Cluster colors---------------------------------------------------------------------------------------------
	# scatColors = np.array([x for x in ['darkseagreen', 'indianred', 'cornflowerblue', 'darkorange', 'indigo']])
	#scatColors = np.array([x for x in ['#93d1ff', '#ffc425', '#fc3366', '#032569']]) #pallete 1
	#scatColors = np.array([x for x in ['#ce3635', '#2caae2', '#2ce2aa', '#c38ce3']]) #pallete Gamut
	scatColors = np.array([x for x in ['#3366CC', '#79BEDB', '#E84150', '#FFB36D', '#6aba8f', '#78136f', '#236B5D',
	                                   '#AB5436','#3366CC', '#AB5436']])
	markers = np.array([x for x in {'o', 'v', 's', '*', '8', 'D', 'd', '+', 'o', 'v'}])
	Colors = itertools.cycle(scatColors)
	Markers = itertools.cycle(markers)

	#Create Grid Frame for Clustering Representation----------------------------------------
	PCAFrame = pandas.DataFrame(data=XPCA, columns=['Var1', 'Var2'])
	g = sns.JointGrid('Var1', 'Var2', PCAFrame)
	f2 = g.fig
	f2.set_dpi(96)
	f2.set_figheight(1080 / 96)
	f2.set_figwidth(1920 / 96)

	#Create figure for signal representation with clusters-----------------------------------
	f, axes = plt.subplots(n_clusters+1, 1)
	f.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
	f.set_dpi(96)
	f.set_figheight(1080/96)
	f.set_figwidth(1920/96)
	f.set_facecolor((face_color_r, face_color_g, face_color_b))

	lines = []
	labels = []

	#Cycle for figure plotting------------------------------------------------------------------------------
	for s in range(0, n_clusters):
		x = XPCA[np.where(y_pred== s)[0], 0]
		y = XPCA[np.where(y_pred== s)[0], 1]
		color = next(Colors)
		marker = next(Markers)
		line = mlines.Line2D([], [], color=color, marker=marker, markersize=15, markeredgecolor='gray',
		                     markerfacecolor=color, markeredgewidth=2, label='Cluster' + str(s + 1))
		lines.append(line)
		cmap = sns.light_palette(color, as_cmap=True, n_colors=120, reverse=True)
		ax = sns.kdeplot(x, y, cmap=cmap, shade=True, shade_lowest=False, alpha=1, ax=g.ax_joint)
		for e, i in enumerate(data.URQ.unique()):
			ind = data.index[data['URQ']==i].tolist()
			if not ind:
				continue
			elif(len(XPCA)>max(ind)):
				plt.plot(XPCA[ind, 0], XPCA[ind, 1], 'o', color=colors2[e])
		g.ax_joint.plot(np.mean(x), np.mean(y), marker=marker, markersize=15, markeredgecolor=color,
		                markerfacecolor=color, markeredgewidth=2)
		# txt = ax.text(np.mean(x), np.mean(y), "Cluster " + str(s + 1), fontproperties=font0, size=30, color=color)
		# txt.set_path_effects([pte.Stroke(linewidth=2.5, foreground='white'), pte.Normal()])
		sns.distplot(XPCA[np.where(y_pred == s)[0], 0], bins=50, kde=True, kde_kws={"color":color, 'shade':True, 'lw':2, 'alpha':0.3},
		             hist=False, rug=False, ax=g.ax_marg_x)
		sns.distplot(XPCA[np.where(y_pred == s)[0], 1], bins=50, kde=True, vertical=True,
		             kde_kws={"color": color, 'shade': True, 'lw': 2, 'alpha': 0.3},
		             hist=False, rug=False, ax=g.ax_marg_y)

	plt.show()

	plt.close('all')
