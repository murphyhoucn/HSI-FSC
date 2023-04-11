import matplotlib as mpl
import matplotlib.pyplot as  plt

colors = ['black', 'gray', 'lime', 'cyan', 'forestgreen', 'hotpink', 'saddlebrown',
            'purple', 'red', 'yellow', 'blue', 'steelblue', 'olive', 'sandybrown', 'mediumaquamarine', 'darkorange',
            'whitesmoke']
print(len(colors))

cmap = mpl.colors.ListedColormap(colors)
print(cmap)