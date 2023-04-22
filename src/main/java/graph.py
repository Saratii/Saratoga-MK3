import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib import rcParams

rcParams['toolbar'] = 'None'
style.use('dark_background')


with open('../logs\\log-graph', 'r') as f:
    data = [line.strip().split(", ") for line in f]
    data = [row for row in data if len(row) == 2]
    x = [float(x) for x, y in data]
    y = [float(y) for x, y in data]
    plt.plot(x, y, marker='o', markerfacecolor='black')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training ')
    plt.show()
f.close()
