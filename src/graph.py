import matplotlib.pyplot as plt
from matplotlib import style
# Set the style to dark mode
style.use('dark_background')

with open('/Users/propleschmaren/Desktop/Saratoga-MK3/log-graph', 'r') as f:
    data = [line.strip().split(', ') for line in f]

    x_vals = [float(x) for x, y in data]
    y_vals = [float(y) for x, y in data]

    plt.plot(x_vals, y_vals, marker = 'o', markerfacecolor='black')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training ')
    plt.show()