import matplotlib.pyplot as plt

x = [25, 50, 100, 250, 500]
y = [0.0836508963522, 0.166864061986, 0.242782762145, 0.740473459369, 0.961789417496]

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(x, y, 'ro', linestyle='--', color='r', label='LSTM: 512 nodes X 2 layes')
for i, j in zip(x, y):
    ax.annotate("(%d, %.2f)" % (i, j), xy=(i + 10, j))

plt.xlabel('Epochs')
plt.ylabel('Modified BLEU Score')
plt.legend(loc='upper left')
plt.show()
