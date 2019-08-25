import matplotlib.pyplot as plt
import numpy as np

# Load the previous scores and calculated running mean of 100 runs
# ---------------------------------------------------------------------------------------
with np.load('scores.npz') as data:
    scores = data['arr_0']
cum_sum = np.cumsum(np.insert(scores, 0, 0))
rolling_mean = (cum_sum[100:] - cum_sum[:-100]) / 100

# Make a pretty plot
# ---------------------------------------------------------------------------------------
plt.figure()
x_max = len(scores)
y_min = scores.min() - 1
x = np.arange(x_max)
plt.scatter(x, scores, s=2, c='k', label='Raw Scores', zorder=4)
plt.plot(x[99:], rolling_mean, lw=2, label='Rolling Mean', zorder=3)
plt.scatter(x_max, rolling_mean[-1], c='g', s=40, marker='*', label='Episode {}'.format(x_max), zorder=5)
plt.plot([0, x_max], [13, 13], lw=1, c='grey', ls='--', label='Target Score = 13', zorder=1)
plt.plot([x_max, x_max], [y_min, rolling_mean[-1]], lw=1, c='grey', ls='--', label=None, zorder=2)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.legend()
plt.xlim([0, x_max + 5])
plt.ylim(bottom=y_min)
plt.show()
plt.savefig('score.png')
