import matplotlib.pyplot as plt
import numpy as np

dir = 'RESULTS/opentest_results/'

x = np.linspace(0.34, 1.6, 127)
ans = []
ans1 = []
GT = []
legend = []
for i in range(1, 11):
    with open(dir + 'out{}'.format(i), 'r') as f:
        lines = f.readlines()
    res = []
    res1 = []
    for line in lines:
        l = line.split()
        if l[0] == 'reject2all:':
            res.append(float(l[-1]))
        elif l[0] == 'reject2old:':
            res1.append(float(l[-1]))
        elif l[0] == 'acc_open(no':
            GT.append(float(l[-1]))
    ans.append(res)
    ans1.append(res1)
    legend.append('out{}'.format(i))
ans = np.array(ans)
ans1 = np.array(ans1)
plt.figure()
for res in ans:
    plt.plot(x, res)
plt.legend(legend, loc='upper left')
fig, axes = plt.subplots(nrows=2, ncols=5)

for i in range(10):
    ax = axes[i // 5, i % 5]
    ax.plot(x, ans[i], '-')
    ax.plot(x, ans1[i], '-.')
    ax.plot(x, np.ones(x.shape) * GT[i])
    ax.legend(['reject2all', 'reject2old', 'no_reject'], loc='upper left')
    ax.set_title('out{}'.format(i + 1))
plt.show()
