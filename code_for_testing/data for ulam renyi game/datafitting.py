import csv
from matplotlib import pyplot as plt
import numpy as np
def piecewise(x):
    if(x<10): return 1
    elif(x<18): return 2
    elif(x<58): return 3
    elif(x<162): return 4
    elif(x<456): return 5
    else: return 6
naivefile = open('naive_reliability_0.8_data_traits_9_1000_times.csv')
ulamfile = open('ulam_reliability_0.8_data_traits_9_1000_times.csv')
naive=[(float(x[1]),float(x[7])/float(x[4])) for x in list(csv.reader(naivefile))[1:] if x[0]!=''][2:]
ulam=[(float(x[1]),float(x[7])/float(x[4])) for x in list(csv.reader(ulamfile))[1:] if x[0]!=''][2:]
ulamerror=[(float(x[1]),float(x[3])) for x in list(csv.reader(ulamfile))[1:] if x[0]!='']

fig = plt.figure()

ax1 = fig.add_subplot(111)

print(naive)
ax1.scatter(*zip(*ulam),s=10, c='b',marker="s", label='Ulam Renyi')
#plt.plot(list(range(512)),list(map(piecewise,range(0,512))),label='e')
ax1.scatter(*zip(*naive),s=10, c='r',marker="o", label='DCFECC')
plt.xlim(0,512)
#plt.ylim(0,1)
#plt.plot([0,512], [9,9], color='r', linestyle='-', linewidth=2, label='DCFECC')

plt.xlabel('Number of classes')
plt.ylabel('Probability of correct classification per question asked')
plt.legend()
plt.show()
