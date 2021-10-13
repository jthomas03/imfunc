from imMF.imfunc import imFunc
import matplotlib.pyplot as plt

image = imFunc.readfile('drift_track001.sxm','Z','forward')

#image = imFunc.planesubtract(image)

#imFunc.image_plot(image)

spec = imFunc.readspec('dI_dV01967.dat')

di = []
dv = []
for i in range(1,len(spec)):
    dv.append(float(spec[i][0]))
    di.append(float(spec[i][4]))

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Bias (V)')
ax.set_ylabel('dI/dV (arb. units)')
ax.plot(dv,di,c='r',linewidth=2, label = '$WS_{2}$')
ax.legend()
plt.show()