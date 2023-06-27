import matplotlib.pyplot as plt
import numpy as np

losses = [92.5282577587667, 94.30095211201107, 74.80092823731503, 113.91446815937212, 119.7658058079108, 103.42972183668628, 100.32099052930158, 125.26087119212849, 110.72094670600528, 107.54478525674769, 92.87485695909047, 80.20899494600029, 111.36633857838417, 111.08811707985431, 99.84096240901476, 103.04893510409751, 135.94511543349188, 114.56616839968319, 109.6224332701056, 97.10760057396128, 77.32367017484317, 107.65385374233679, 105.09608198740892, 98.26569659195249, 102.5111849758392, 129.4975161902862, 116.64887758172455]

x = np.arange(len(losses))

 
plt.scatter(x, losses, c ="blue")
plt.ylim((0, 150))
 
# To show the plot
plt.show()