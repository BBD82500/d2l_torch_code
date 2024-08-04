import torch
from d2l import torch as d2l

d2l.set_figsize()
img = d2l.plt.imread('catdog.jpg')
fig = d2l.plt.imshow(img)

dog_bbox, cat_bbox = [60, 45, 378, 516], [400, 112, 655, 493]
fig.axes.add_patch(d2l.bbox_to_rect(dog_bbox, 'blue'))
fig.axes.add_patch(d2l.bbox_to_rect(cat_bbox, 'red'))

d2l.plt.show()