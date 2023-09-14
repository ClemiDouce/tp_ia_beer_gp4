from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

datas = load_digits(n_class=2)
print(datas.data[0])
print(datas.images[0])
print(datas.target)

fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))

for ax, image, label in zip(axes, datas.images, datas.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)
fig.savefig("./output/show_digit.png")
