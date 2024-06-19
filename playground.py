from numpy import load
from PIL import Image
import numpy as np


def load_image(infilename):
    img = Image.open(infilename)
    img.load()
    data = np.asarray(img, dtype="int32")
    return data


def save_image(npdata, outfilename):
    img = Image.fromarray(np.asarray(np.clip(npdata, 0, 255), dtype="uint8"), "L")
    img.save(outfilename)


data = load("../data/octmnist.npz")
print(data.files)
# print(data["test_images"][0]

# lst = data.files
# for item in lst:
#     print(item)
#     print(data[item])


# im = Image.fromarray(data["test_images"][0])
# im.save("your_file.jpeg")

save_image(data["test_images"][0], "imgs/octmnist_test_image_0.jpg")
save_image(data["test_images"][1], "imgs/octmnist_test_image_1.jpg")
save_image(data["test_images"][2], "imgs/octmnist_test_image_2.jpg")
save_image(data["test_images"][3], "imgs/octmnist_test_image_3.jpg")
save_image(data["test_images"][4], "imgs/octmnist_test_image_4.jpg")

with open("imgs/labels.txt", "w") as f:
    for i in range(5):
        f.write(f"octmnist_test_image_{i} - {str(data["test_labels"][i])}\n")

# (
#     print("worked")
#     if load_image("octmnist_img.jpg").all() == data["test_images"][0].all()
#     else print("did not work")
# )
# print(load_image("octmnist_img.jpg").shape)
print(data["test_labels"][0])
