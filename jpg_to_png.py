from PIL import Image

for i in range(10):
    im1 = Image.open('data/membrane/test_data_jpg/{}.jpg'.format(i))
    im1.save('data/membrane/test_data/{}.png'.format(i))