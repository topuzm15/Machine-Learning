import inception

inception.download()

model = inception.Inception()

image = 'images.jpeg'
pred = model.classify(image_path=image)
model.print_scores(pred, k=10, only_first_name=True)

