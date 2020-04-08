import csv
with open('E:/meta_learning/mini_imagenet/train.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    next(csv_reader, None)
    for i, row in enumerate(csv_reader):
        if i == 5:
            break
        print(row[0],row[1])