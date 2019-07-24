root = "/Volumes/My\ Passport/learning/Jacquard_Dataset_Merged/"
minlines = 100000
for i in range(1, 54466):
    with open(root + str(i) + '.txt') as f:
        text = f.readlines()
        size = len(text)
        if size < minlines:
            minlines = size
            min_txt = str(i) + '.txt'
print(minlines, mintxt)