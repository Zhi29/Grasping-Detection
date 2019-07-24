root = "/home/yunchu/python_workspace/data_test/"
minlines = 100000
min_txt = ""
for i in range(1, 3001):
    with open(root + str(i) + '.txt') as f:
        text = f.readlines()
        size = len(text)
        if size < minlines:
            minlines = size
            min_txt = str(i) + '.txt'
print(minlines, min_txt)  