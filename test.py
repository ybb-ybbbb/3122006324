f1 = open("E:\桌面\测试文本\orig.txt", 'r', encoding='UTF-8')
f2 = open("E:\桌面\测试文本\orig_0.8_add.txt", 'r', encoding='UTF-8')
print(f1.read())
print(f2.read())
f1.close()
f2.close()
