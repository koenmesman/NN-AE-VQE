a = 2
b = 4
print(list(range(100))[2:-1:10])
lst = list(range(100))
step = 10
split = [lst[i:i+step] for i in range(len(lst))[2:-1:step]]
for i in split:
    print(i)