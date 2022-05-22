import random
list1=[i for i in range(1,6)]
list2=[i for i in range(1,4)]
main_list = list(set(list1) - set(list2))
print(random.sample(range(1,6), 5))