import numpy as np
l1 = [-145.,  -92., -212., -161., -528., -520., -366., -365., -174., -152., -314., -761.]
l2 = [ -141., -83., -309., -242., -703., -711., -584., -524., -257., -238., -559., -1056.]
# print(sum(l1))

# Print the mean of the list
print(l1)
print(type(l1))
print(len(l1))
z = (sum(l1) / len(l1))
print("\nl1 mean", z)
print("l2 mean", sum(l2) / len(l2))

# Print the sum of every 2nd element in the list
print("\nsums l1")
print(sum(l1[::2]))
print(sum(l1[1::2]))

print("sum l2")
print(sum(l2[::2]))
print(sum(l2[1::2]))