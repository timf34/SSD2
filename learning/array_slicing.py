sample_array = [i for i in range(10)]
print(sample_array)

step_size = len(sample_array) // 2
print("step size", step_size)

# Now I want to loop through the whole array with a step size of 5 (i.e. every 5th element)
for i in range(step_size):
    for j in range(i, len(sample_array), step_size):
        # print the value
        print(sample_array[j])

print("sai dude")
# Now I want to loop through the array again, with a step size of 5, but to visit each element using slicing
# This is the one:)
for i in range(step_size):
    print(sample_array[i::step_size])

# Now to repeat with a better sample array
sample_rewards = [0., -1.,  1.,  0., -1.,  0.,  -50., -1., 0, 0]

for i in range(step_size):
    print(sample_rewards[i::step_size])


array_to_sum = [1., -50]
# Some this array
print(sum(array_to_sum))

