import sys

action =int(sys.argv[1])

lights = []
for i in range(5):
    lights += [action%4]
    action //= 4

temp = action%10 + 20
action //= 10


print(lights,temp)
