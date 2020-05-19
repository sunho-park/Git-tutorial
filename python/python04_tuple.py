# 2. 튜플

# 리스트와 거의 같으나, 삭제, 수정이 안된다.
a = (1, 2, 3)
b = 1, 2, 3

c = [1, 2, 3]

print(type(a))
print(type(b))
print(type(c))

# a.remove(2) 
# print(a)                   AttributeError: 'tuple' object has no attribute 'remove'
 
print(a + b)
print(a*3)

# print(a - 3)               TypeError: unsupported operand type(s) for -: 'tuple' and 'int'


