#자료형

# 1.리스트

a = [1,2,3,4,5]
b= [1,2,3,'a','b']
print(b)
print(a[0]+ a[3]) 
# print(b[0], + b[3]) 오류
print(type(a))
print(a[-2])
print(a[1:3])
print('---------------------------------')
a = [1, 2, 3, ['a', 'b', 'c']]
print(a[1])
print(a[-1]) #[a, b, c]
print(a[-1][1]) # b
print('---------------------------------')

#1-2. 리스트 슬라이싱

a = [1,2,3,4,5]
print(a[:2])
print('---------------------------------')

#1-3. 리스트 더하기

a = [1, 2, 3]
b = [4, 5, 6]
print(a + b) #      리스트 + 리스트 // class numpy.ndarray, numpy 로 계산하면 5,7,9 로 나올수 있음, numpy 속도가 무진장 빠름, 같은 type 만 사용가능

c = [7, 8, 9, 10]
print(a+c)

print(a*3) # numpy로 하면 3, 6, 9 로 나옴

# print(a[2] + 'hi')
print(str(a[2]) + 'hi')

f= '5'
# print(a[2] + f)
print(a[2] + int(f))

# 리스트 관련 함수

a = [1, 2, 3]
a.append(4)
print(a)

# a = a.append(5) # 오류
a = [1, 3, 4, 2]
a.sort()
print(a)

a.reverse()
print(a) #[4,3,2,1]

print(a.index(3)) # == a[3]
print(a.index(1)) # == a[1] 

a.insert(0, 7) #[7, 4, 3, 2, 1]
print(a)
a.insert(3, 3)  #[7, 4, 3, 3, 2, 1]
print(a)

a.remove(7) # [4, 3, 3, 2, 1]
print(a)

a.remove(3) #[4, 3, 2, 1]
print(a)


