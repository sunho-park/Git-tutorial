# 3. 딕셔너리 # 중복 x
#{키 : 벨류}
#{key : value}

a = {1 : 'hi', 2: 'hello'}
print(a)
print(a[1])

b = {'hi' : 1, 'hello' : 2}
print(b['hello'])

# 딕셔너리 요소 삭제

del a[1]
print(a)

del a[2]
print(a)

a = {1 : 'a', 1 : 'b', 1 : 'c'}
print(a)                           #가장 마지막 요소가 출력

b = {1 : 'a', 2 : 'a', 3 : 'a'}
print(b)

a={'name' : 'yun', 'phone' : '010', 'birth' : '0511'}
print(a.keys())
print(a.values())
print(type(a))
print(a.get('name'))
print(a['name'])
print(a.get('phone'))
print(a['phone'])                        # 조건문, 반복문 // 구글링 중요




