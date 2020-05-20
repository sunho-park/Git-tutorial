a = {'name' : 'yun', 'phone':'010', 'birth' : '0511'}

'''a = 0
for(i=1, i=100, i++){
    a= i + a
}

print(a)

for i in 100:

a i 결과a
0 1 1
1 2 3
3 3 6
6 4 10

213 100'''

for i in a.keys():
    print(i)

a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for i in a:
    i = i*i
    print(i)
#   print('melong') melong 이 10번출력됨
#print('melong') melong 이 1번만 출력됨


a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for i in a:
    print(i)
    
## while 문

'''
while 조건문 :           #참일 동안 계속 돈다
    수행할 문장
'''

### if 문

if 1:
    print('True')
else:
    print('False')


if 3:
    print('True')
else:
    print('False')


if 0:
    print('True')
else:
    print('False')

    
if -1:
    print('True')
else:
    print('False')

'''비교 연산자

>, <, ==, !=, >=, <=
'''
a = 1
if a == 1:
    print('출력잘되')

money = 10000
if money >= 30000:
    print('한우먹자')
else:
    print('라면먹자')


###조건 연산자
#and, or, not
money = 20000
card = 1
if money >= 30000 or card == 1:
    print("한우먹자")
else:
    print('라면먹자')

##################################################
# break, continue
print("=============================")
jumsu = [90, 25, 67, 45, 80]
number = 0
for i in jumsu:
    if i >= 60:
        print("경] 합격 [축")
        number = number+1

print("합격인원 : ", number, "명")


print("=============================")
jumsu = [90, 25, 67, 45, 80]
number = 0
for i in jumsu:
    if i < 30:
        break
    if i >= 60:
        print("경] 합격 [축")
        number = number+1

print("합격인원 : ", number, "명")

print("=============================")
jumsu = [90, 25, 67, 45, 80]
number = 0
for i in jumsu:
    if i < 60:
        continue        #25면 그 하단부분 실행하지 않고 다시 for 문으로 돌아감
    if i >= 60:
        print("경] 합격 [축")
        number = number+1

print("합격인원 : ", number, "명")



