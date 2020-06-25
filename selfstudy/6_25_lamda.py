def get_square(a):
    return a**2
print('3의 제곱은:', get_square(3))
              # 입력인자 : 반환인자
lambda_square = lambda x: x**2
print('3의 제곱은:', lambda_square(3))

a = [1, 2, 3]
squares = map(lambda x : x**2, a)
print(list(squares))

