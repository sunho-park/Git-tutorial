# parameter 튜닝시 반드시 learning_rate 를 넣어라
weight = 0.5
input = 0.5 
goal_prediction = 0.8

lr = 0.001 # 0.1 / 1 / 0.0001  값 변경해보기

for iteration in range(100):
    prediction = input * weight
    error = (prediction - goal_prediction) ** 2

    print("Error : " + str(error) + "\tPrediction : " + str(prediction)) # \t 띄어쓰기 탭

    up_prediction = input * (weight + lr)
    up_error = (goal_prediction - up_prediction)**2

    down_prediction = input * (weight - lr)
    down_error = (goal_prediction - down_prediction)**2

    if(down_error < up_error):
        weight = weight - lr
    if(down_error > up_error):
        weight = weight + lr


