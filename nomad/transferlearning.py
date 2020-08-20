

model_A = keras.models.load_model("my_model_A.h5")
model_B_on_A = keras.models.Sequential(model_A.layers[-1:])
model_B_on_A.add(keras.layers.Dense(1, activation='sigmoid'))

model_A_clone = keras.models.clone_model(model_A)
model_A_clone.set_weights(model_A.get_weights())


for layer in model_B_on_A.layers[:-1]:
    layer.trainable = False

model_B_on_A.compile(loss='binary_crossentropy', optimizer="sgd", metrics=["accuracy"])

#############
histroy = model_B_on_A.fit(X_train_B, y_train_B, epochs=4, validation_data = (X_valid_B, y_valid_B))

for layer in model_B_on_A.layers[:-1]:
    layer.trainable = True

optimizer = keras.optimizer.SGD(lr=1e-4)    # 기본 학습률은 1e-2
model_B_on_A.compile(loss="binary_crossentropy", optimizer = optimizer, metrics = ["accuracy"])

history = model_B_on_A.fit(X_train_B, y_train_B, epochs=16, validation_data = (X_valid_B, y_valid_B))

model_B_on_A.evaluate(X_test_B, y_test_B)

