from deepface import DeepFace

result = DeepFace.verify(r"C:\Users\bitcamp\Desktop\opencv_dnn_202005\image\ujung.jpg")

print("Is verified: ", result["verified"])
print()

