import sys
print(sys.path)
# 현재 작업 공간

from test_import import p62_import
p62_import.sum2()

print("=================================================")

from test_import.p62_import import sum2
sum2()