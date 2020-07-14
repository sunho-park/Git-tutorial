from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello333():
    return "<h1>hello youngsun world</h1>"
# <h1>이 가장큼 h2 > h3

@app.route('/bit')  # ip 뒤에 /bit 입력
def hello334():
    return "<h1>hello bit computer world</h1>"

@app.route('/gema')  # ip 뒤에 /bit 입력
def hello335():
    return "<h1>hello GEMA computer world</h1>"


@app.route('/bit/bitcamp')  # ip 뒤에 /bit 입력
def hello336():
    return "<h1>hello bitcamp world</h1>"

if __name__ == '__main__':
    app.run(host='127.0.0.1', port= 8888, debug=True)

# http://127.0.0.1 자기컴퓨터 = 로컬호스트
# flask 서버를 구동시킨것
