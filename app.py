from flask import Flask
from flask import request
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import sys

app = Flask(__name__)

@app.route('/')
def hello():
    return "HelloWorld"


@app.route("/model",methods=['POST'])
def decision():
    json_data = request.get_json() 
    text = json_data['userRequest']['utterance']

    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    # df = pd.read_csv('wellness_dataset.csv')
    df = pd.read_csv('answer.csv')
    df['embedding'] = df['embedding'].apply(json.loads)
    embedding = model.encode(text)
    #미리 임베딩한 데이터셋에서 사용자가 입력한 문장의 임베딩과 비교를 하여 가장 유사한것을 찾음
    df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())

    df.head()

    answer = df.loc[df['distance'].idxmax()]
    print('[질문]', text)
    print('[구분]', answer['구분'])
    print('[유사한 질문]', answer['유저'])
    print('[유사도]', answer['distance'])
    print('\n[챗봇 답변]', answer['챗봇'])

    # 사용자 발화와 답변의 유사도가 0.6 미만이면 예외 처리 답변을 반환합니다.
    if(answer['distance'] < 0.6):
        useranswer = "증상을 조금 더 구체적으로 말씀해주세요. (예시: 두통이 심한데 어느 병원을 가야할까요?)"
    else:
        useranswer = str(answer['챗봇'])


    responseBody = {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": useranswer
                    }
                }
            ]
        }
    }
    
    return responseBody
    

@app.route('/model', methods=['GET'])
def connect():
    return "Backend Server Connect"


if __name__ == '__main__':
    app.run(host='0.0.0.0' ,port = 5002, debug=True)