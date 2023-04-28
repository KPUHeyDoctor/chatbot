# chatbot

## Start
### embadding.ipynb  

**모듈 import**
```
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
```  

**SentenceBERT 모델 로드**  
```
model = SentenceTransformer('jhgan/ko-sroberta-multitask')

sentences = ["안녕하세요?", "한국어 문장 임베딩을 위한 버트 모델입니다."]
embeddings = model.encode(sentences)

print(embeddings)
```  
<br>

**데이터셋 로드**  
```
df = pd.read_csv('input.csv', delimiter=',')

df.head()
```  
```
df = df[~df['챗봇'].isna()]

df.head()
```  
```
model.encode(df.loc[0, '유저'])
```  
<br>

**유저 대화내용 인코딩**  
```
df['embedding'] = pd.Series([[]] * len(df)) # dummy

df['embedding'] = df['유저'].map(lambda x: list(model.encode(x)))

df.head()
```  
```
df.to_csv('answer.csv', index=False)
``` 

위 순서대로 셀 실행


## File info
app.py: 챗봇 엔진(서버)
<br>

embadding.ipynb: csv파일을 읽어와 임베딩(벡터화) 처리
<br>

input.csv: 사용자 예상 발화에 대한 데이터셋
<br>

answer.csv: input.csv가 embadding.ipynb를 거쳐 임베딩이 완료된 데이터셋
<br>

wellness_dataset.csv -> 정신감정분류 데이터셋