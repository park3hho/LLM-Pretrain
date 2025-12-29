# LLM Pretrain
***Duration***: 2025 Nov 07 - 2025 Dec 14  
***Curriculum***: Training - Inference  
***Imitated LLM Model***: `GPT 2`  

**배경:**

- 온 세상의 기본이 되어가고 있는 머신러닝의 힘을 보고, 겉핥기라도 알아야겠다는 생각이 듦.
- 와중 수 많은 머신러닝 방식이 트랜스포머를 이용한 학습 방식이 위주가 되고 있는 것을 파악함.

**목표:** 

- Pytorch를 활용한 Pretrain 과정 이해하기.
- 데이터셋 처리하는 방식 이해하기.
- 머신러닝 관련 내용들 이해하기

**개발 환경**:

- OS: Window
- IDE: Pycharm
- Docker
    - 라이브러리나 기타 파일들이 설치되는 것이 싫어서 Docker로 관리함.
- Dataset: Harry Poter

**수확**

- **Pretrain의 기댓값** - Pretrain과정은 지금 쓰고 있는 LLM과 달리 간단한 `Next Token Prediction Model`이다. 대화의 영역은 SFT와 FFT를 거쳐야 가능하다.
- **DataProcessing** - Pretrain의 데이터셋 준비과정은 생각보다 간단하다. 막대한 양의 내용을 집어넣기만 하면된다. 물론 그 과정에서 데이터 출처에 따라 가공해야하는 과정이 달라지지만, 쓸모없는 띄어쓰기를 삭제해주는 과정. 중복 데이터를 지우는 과정 등등 생각했던 내용 내이다.
- **Tokenization** - 이것도 생각보다 간단했다. 이미 나와있는 라이브러리를 사용하면 된다. Tiktokenization에서 GPT 2가 사용한 토크나이즈 방식을 선택할 수 있다. 문장을 임베딩할 때 슬라이딩 방식으로 몇개씩 겹쳐서 넣을 것인지, 문장의 크기
- **QKV** - 이게 가장 헷갈렸던 부분이다. 각 Query, Key, Value가 무엇을 의미하는지, 왜냐하면 모두 똑같은 값을 나누었기 때문에…
- **Multi Head Attention** - 재밌는 방식이었다. Head가 하나의 공간을 얼마나 나누어 볼 것인지 정하는 것인지,

- 기타 등등…
    - Linearization(선형 변환), 잔차계산(Residual), LayerNormalization, 활성 함수(Activation Function), softmax, dropout etc…

## AFTER Process
### Losses Graph
![훈련 손실 그래프](images/loss.png)




## Inference

--- 
Github: https://github.com/park3hho/LLM-Pretrain

## Sources
### Notion Page:

### Main
https://github.com/HongLabInc/HongLabLLM/blob/main/01_pretraining.ipynb
https://www.youtube.com/watch?v=osv2csoHVAo&t=724s
### Dataset Splitting
https://pozalabs.github.io/Dataset_Splitting/
