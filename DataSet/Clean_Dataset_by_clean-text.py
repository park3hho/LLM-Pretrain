from cleantext import clean
import os


def clean_text_file(input_filename):
    # 입력 파일 이름에서 'clean_' 접두사를 추가한 새로운 파일 이름 생성
    output_filename = f"clean_{input_filename}"

    # 파일 읽기
    with open(input_filename, 'r', encoding='utf-8') as file:
        text = file.read()

    # clean-text를 사용하여 텍스트 정리
    cleaned_text = clean(text,
                        clean_all = False,
                        extra_spaces = True,
                        stemming = False,
                        stopwords = False,
                        lowercase = True,
                        numbers = False,
                        punct = False,
                        reg = '',
                        reg_replace = ''
                        ) # 불필요한 공백 제거
    # clean_text 사용법
    """ 
    1. clean_all (기본값: True)
    이 옵션이 True이면 기본적인 텍스트 클리닝 작업을 모두 수행합니다.
    포함되는 기본 작업:
    URL, 이메일 제거
    구두점 제거
    불필요한 공백 제거
    숫자 제거
    모든 텍스트를 소문자로 변경
    텍스트에서 불필요한 문자를 제거

    2. extra_spaces (기본값: True)
    True일 때, 텍스트 내에 있는 불필요한 공백(예: 연속된 공백)을 하나의 공백으로 줄여줍니다.
    False일 경우, 여러 공백을 그대로 두고, 공백을 추가로 제거하지 않습니다.

    3. stemming (기본값: False)
    True일 때, 텍스트에서 어간 추출(stemming)을 수행하여, 단어들을 그 기본 형태로 변환합니다.
    예: "running" → "run", "better" → "good".
    False일 경우, 어간 추출을 수행하지 않습니다.

    4. stopwords (기본값: False)
    True일 경우, 텍스트에서 **불용어(stopwords)**를 제거합니다. 불용어는 의미가 적거나 문장에서 중요하지 않은 단어들로, 예를 들어 "the", "is", "in", "at" 등이 있습니다.
    False일 경우, 불용어를 제거하지 않습니다.

    5. lowercase (기본값: False)
    True일 경우, 모든 텍스트를 소문자로 변환합니다. 대소문자 구분을 없애고 싶을 때 사용합니다.
    False일 경우, 소문자로 변환하지 않습니다.

    6. numbers (기본값: False)
    True일 경우, 텍스트에서 숫자를 제거합니다.
    False일 경우, 숫자를 그대로 유지합니다.

    7. punct (기본값: False)
    True일 경우, 구두점(예: .,!?)을 텍스트에서 제거합니다.
    False일 경우, 구두점을 그대로 유지합니다.

    8. reg (기본값: '')
    텍스트를 **정규 표현식(Regex)**을 사용해 변환하려는 경우, 이 옵션을 사용하여 정규 표현식을 정의할 수 있습니다.
    예를 들어, 특정 패턴을 찾고 이를 변경하거나 제거하는 데 사용합니다.
    reg_replace와 함께 사용하여 텍스트 내에서 원하는 부분을 대체하거나 수정할 수 있습니다.

    9. reg_replace (기본값: '')
    reg에서 정의한 정규 표현식에 매칭되는 부분을 대체할 문자열을 지정합니다.
    reg와 함께 사용되어, 텍스트에서 특정 패턴을 찾아 다른 문자열로 대체할 수 있습니다.
    """

    # 정리된 텍스트를 새로운 파일로 저장
    with open(output_filename, 'w', encoding='utf-8') as file:
        file.write(cleaned_text)

    print(f"Cleaned text has been saved to {output_filename}")


# 사용 예시
input_file = "07 Harry Potter and the Deathly Hallows.txt"  # 여기에는 정리하고자 하는 파일명을 넣으세요
clean_text_file(input_file)
