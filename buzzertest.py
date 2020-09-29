import RPi.GPIO as GPIO                    # RPi.GPIO에 정의된 기능을 GPIO라는 명칭으로 사용
import time                                # time 모듈

GPIO.setmode(GPIO.BCM)                     # GPIO 이름은 BCM 명칭 사용

buzz = 1                                   # 핀번호 1 대신 buzz 명칭사용을 위해 치환
GPIO.setup(buzz, GPIO.OUT)                 # GPIO buzz핀(1)을 출력으로 설정

freq = [523,587,659,698,784,880,988,1047]  # freq 리스트 ( 음계 주파수 리스트 )

def makeTone(freq):                        # 매개변수로freq를 받는 makeTone함수 정의 시작
	scale = GPIO.PWM(buzz, freq)           # buzz핀으로 freq(Hz) PWM파형을 생성하는 scale 정의
	scale.start(10)                        #      scale 시작
	time.sleep(0.5)                        #      0.5초 대기
	scale.stop()                           #      scale 정지 ( makeTone함수 정의 끝 )

try:

	for x in freq:                         #   freq 리스트 만큼 반복 시작 
		makeTone(x)                        #   makeTone 함수에 freq 
	GPIO.cleanup()                         #   GPIO 관련설정 Clear

except KeyboardInterrupt:                  # Ctrl-C 입력 시 
	GPIO.cleanup()                         # GPIO 관련설정 Clear 