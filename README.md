# 공장 A ESS 팀 공통 기준 코드

같은 부하, 요금, ESS 제약에서 원래 운영, 시간대 요금 규칙, 피크 방어 규칙, Dueling Double DQN을 비교합니다.

| 항목 | 기본 조건 |
|---|---|
| 데이터 | 공장 A, 2012년 학습 / 2013년 검증 |
| 설비 | 2012년 전체 최대 부하의 5% 출력, 2h 용량, 왕복효율 83% |
| 에피소드 | 달력 기준 1개월, 시간당 1스텝, 월초 SoC 50%, 월말 부족분 정산 |
| 학습 | γ=1, 균등 경험 재생, 1-step, Adam 학습률 0.0003, CPU 1스레드 |
| 모델 선택 | 12 에피소드마다 2013년 순절감을 비교해 최고 모델 저장 |
| 요금 | 팀이 제공한 2021 산업용(을) 고압 A 선택Ⅱ 시나리오, 당월 피크 기준 |

Python 3.14.4에서 검증했습니다. `data` 폴더를 만든 뒤 팀이 사용하는 `factory_A_hourly_kwh_v3.csv` 원본을 그 안에 넣으세요. 다른 위치라면 `--data`에 경로를 지정할 수 있습니다. 원본 데이터는 팀 내부에서 전달하고 코드와 별도로 관리합니다.

Windows CMD에서 아래 명령을 실행하세요.

```cmd
python -m venv .venv
.venv\Scripts\python -m pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple -r requirements.txt
.venv\Scripts\python -m unittest -v
.venv\Scripts\python run.py --episodes 300 --seed 0
```

설치와 실행만 확인하려면 `--episodes 2`로 실행하세요. 팀 비교는 모두 300회로 맞추고, 나중에 시드 1·2를 같은 방식으로 추가합니다. 같은 시드라도 Python·PyTorch·CPU 환경이 다르면 결과가 달라질 수 있으므로 실행 환경도 확인합니다.

`outputs` 아래에 실행별 폴더가 생성됩니다. `validation_annual.csv`에서 네 정책의 연간 절감액을, `validation_history.csv`에서 체크포인트별 성능을 비교할 수 있습니다. `best_model.pt`와 `last_model.pt`를 각각 저장합니다. `settings.json`의 설정·시드·데이터 SHA-256·라이브러리 버전을 대조하면 같은 조건으로 실행했는지 확인할 수 있습니다. 공장 A 원본 SHA-256은 `3b128d0358017ebd8bc2de264469db8f9f225031fa684fb19cb16495e4d37faa`입니다.

`config.json`에 계절별 단가, 공휴일, 겨울 22~23시 최대부하, 기후환경요금과 분기별 연료비조정을 기록했습니다. 기본요금은 중간·최대부하 시간의 당월 최대수전과 계약전력 30% 중 큰 값에 8,320원/kW를 곱합니다. 계약전력은 2012년 전체 최대부하의 1.1배라는 팀 가정을 사용합니다. 운영 순절감은 전기요금 절감에서 열화비와 종료 잔량 부족 정산을 뺀 값입니다. 래칫·세금·설치비·유지비는 기본 실험 범위에서 제외합니다. 2014년은 최종 평가용으로 남깁니다.

GitHub에 공유할 때는 이 폴더를 저장소 루트로 사용하세요. 저장소에는 소스·설정·설치 안내를 올리고, 원본 CSV·가상환경·실험 출력은 `.gitignore`로 제외합니다.
