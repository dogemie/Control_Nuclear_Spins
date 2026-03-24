# Control Nuclear Spins: NV 센터 내 13C 큐비트 초기화 게이트 최적화

## 1. 개요 (Introduction)

본 프로젝트는 다이아몬드 격자 내 질소-공공(NV, Nitrogen-Vacancy) 센터 양자 시스템을 기반으로, 13C 핵스핀 큐비트의 초기화 게이트를 최적화하는 알고리즘 및 머신러닝 파이프라인을 구현합니다. 양자 컴퓨팅에서 큐비트를 높은 충실도(Fidelity)로 특정 상태로 초기화하는 것은 필수적이나, 이를 위한 제어 파라미터 탐색에는 막대한 계산 비용이 소요됩니다. 

본 연구는 양자 회로 내 단일 NV 큐비트 조작과 13C 큐비트의 Swap 게이트 초기화 과정을 최적화하기 위해, 로컬 및 글로벌 최적화 알고리즘과 인공신경망(ANN)을 결합하여 연산 반복 횟수를 대폭 감소시키고 효율적으로 초기화 파라미터를 탐색합니다.

## 2. 연구 목표 (Objectives)

* **NV Center 단일 큐비트 최적화:** NV 큐비트를 타겟 상태로 이동시키기 위한 최적의 회전 각도 theta 및 phi를 탐색합니다.
* **13C Spin 초기화 연산 최적화:** NV 스핀과 13C 스핀 간의 제어 게이트(Controlled X, Controlled Z, Z gate)를 구성하여 Swap 연산을 수행합니다.
* **충실도 및 시간 비용의 균형점 도출:** 충실도 99% 이상을 만족하는 조건 하에서, 초기화 게이트 소요 시간 수식인 2 * Xtau * XN + Ztau * ZN 값이 최소가 되는 파라미터 조합을 찾습니다.

## 3. 연구 방법 및 접근 (Methodology)

### 3.1. 양자 게이트 최적화 알고리즘 비교

* **비용 함수 정의:** 타겟 상태와 계산된 밀도 행렬(Density Matrix) 투영값(x, y, z) 간의 오차를 비용 함수(Cost Function)로 정의하여 최적화를 진행했습니다.
* **알고리즘 성능 비교:** Powell, Nelder-Mead, Differential Evolution, Dual Annealing, SHGO 등 다양한 Local/Global Optimizer의 연산 속도와 오차를 비교 분석했습니다.
* **최적 모델 선정:** 분석 결과, Nelder-Mead 알고리즘이 Powell 대비 오차율은 소폭 높으나 계산 속도가 약 3배 빠르며 효율적인 연산 비율을 보여 단일 큐비트 제어의 주요 알고리즘으로 채택했습니다.

### 3.2. 머신러닝(ANN)을 활용한 제어 파라미터 예측

* **데이터셋 구축:** 특정 초미세 상호작용(Hyperfine interaction) 파라미터인 A_parallel 및 A_perpendicular 값이 주어졌을 때, SHGO 알고리즘을 통해 탐색된 게이트 파라미터 데이터셋(약 46,135개 ~ 48,671개)을 구축했습니다.
* **모델 아키텍처:** 2개의 입력 노드(A_parallel, A_perpendicular)와 4개의 출력 노드(Xtau, XN, Ztau, ZN)를 가지며, ReLU 활성화 함수를 적용한 400개의 노드로 구성된 은닉층 6개를 설계했습니다.
* **학습 안정화:** 학습 과정에서 기울기에 따른 학습률 변동을 제어하기 위해 AdaBound Optimizer를 적용하여 안정성을 확보했습니다.

### 3.3. 강화학습(RL) 적용 시도

* **제어 궤적 탐색:** 머신러닝 예측 외에도, 상태 공간에서 최적의 제어 궤적을 찾기 위해 강화학습 개념을 실험적으로 도입했습니다.
* **보상 체계 모델링:** Snake AI 알고리즘의 보상 체계(거리, 목표물 도달 등)를 차용하여, Fidelity 및 Time Optimal 지표를 보상으로 설정하고 Rx, Ry, -Rx, -Ry 중 최적의 회전 방향을 탐색하도록 모델링했습니다.

## 4. 연구 결과 (Results)

* **연산 복잡도 획기적 감소:** 기존 알고리즘을 단독으로 사용하여 특정 A 파라미터에 대한 최적의 tau, N 값을 찾기 위해서는 20,000회에서 최대 135,000회 이상의 연산이 필요했습니다.
* **탐색 효율성 증대:** ANN 모델 기반의 예측 값 주변을 Local Optimizer로 순차 탐색하도록 구조를 변경한 결과, 탐색 연산 횟수를 최소 200회에서 평균 1,200회 수준으로 대폭 축소하는 데 성공했습니다.
* **성능 우위 입증:** 기존의 pi-pulse 방식과 비교했을 때, 제시된 방법론이 훨씬 적은 반복 횟수로도 더 높은 게이트 충실도를 일관되게 달성함을 확인했습니다.

## 5. 사용 기술 및 환경 (Tech Stack)

* **Language:** Python
* **Quantum Simulation & Linear Algebra:** QuTiP, NumPy, SciPy
* **Machine Learning / Deep Learning:** AdaBound Gradient Optimizer, ANN
* **Optimization Algorithms:** Nelder-Mead, Powell, SHGO, Differential Evolution, Dual Annealing
