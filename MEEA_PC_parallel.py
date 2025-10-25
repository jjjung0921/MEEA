# 필수 라이브러리 임포트
import pickle  # 데이터 직렬화를 위한 라이브러리
from multiprocessing import Process  # 병렬 처리를 위한 프로세스 생성
import multiprocessing  # 멀티프로세싱 관련 유틸리티
import torch  # PyTorch 딥러닝 프레임워크
import numpy as np  # 수치 계산용 NumPy
from valueEnsemble import ValueEnsemble  # 가치 함수 앙상블 모델
import signal  # 시그널 처리 (타임아웃 구현용)
import time  # 시간 측정
from contextlib import contextmanager  # 컨텍스트 매니저 데코레이터
from policyNet import MLPModel  # 정책 네트워크 모델
import os  # 운영체제 인터페이스
from rdkit import Chem  # 분자 구조 처리
from rdkit.Chem import AllChem  # 화학 정보학 알고리즘
import pandas as pd  # 데이터 프레임 처리

# 타임아웃 예외 정의
class TimeoutException(Exception): pass


@contextmanager
def time_limit(seconds):
    """
    지정된 시간(초) 내에 코드 블록을 실행하도록 제한하는 컨텍스트 매니저

    Args:
        seconds: 타임아웃 시간(초)

    Raises:
        TimeoutException: 지정된 시간 초과 시 발생
    """
    def signal_handler(signum, frame):
        raise TimeoutException
    signal.signal(signal.SIGALRM, signal_handler)  # SIGALRM 시그널 핸들러 등록
    signal.alarm(seconds)  # 알람 설정
    try:
        yield
    finally:
        signal.alarm(0)  # 알람 해제


def prepare_starting_molecules():
    """
    시작 분자(알려진 화합물) 세트를 준비하는 함수

    Returns:
        set: 사용 가능한 시작 분자들의 SMILES 문자열 집합
    """
    starting_mols = set(list(pd.read_csv('./prepare_data/origin_dict.csv')['mol']))
    return starting_mols

def smiles_to_fp(s, fp_dim=2048, pack=False):
    """
    SMILES 문자열을 Morgan Fingerprint로 변환

    Args:
        s: SMILES 문자열
        fp_dim: Fingerprint 차원 (기본값: 2048)
        pack: 비트 배열 압축 여부 (기본값: False)

    Returns:
        numpy.ndarray: 분자의 fingerprint 벡터
    """
    mol = Chem.MolFromSmiles(s)  # SMILES를 RDKit 분자 객체로 변환
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=fp_dim)  # Morgan FP 생성 (반지름 2)
    onbits = list(fp.GetOnBits())  # 활성화된 비트 인덱스 추출
    arr = np.zeros(fp.GetNumBits(), dtype=np.bool)  # 영벡터 생성
    arr[onbits] = 1  # 활성 비트 설정
    if pack:
        arr = np.packbits(arr)  # 압축이 필요한 경우 비트 패킹
    return arr


def batch_smiles_to_fp(s_list, fp_dim=2048):
    """
    여러 SMILES 문자열을 배치로 fingerprint로 변환

    Args:
        s_list: SMILES 문자열 리스트
        fp_dim: Fingerprint 차원 (기본값: 2048)

    Returns:
        numpy.ndarray: (len(s_list), fp_dim) 형태의 fingerprint 배열
    """
    fps = []
    for s in s_list:
        fps.append(smiles_to_fp(s, fp_dim))
    fps = np.array(fps)
    assert fps.shape[0] == len(s_list) and fps.shape[1] == fp_dim  # 형태 검증
    return fps


class MinMaxStats(object):
    """
    Min-Max 정규화를 위한 통계 클래스
    값의 범위를 추적하고 [0, 1] 범위로 정규화
    """
    def __init__(self, min_value_bound=None, max_value_bound=None):
        """
        Args:
            min_value_bound: 최소값 초기 경계
            max_value_bound: 최대값 초기 경계
        """
        self.maximum = min_value_bound if min_value_bound else -float('inf')
        self.minimum = max_value_bound if max_value_bound else float('inf')

    def update(self, value):
        """
        새로운 값으로 최대/최소 통계 업데이트

        Args:
            value: 업데이트할 값
        """
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value) -> float:
        """
        값을 [0, 1] 범위로 정규화

        Args:
            value: 정규화할 값

        Returns:
            float: 정규화된 값
        """
        if self.maximum > self.minimum:
            return (np.array(value) - self.minimum) / (self.maximum - self.minimum)
        return value


class Node:
    """
    MCTS(Monte Carlo Tree Search) 트리의 노드 클래스
    역합성 경로 탐색을 위한 탐색 트리의 각 상태를 나타냄
    """
    def __init__(self, state, h, prior, cost=0, action_mol=None, fmove=0, reaction=None, template=None, parent=None, cpuct=1.5):
        """
        Args:
            state: 현재 상태의 분자 리스트
            h: 휴리스틱 값 (추정 비용)
            prior: 사전 확률
            cost: 이 노드에 도달하기 위한 비용
            action_mol: 이 노드를 생성한 행동의 분자
            fmove: 형제 노드 중 몇 번째인지 (자식 인덱스)
            reaction: 이 노드를 생성한 반응
            template: 사용된 반응 템플릿
            parent: 부모 노드
            cpuct: UCT(Upper Confidence Bound for Trees) 탐색 상수
        """
        self.state = state  # 현재 상태 (분자 리스트)
        self.cost = cost  # 에지 비용
        self.h = h  # 휴리스틱 값 (목표까지의 추정 비용)
        self.prior = prior  # 사전 확률
        self.visited_time = 0  # 방문 횟수
        self.is_expanded = False  # 확장 여부
        self.template = template  # 반응 템플릿
        self.action_mol = action_mol  # 작용 분자
        self.fmove = fmove  # 부모의 자식 리스트에서의 인덱스
        self.reaction = reaction  # 반응 정보
        self.parent = parent  # 부모 노드
        self.cpuct = cpuct  # UCT 탐색 파라미터
        self.children = []  # 자식 노드 리스트
        self.child_illegal = np.array([])  # 자식이 불법인지 표시하는 배열

        # g(n): 시작점부터 현재 노드까지의 실제 비용
        if parent is not None:
            self.g = self.parent.g + cost  # 부모의 g 값에 현재 비용 추가
            self.parent.children.append(self)  # 부모의 자식 리스트에 추가
            self.depth = self.parent.depth + 1  # 깊이 증가
        else:
            self.g = 0  # 루트 노드
            self.depth = 0

        self.f = self.g + self.h  # f(n) = g(n) + h(n): A* 알고리즘의 평가 함수
        self.f_mean_path = []  # 경로상의 f 값들의 평균을 저장

    def child_N(self):
        """
        모든 자식 노드의 방문 횟수 반환

        Returns:
            numpy.ndarray: 자식 노드들의 방문 횟수 배열
        """
        N = [child.visited_time for child in self.children]
        return np.array(N)

    def child_p(self):
        """
        모든 자식 노드의 사전 확률 반환

        Returns:
            numpy.ndarray: 자식 노드들의 사전 확률 배열
        """
        prior = [child.prior for child in self.children]
        return np.array(prior)

    def child_U(self):
        """
        UCB(Upper Confidence Bound) 값 계산
        탐색(exploration)을 장려하는 보너스 항

        Returns:
            numpy.ndarray: 각 자식에 대한 UCB 값
        """
        child_Ns = self.child_N() + 1  # 0으로 나누기 방지
        prior = self.child_p()
        # UCT 공식: c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        child_Us = self.cpuct * np.sqrt(self.visited_time) * prior / child_Ns
        return child_Us

    def child_Q(self, min_max_stats):
        """
        자식 노드의 품질(Quality) 값 계산
        낮은 f 값(비용)이 높은 Q 값으로 변환됨

        Args:
            min_max_stats: 정규화를 위한 통계 객체

        Returns:
            numpy.ndarray: 각 자식의 품질 값
        """
        child_Qs = []
        for child in self.children:
            if len(child.f_mean_path) == 0:
                child_Qs.append(0.0)  # 방문하지 않은 노드
            else:
                # 정규화된 평균 비용을 품질 점수로 변환 (낮은 비용 = 높은 품질)
                child_Qs.append(1 - np.mean(min_max_stats.normalize(child.f_mean_path)))
        return np.array(child_Qs)

    def select_child(self, min_max_stats):
        """
        PUCT(Predictor + UCT) 알고리즘으로 최선의 자식 선택
        Q(s,a) + U(s,a) - illegal_penalty를 최대화

        Args:
            min_max_stats: 정규화를 위한 통계 객체

        Returns:
            int: 선택된 자식 노드의 인덱스
        """
        # 활용(exploitation) + 탐색(exploration) - 불법 페널티
        action_score = self.child_Q(min_max_stats) + self.child_U() - self.child_illegal
        best_move = np.argmax(action_score)
        return best_move


def prepare_value(model_f, gpu=None):
    """
    가치 함수 모델을 로드하고 준비

    Args:
        model_f: 모델 파일 경로
        gpu: GPU 번호 (-1이면 CPU 사용)

    Returns:
        ValueEnsemble: 로드된 가치 함수 모델
    """
    if gpu == -1:
        device = 'cpu'
    else:
        device = 'cuda:' + str(gpu)
    # 입력 차원: 2048, 은닉층 차원: 128, 드롭아웃: 0.1
    model = ValueEnsemble(2048, 128, 0.1).to(device)
    model.load_state_dict(torch.load(model_f, map_location=device))
    model.eval()  # 평가 모드로 설정
    return model

def prepare_expand(model_path, gpu=-1):
    """
    확장(expansion) 함수 모델 준비 - 역합성 반응 예측 모델

    Args:
        model_path: 정책 모델 경로
        gpu: GPU 번호 (-1이면 CPU 사용)

    Returns:
        MLPModel: 로드된 정책 네트워크 모델
    """
    if gpu == -1:
        device = 'cpu'
    else:
        device = 'cuda:' + str(gpu)
    # 템플릿 기반 역합성 예측 모델 로드
    one_step = MLPModel(model_path, './saved_model/template_rules.dat', device=device)
    return one_step



def value_fn(model, mols, device):
    """
    분자 리스트의 가치(합성 난이도)를 평가하는 함수

    Args:
        model: 가치 함수 모델
        mols: 평가할 분자들의 SMILES 리스트
        device: 연산 디바이스 (CPU 또는 CUDA)

    Returns:
        float: 예측된 가치 (낮을수록 합성이 쉬움)
    """
    num_mols = len(mols)
    fps = batch_smiles_to_fp(mols, fp_dim=2048).reshape(num_mols, -1)
    index = len(fps)

    # 최소 5개의 입력으로 패딩 (모델의 입력 형식에 맞춤)
    if len(fps) <= 5:
        mask = np.ones(5)
        mask[index:] = 0  # 패딩된 부분은 마스킹
        fps_input = np.zeros((5, 2048))
        fps_input[:index, :] = fps
    else:
        mask = np.ones(len(fps))
        fps_input = fps

    # 텐서로 변환하고 디바이스로 이동
    fps = torch.FloatTensor([fps_input.astype(np.float32)]).to(device)
    mask = torch.FloatTensor([mask.astype(np.float32)]).to(device)

    # 모델 예측 및 결과 반환
    v = model(fps, mask).cpu().data.numpy()
    return v[0][0]


class MCTS_A:
    """
    MCTS 기반 역합성 경로 계획 클래스
    목표 분자를 시작 물질로 분해하는 합성 경로를 탐색
    """
    def __init__(self, target_mol, known_mols, value_model, expand_fn, device, simulations, cpuct):
        """
        Args:
            target_mol: 목표 분자 (SMILES)
            known_mols: 알려진/구매 가능한 시작 물질 집합
            value_model: 가치 평가 모델
            expand_fn: 역합성 반응 예측 함수
            device: 연산 디바이스
            simulations: 시뮬레이션 횟수
            cpuct: UCT 탐색 상수
        """
        self.target_mol = target_mol  # 합성하고자 하는 목표 분자
        self.known_mols = known_mols  # 구매 가능한 시작 물질
        self.expand_fn = expand_fn  # 역합성 예측 모델
        self.value_model = value_model  # 가치 함수
        self.device = device  # CPU/GPU
        self.cpuct = cpuct  # UCT 탐색 파라미터

        # 루트 노드 생성 (목표 분자에서 시작)
        root_value = value_fn(self.value_model, [target_mol], self.device)
        self.root = Node([target_mol], root_value, prior=1.0, cpuct=self.cpuct)

        self.open = [self.root]  # 열린 노드 리스트
        self.visited_policy = {}  # 방문한 분자의 정책 캐시 (중복 계산 방지)
        self.visited_state = []  # 방문한 상태 리스트 (사이클 방지)
        self.min_max_stats = MinMaxStats()  # 정규화를 위한 통계
        self.min_max_stats.update(self.root.f)  # 루트의 f 값으로 초기화
        self.opening_size = simulations  # 한 번에 선택할 리프 노드 수
        self.iterations = 0  # 현재까지의 반복 횟수

    def select_a_leaf(self):
        """
        트리에서 리프 노드(확장되지 않은 노드)를 선택
        PUCT 알고리즘으로 최선의 경로를 따라 내려감

        Returns:
            Node: 선택된 리프 노드
        """
        current = self.root
        while True:
            current.visited_time += 1  # 방문 횟수 증가
            if not current.is_expanded:  # 확장되지 않은 노드면 반환
                return current
            # PUCT로 최선의 자식 선택하여 계속 탐색
            best_move = current.select_child(self.min_max_stats)
            current = current.children[best_move]

    def select(self):
        """
        여러 번의 리프 선택 중 가장 유망한(f 값이 낮은) 노드 선택

        Returns:
            Node: 확장할 최선의 리프 노드
        """
        # opening_size만큼 리프 노드를 샘플링
        openings = [self.select_a_leaf() for _ in range(self.opening_size)]
        # 그 중 f 값(비용)이 가장 낮은 노드 선택
        stats = [opening.f for opening in openings]
        index = np.argmin(stats)
        return openings[index]

    def expand(self, node):
        """
        노드를 확장: 역합성 반응을 예측하여 자식 노드 생성

        Args:
            node: 확장할 노드

        Returns:
            tuple: (성공 여부, 성공 시 생성된 리프 노드 또는 None)
        """
        node.is_expanded = True  # 확장 완료 표시
        expanded_mol_index = 0  # 상태의 첫 번째 분자를 확장
        expanded_mol = node.state[expanded_mol_index]

        # 이미 방문한 분자면 캐시된 정책 사용
        if expanded_mol in self.visited_policy.keys():
            expanded_policy = self.visited_policy[expanded_mol]
        else:
            # 역합성 모델로 가능한 반응 예측 (상위 50개)
            expanded_policy = self.expand_fn.run(expanded_mol, topk=50)
            self.iterations += 1  # 반복 횟수 증가
            # 정책 캐싱
            if expanded_policy is not None and (len(expanded_policy['scores']) > 0):
                self.visited_policy[expanded_mol] = expanded_policy.copy()
            else:
                self.visited_policy[expanded_mol] = None

        # 유효한 반응이 있는 경우
        if expanded_policy is not None and (len(expanded_policy['scores']) > 0):
            # 각 자식에 대한 불법 플래그 초기화
            node.child_illegal = np.array([0] * len(expanded_policy['scores']))

            # 각 예측된 반응에 대해 자식 노드 생성
            for i in range(len(expanded_policy['scores'])):
                # 반응물에서 알려진 분자들을 제거 (더 이상 분해할 필요 없음)
                reactant = [r for r in expanded_policy['reactants'][i].split('.') if r not in self.known_mols]
                # 현재 상태의 다른 분자들과 병합
                reactant = reactant + node.state[: expanded_mol_index] + node.state[expanded_mol_index + 1:]
                reactant = sorted(list(set(reactant)))  # 중복 제거 및 정렬

                # 반응 비용 계산 (음의 로그 확률)
                cost = - np.log(np.clip(expanded_policy['scores'][i], 1e-3, 1.0))
                template = expanded_policy['template'][i]
                reaction = expanded_policy['reactants'][i] + '>>' + expanded_mol

                # 균등한 사전 확률
                priors = np.array([1.0 / len(expanded_policy['scores'])] * len(expanded_policy['scores']))

                # 모든 반응물이 알려진 물질인 경우 (성공!)
                if len(reactant) == 0:
                    child = Node([], 0, cost=cost, prior=priors[i], action_mol=expanded_mol, reaction=reaction, fmove=len(node.children), template=template, parent=node, cpuct=self.cpuct)
                    return True, child  # 합성 경로 발견
                else:
                    # 아직 분해가 필요한 분자가 있는 경우
                    h = value_fn(self.value_model, reactant, self.device)  # 휴리스틱 값 계산
                    child = Node(reactant, h, cost=cost, prior=priors[i], action_mol=expanded_mol, reaction=reaction, fmove=len(node.children), template=template, parent=node, cpuct=self.cpuct)

                    # 이미 방문한 상태인지 확인 (사이클 방지)
                    if '.'.join(reactant) in self.visited_state:
                        node.child_illegal[child.fmove] = 1000  # 불법으로 표시
                        # 역전파: 모든 자식이 불법이면 부모도 불법으로 표시
                        back_check_node = node
                        while back_check_node.parent != None and np.all(back_check_node.child_illegal > 0):
                            back_check_node.parent.child_illegal[back_check_node.fmove] = 1000
                            back_check_node = back_check_node.parent
        else:
            # 유효한 반응이 없는 경우 (막다른 길)
            if node is not None and node.parent is not None:
                node.parent.child_illegal[node.fmove] = 1000  # 이 자식을 불법으로 표시
                # 역전파: 모든 자식이 불법이면 부모도 불법으로 표시
                back_check_node = node.parent
                while back_check_node != None and back_check_node.parent != None and np.all(back_check_node.child_illegal > 0):
                    back_check_node.parent.child_illegal[back_check_node.fmove] = 1000
                    back_check_node = back_check_node.parent
        return False, None  # 합성 경로를 찾지 못함

    def update(self, node):
        """
        노드의 통계 업데이트 및 역전파

        Args:
            node: 업데이트할 노드
        """
        stat = node.f  # 현재 노드의 f 값
        self.min_max_stats.update(stat)  # 전역 통계 업데이트

        # 루트까지 역전파하며 각 조상 노드의 경로 통계 업데이트
        current = node
        while current is not None:
            current.f_mean_path.append(stat)  # 이 경로의 f 값 기록
            current = current.parent

    def search(self, times):
        """
        MCTS 탐색 메인 루프

        Args:
            times: 최대 반복 횟수

        Returns:
            tuple: (성공 여부, 성공 시 최종 노드, 실제 반복 횟수)
        """
        success, node = False, None
        progress_interval = max(1, times // 10)  # 진행 상황 출력 주기

        # 반복 한도에 도달하지 않고, 아직 성공하지 못했고, 루트가 막히지 않았으면 계속 탐색
        while self.iterations < times and not success and (not np.all(self.root.child_illegal > 0) or len(self.root.child_illegal) == 0):
            expand_node = self.select()  # 확장할 노드 선택

            # 이미 방문한 상태인 경우 (사이클 방지)
            if '.'.join(expand_node.state) in self.visited_state:
                expand_node.parent.child_illegal[expand_node.fmove] = 1000
                # 역전파
                back_check_node = expand_node.parent
                while back_check_node != None and back_check_node.parent != None and np.all(back_check_node.child_illegal > 0):
                    back_check_node.parent.child_illegal[back_check_node.fmove] = 1000
                    back_check_node = back_check_node.parent
                continue
            else:
                # 새로운 상태 방문
                self.visited_state.append('.'.join(expand_node.state))
                success, node = self.expand(expand_node)  # 노드 확장
                self.update(expand_node)  # 통계 업데이트

            # 진행 상황 출력
            if (self.iterations % progress_interval == 0 or self.iterations == 1) and self.iterations <= times:
                print(f"[MCTS] target={self.target_mol} iterations={self.iterations}/{times}", flush=True)

            # 목표 분자를 확장할 수 없는 경우 (정책이 None)
            if self.visited_policy[self.target_mol] is None:
                return False, None, times

        return success, node, self.iterations

    def vis_synthetic_path(self, node):
        """
        성공 노드로부터 합성 경로 추출

        Args:
            node: 최종 노드 (알려진 물질로만 구성)

        Returns:
            tuple: (반응 경로 리스트, 템플릿 경로 리스트)
        """
        if node is None:
            return [], []

        reaction_path = []
        template_path = []
        current = node

        # 리프에서 루트까지 역추적
        while current is not None:
            reaction_path.append(current.reaction)
            template_path.append(current.template)
            current = current.parent

        # 루트에서 리프 방향으로 뒤집기 (정방향 합성 경로)
        return reaction_path[::-1], template_path[::-1]


def play(dataset, mols, thread, known_mols, value_model, expand_fn, device, simulations, cpuct, times):
    """
    여러 분자에 대해 역합성 경로 탐색을 실행하는 워커 함수
    (병렬 처리를 위한 개별 프로세스에서 실행됨)

    Args:
        dataset: 데이터셋 이름
        mols: 처리할 분자 리스트
        thread: 스레드 번호 (출력용)
        known_mols: 알려진 시작 물질
        value_model: 가치 함수 모델
        expand_fn: 확장 함수
        device: 연산 디바이스
        simulations: 시뮬레이션 횟수
        cpuct: UCT 파라미터
        times: 최대 반복 횟수
    """
    routes = []
    templates = []
    successes = []
    depths = []
    counts = []
    total = len(mols)

    for idx, mol in enumerate(mols, 1):
        print(f"[Thread {thread}] Start molecule {idx}/{total}: {mol}", flush=True)
        try:
            # 600초 타임아웃 설정
            with time_limit(600):
                player = MCTS_A(mol, known_mols, value_model, expand_fn, device, simulations, cpuct)
                success, node, count = player.search(times)
                route, template = player.vis_synthetic_path(node)
        except:
            # 타임아웃 또는 오류 발생 시
            success = False
            route = [None]
            template = [None]
            print(f"[Thread {thread}] Molecule {idx}/{total} timed out or failed", flush=True)

        # 결과 저장
        routes.append(route)
        templates.append(template)
        successes.append(success)

        if success:
            depths.append(node.depth)
            counts.append(count)
            print(f"[Thread {thread}] Completed molecule {idx}/{total} in {count} iterations (depth {node.depth})", flush=True)
        else:
            depths.append(32)  # 실패 시 최대 깊이로 표시
            counts.append(-1)
            print(f"[Thread {thread}] Failed molecule {idx}/{total}", flush=True)

    # 결과를 딕셔너리로 정리
    ans = {
        'route': routes,
        'template': templates,
        'success': successes,
        'depth': depths,
        'counts': counts
    }

    # 각 스레드의 결과를 별도 파일로 저장
    with open('./test/stat_norm_retro_' + dataset + '_' + str(simulations) + '_' + str(cpuct) + '_' + str(thread) + '.pkl', 'wb') as writer:
        pickle.dump(ans, writer, protocol=4)


def gather(dataset, simulations, cpuct, times, elapsed_time):
    """
    병렬 프로세스에서 생성된 결과 파일들을 수집하고 통합

    Args:
        dataset: 데이터셋 이름
        simulations: 시뮬레이션 횟수
        cpuct: UCT 파라미터
        times: 최대 반복 횟수
        elapsed_time: 실행에 소요된 시간(초)
    """
    result = {
        'route': [],
        'template': [],
        'success': [],
        'depth': [],
        'counts': []
    }

    # 28개의 워커 프로세스 결과 수집
    for i in range(28):
        file = './test/stat_norm_retro_' + dataset + '_' + str(simulations) + '_' + str(cpuct) + '_' + str(i) + '.pkl'
        with open(file, 'rb') as f:
            data = pickle.load(f)
        # 각 키의 데이터를 통합
        for key in result.keys():
            result[key] += data[key]
        os.remove(file)  # 개별 파일 삭제

    # 통계 계산
    success = np.mean(result['success'])  # 성공률
    depth = np.mean(result['depth'])  # 평균 깊이

    # 결과 요약을 텍스트 파일에 추가
    fr = open('result_simulation.txt', 'a')
    # Format: dataset, simulations, times, cpuct, success_rate, avg_depth, elapsed_time(s), elapsed_time(h:m:s)
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    fr.write(f"{dataset}\t{simulations}\t{times}\t{cpuct}\t{success:.4f}\t{depth:.2f}\t{elapsed_time:.2f}\t{time_str}\n")

    # 통합된 결과를 피클 파일로 저장
    f = open('./test/stat_pc_' + dataset + '_' + str(simulations) + '_' + str(cpuct) + '.pkl', 'wb')
    pickle.dump(result, f)
    f.close()


if __name__ == '__main__':
    """
    메인 실행 블록: 병렬 역합성 경로 탐색 실행
    """
    # 알려진 시작 분자 로드
    known_mols = prepare_starting_molecules()

    # MCTS 하이퍼파라미터
    simulations = 100  # 각 선택 단계에서 샘플링할 리프 노드 수
    cpuct = 4.0  # UCT 탐색 상수 (높을수록 탐색 지향적)

    # 멀티프로세싱 설정 ('spawn' 방식 사용)
    multiprocessing.set_start_method('spawn')

    # 모델 및 디바이스 준비
    one_steps = []  # 정책 모델 리스트
    devices = []  # 디바이스 리스트
    value_models = []  # 가치 모델 리스트

    # GPU 할당: GPU 4번만 사용 (28개 워커 모두 GPU 4번 사용)
    # CUDA_VISIBLE_DEVICES=4로 설정되어 있으므로 PyTorch는 디바이스 0번으로 인식
    gpus = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # 모델 파일 경로
    model_path = './saved_model/policy_model.ckpt'
    model_f = './saved_model/value_pc.pt'

    # 각 워커를 위한 모델 로드
    for i in range(len(gpus)):
        one_step = prepare_expand(model_path, gpus[i])  # 정책 모델
        device = torch.device('cuda:' + str(gpus[i]))
        value_model = prepare_value(model_f, gpus[i])  # 가치 모델
        value_models.append(value_model)
        one_steps.append(one_step)
        devices.append(device)

    # 테스트할 데이터셋 리스트
    dataset = ['USPTO', 'ClinTox', 'logS', 'Toxicity_LD50', 'BBBP', 'Ames', 'logP', 'bace', 'DPP4', 'SVS']

    # 각 데이터셋에 대해 실행
    for data in dataset:
        fileName = ('./test_dataset/' + data + '.pkl')
        with open(fileName, 'rb') as f:
            targets = pickle.load(f)  # 목표 분자 리스트 로드

        # 분자들을 28개 워커에 분배
        intervals = int(len(targets) / len(gpus))  # 기본 간격
        num_more = len(targets) - intervals * len(gpus)  # 나머지 분자 수

        # 최대 반복 횟수
        for times in [500]:
            start_time = time.time()
            jobs = []

            # 나머지가 있는 워커들은 1개씩 더 많은 분자를 처리
            jobs = [Process(target=play, args=(data, targets[i * (intervals + 1): (i + 1) * (intervals + 1)], i, known_mols, value_models[i], one_steps[i], devices[i], simulations, cpuct, times)) for i in range(num_more)]

            # 나머지 워커들은 기본 간격만큼 처리
            start = num_more * (intervals + 1)
            for i in range(len(gpus) - num_more):
                jobs.append(Process(target=play, args=(data, targets[start + i * intervals: start + (i + 1) * intervals], num_more + i, known_mols, value_models[num_more + i], one_steps[num_more + i], devices[num_more + i], simulations, cpuct, times)))

            # 모든 프로세스 시작
            for j in jobs:
                j.start()

            # 모든 프로세스가 완료될 때까지 대기
            for j in jobs:
                j.join()

            elapsed_time = time.time() - start_time

            # 결과 수집 및 통합
            gather(data, simulations, cpuct, times, elapsed_time)
