"""
policyNet.py
============
정책 네트워크 모듈 - 역합성 반응 예측 모델

역할:
- 목표 분자에 적용할 수 있는 역합성 템플릿을 예측
- 템플릿 기반 역합성 접근법 사용
- MCTS의 expand 함수에서 사용됨

주요 컴포넌트:
1. RolloutPolicyNet: 템플릿 분류 신경망
2. MLPModel: 추론을 위한 래퍼 클래스
3. 학습 및 평가 함수들
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange
from time import strftime, localtime
import numpy as np
import gzip
import pickle
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, rdFingerprintGenerator
from collections import defaultdict, OrderedDict
from rdchiral.main import rdchiralRunText, rdchiralRun  # 키랄성을 고려한 반응 적용

# RDKit 경고 메시지 비활성화
RDLogger.DisableLog('rdApp.*')


def preprocess(X, fp_dim):
    """
    SMILES 문자열을 Morgan Fingerprint로 변환

    Args:
        X: SMILES 문자열
        fp_dim: Fingerprint 차원

    Returns:
        numpy.ndarray: 이진 fingerprint 배열
    """
    mol = Chem.MolFromSmiles(X)
    # 키랄성을 고려한 Morgan Fingerprint 생성 (반지름 2) - 최신 API 사용
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=int(fp_dim), includeChirality=True)
    fp = generator.GetFingerprint(mol)
    onbits = list(fp.GetOnBits())
    arr = np.zeros(fp.GetNumBits())
    arr[onbits] = 1
    return arr


def merge(reactant_d):
    """
    중복된 반응물을 병합하고 점수를 합산

    Args:
        reactant_d: {반응물: [(점수, 템플릿), ...]} 딕셔너리

    Returns:
        tuple: (반응물 리스트, 점수 리스트, 템플릿 리스트)
               점수 기준 내림차순 정렬
    """
    ret = []
    for reactant, l in reactant_d.items():
        ss, ts = zip(*l)  # 점수와 템플릿 분리
        ret.append((reactant, sum(ss), list(ts)[0]))  # 점수 합산
    # 점수 기준 내림차순 정렬
    reactants, scores, templates = zip(*sorted(ret, key=lambda item: item[1], reverse=True))
    return list(reactants), list(scores), list(templates)


class RolloutPolicyNet(nn.Module):
    """
    역합성 템플릿 분류를 위한 신경망

    구조:
    - 입력: 분자 fingerprint (2048차원)
    - 은닉층: 512차원 (배치 정규화 + 드롭아웃)
    - 출력: 템플릿 개수만큼의 로짓 (보통 수만 개)
    """
    def __init__(self, n_rules, fp_dim=2048, dim=512, dropout_rate=0.3):
        """
        Args:
            n_rules: 반응 템플릿 개수 (분류 클래스 수)
            fp_dim: Fingerprint 차원
            dim: 은닉층 차원
            dropout_rate: 드롭아웃 비율
        """
        super(RolloutPolicyNet, self).__init__()
        self.fp_dim = fp_dim
        self.n_rules = n_rules
        self.dropout_rate = dropout_rate

        # 2층 MLP: fingerprint -> 은닉층 -> 템플릿 로짓
        self.fc1 = nn.Linear(fp_dim, dim)  # 첫 번째 선형층
        self.bn1 = nn.BatchNorm1d(dim)  # 배치 정규화 (학습 안정화)
        self.dropout1 = nn.Dropout(dropout_rate)  # 과적합 방지
        self.fc3 = nn.Linear(dim, n_rules)  # 출력층 (템플릿 분류)

    def forward(self, x, y=None, loss_fn=nn.CrossEntropyLoss()):
        """
        순전파

        Args:
            x: 입력 fingerprint (batch_size, fp_dim)
            y: 정답 레이블 (학습 시에만 사용)
            loss_fn: 손실 함수

        Returns:
            학습 시: 손실 값
            추론 시: 로짓 (batch_size, n_rules)
        """
        # fingerprint -> 은닉층 -> BN -> 드롭아웃 -> 출력층
        x = self.fc3(self.dropout1(self.bn1(self.fc1(x))))

        if y is not None:
            # 학습 모드: 손실 반환
            return loss_fn(x, y)
        else:
            # 추론 모드: 로짓 반환
            return x


def load_parallel_model(state_path, template_rule_path, fp_dim=2048):
    """
    학습된 정책 모델과 템플릿 규칙을 로드

    Args:
        state_path: 모델 체크포인트 파일 경로
        template_rule_path: 템플릿 규칙 파일 경로 (텍스트 파일)
        fp_dim: Fingerprint 차원

    Returns:
        tuple: (로드된 모델, 인덱스->템플릿 딕셔너리)
    """
    # 템플릿 규칙 파일 로드 (각 줄이 하나의 템플릿)
    template_rules = {}
    with open(template_rule_path, 'r') as f:
        for i, l in tqdm(enumerate(f), desc='template rules'):
            rule = l.strip()
            template_rules[rule] = i  # 템플릿 -> 인덱스 매핑

    # 인덱스 -> 템플릿 역매핑 생성
    idx2rule = {}
    for rule, idx in template_rules.items():
        idx2rule[idx] = rule

    # 모델 생성 및 가중치 로드
    rollout = RolloutPolicyNet(len(template_rules), fp_dim=fp_dim)
    checkpoint = torch.load(state_path, map_location='cpu')
    # DataParallel로 학습된 경우 'module.' 접두사를 제거
    if any(k.startswith('module.') for k in checkpoint.keys()):
        checkpoint = {k.replace('module.', '', 1): v for k, v in checkpoint.items()}
    rollout.load_state_dict(checkpoint)

    # 주석 처리된 코드: DataParallel로 학습된 모델을 로드할 때 사용
    #new_state_dict = OrderedDict()
    #for k, v in checkpoint.items():
    #    name = k[7:]  # 'module.' 접두사 제거
    #    new_state_dict[name] = v
    #rollout.load_state_dict(new_state_dict)

    return rollout, idx2rule


class MLPModel(object):
    """
    역합성 예측을 위한 추론 래퍼 클래스
    MCTS의 expand 함수에서 사용됨

    동작 과정:
    1. 분자 fingerprint 생성
    2. 신경망으로 템플릿 확률 예측
    3. Top-k 템플릿 선택
    4. 각 템플릿을 분자에 적용하여 반응물 생성
    5. 결과 정규화 및 반환
    """
    def __init__(self, state_path, template_path, device=-1, fp_dim=2048):
        """
        Args:
            state_path: 모델 체크포인트 경로
            template_path: 템플릿 규칙 파일 경로
            device: 연산 디바이스
            fp_dim: Fingerprint 차원
        """
        super(MLPModel, self).__init__()
        self.fp_dim = fp_dim
        # 모델과 템플릿 로드
        self.net, self.idx2rules = load_parallel_model(state_path, template_path, fp_dim)
        self.net.eval()  # 평가 모드로 설정
        self.device = device
        self.net.to(device)

    def run(self, x, topk=10):
        """
        분자에 대한 역합성 예측 실행

        Args:
            x: 목표 분자 SMILES 문자열
            topk: 상위 k개 템플릿 선택

        Returns:
            dict 또는 None:
                - 'reactants': 반응물 SMILES 리스트
                - 'scores': 각 반응물의 점수 (정규화됨)
                - 'template': 사용된 템플릿 리스트
                반응물이 없으면 None 반환
        """
        # 1. Fingerprint 생성 및 전처리
        arr = preprocess(x, self.fp_dim)
        arr = np.reshape(arr, [-1, arr.shape[0]])  # (1, fp_dim) 형태로 변환
        arr = torch.tensor(arr, dtype=torch.float32)
        arr = arr.to(self.device)

        # 2. 신경망으로 템플릿 확률 예측
        preds = self.net(arr)
        preds = F.softmax(preds, dim=1)  # 확률로 변환
        preds = preds.cpu()

        # 3. Top-k 템플릿 선택
        probs, idx = torch.topk(preds, k=topk)
        rule_k = [self.idx2rules[id] for id in idx[0].numpy().tolist()]

        # 4. 각 템플릿을 분자에 적용
        reactants = []
        scores = []
        templates = []
        for i, rule in enumerate(rule_k):
            try:
                # rdchiral로 템플릿 적용 (키랄성 보존)
                out1 = rdchiralRunText(rule, x)
                if len(out1) == 0:  # 적용 불가능한 템플릿
                    continue
                out1 = sorted(out1)
                # 한 템플릿이 여러 반응물을 생성할 수 있음
                for reactant in out1:
                    reactants.append(reactant)
                    scores.append(probs[0][i].item()/len(out1))  # 확률을 균등 분배
                    templates.append(rule)
            except (ValueError, RuntimeError, KeyError, IndexError) as e:
                pass  # 오류 발생 시 스킵

        if len(reactants) == 0:
            return None  # 유효한 반응물이 없음

        # 5. 중복 반응물 병합 (다른 템플릿이 같은 반응물을 생성할 수 있음)
        reactants_d = defaultdict(list)
        for r, s, t in zip(reactants, scores, templates):
            if '.' in r:
                # 여러 반응물이 있는 경우 정렬하여 정규화
                str_list = sorted(r.strip().split('.'))
                reactants_d['.'.join(str_list)].append((s, t))
            else:
                reactants_d[r].append((s, t))

        # 6. 병합 및 정규화
        reactants, scores, templates = merge(reactants_d)
        total = sum(scores)
        scores = [s / total for s in scores]  # 확률 합이 1이 되도록 정규화

        return {'reactants': reactants,
                'scores': scores,
                'template': templates}


def top_k_acc(preds, gt, k=1):
    """
    Top-k 정확도 계산 (평가용)

    Args:
        preds: 모델 예측 로짓 (batch_size, n_classes)
        gt: 정답 레이블 (batch_size,)
        k: 상위 k개 고려

    Returns:
        tuple: (정답 개수, 전체 개수)
    """
    probs, idx = torch.topk(preds, k=k)
    idx = idx.cpu().numpy().tolist()
    gt = gt.cpu().numpy().tolist()
    num = preds.size(0)
    correct = 0
    for i in range(num):
        if gt[i] in idx[i]:  # 정답이 top-k에 있는지 확인
            correct += 1
    return correct, num


class OneStepDataset(Dataset):
    """
    역합성 학습용 데이터셋
    (분자 fingerprint, 템플릿 레이블) 쌍을 제공
    """
    def __init__(self, data):
        """
        Args:
            data: {'smiles': 압축된 fingerprint 배열, 'template': 레이블 배열}
        """
        self.x = data['smiles']
        self.y = data['template']

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        """
        Args:
            item: 인덱스

        Returns:
            tuple: (fingerprint, 템플릿 레이블)
        """
        x_fp = np.unpackbits(self.x[item])  # 압축된 비트 배열 해제
        return x_fp, self.y[item]


def dataset_iterator(data, batch_size=1024, shuffle=True, num_workers=4):
    """
    데이터 로더 생성

    Args:
        data: 데이터 딕셔너리
        batch_size: 배치 크기
        shuffle: 셔플 여부
        num_workers: 워커 프로세스 수

    Returns:
        DataLoader: PyTorch 데이터 로더
    """
    dataset = OneStepDataset(data)

    def collate_fn(batch):
        """배치 데이터를 텐서로 변환"""
        X, y = zip(*batch)
        X = np.array(X)
        y = np.array(y)
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.int64)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)


def train_one_epoch(net, train_loader, optimizer, it, device):
    """
    1 에포크 학습

    Args:
        net: 신경망 모델
        train_loader: 학습 데이터 로더
        optimizer: 옵티마이저
        it: tqdm 진행 표시줄
        device: 연산 디바이스

    Returns:
        list: 배치별 손실 값 리스트
    """
    losses = []
    net.train()  # 학습 모드
    fr = open('loss.txt', 'a')  # 손실 기록 파일

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # 순전파 + 역전파 + 최적화
        optimizer.zero_grad()
        loss_v = net(X_batch, y_batch)  # forward에서 손실 계산
        loss_v = loss_v.mean()
        loss_v.backward()
        nn.utils.clip_grad_norm_(net.parameters(), max_norm=5)  # 그래디언트 클리핑
        optimizer.step()

        # 손실 기록
        losses.append(loss_v.item())
        fr.write(str(loss_v.item()) + '\n')
        it.set_postfix(loss=np.mean(losses[-10:]) if losses else None)  # 최근 10개 평균

    fr.close()
    return losses


def eval_one_epoch(net, val_loader, device):
    """
    1 에포크 평가 (검증 세트)

    Args:
        net: 신경망 모델
        val_loader: 검증 데이터 로더
        device: 연산 디바이스

    Returns:
        tuple: (top-1 정확도, top-10 정확도, top-50 정확도, 평균 손실)
    """
    net.eval()  # 평가 모드
    eval_top1_correct, eval_top1_num = 0, 0
    eval_top10_correct, eval_top10_num = 0, 0
    eval_top50_correct, eval_top50_num = 0, 0
    loss = 0.0

    for X_batch, y_batch in tqdm(val_loader):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        with torch.no_grad():  # 그래디언트 계산 비활성화
            y_hat = net(X_batch)
            loss += F.cross_entropy(y_hat, y_batch).item()

            # Top-k 정확도 계산
            top_1_correct, num1 = top_k_acc(y_hat, y_batch, k=1)
            top_10_correct, num10 = top_k_acc(y_hat, y_batch, k=10)
            top_50_correct, num50 = top_k_acc(y_hat, y_batch, k=50)

            eval_top1_correct += top_1_correct
            eval_top1_num += num1
            eval_top10_correct += top_10_correct
            eval_top10_num += num10
            eval_top50_correct += top_50_correct
            eval_top50_num += num50

    # 정확도 및 손실 계산
    val_1 = eval_top1_correct/eval_top1_num
    val_10 = eval_top10_correct/eval_top10_num
    val_50 = eval_top50_correct/eval_top50_num
    loss = loss / (len(val_loader.dataset))

    return val_1, val_10, val_50, loss


def train(net, dataTrain, dataTest, lr=0.001, batch_size=16, epochs=100, wd=0, saved_model='./model/saved_states'):
    """
    전체 학습 루프

    Args:
        net: 신경망 모델
        dataTrain: 학습 데이터
        dataTest: 검증 데이터
        lr: 학습률
        batch_size: 배치 크기
        epochs: 에포크 수
        wd: 가중치 감쇠 (L2 정규화)
        saved_model: 모델 저장 경로 접두사
    """
    it = trange(epochs)
    # 가용 GPU를 사용해 병렬 학습 시도, 없으면 CPU 사용
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)  # 모든 가용 GPU 자동 사용
    else:
        device = torch.device('cpu')
    net = net.to(device)

    # 옵티마이저 및 스케줄러 설정
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-6)  # 손실이 개선되지 않으면 학습률 감소

    # 데이터 로더 생성
    train_loader = dataset_iterator(dataTrain, batch_size=batch_size)
    val_loader = dataset_iterator(dataTest, batch_size=batch_size, shuffle=False)

    best = -1  # 최고 top-1 정확도

    for e in it:
        # 학습 및 평가
        train_one_epoch(net, train_loader, optimizer, it, device)
        val_1, val_10, val_50, loss = eval_one_epoch(net, val_loader, device)
        scheduler.step(loss)

        # 최고 성능 모델 저장
        if best < val_1:
            best = val_1
            state = net.state_dict()
            time_stamp = strftime("%Y-%m-%d_%H:%M:%S", localtime())
            save_path = saved_model + "_" + time_stamp + '.ckpt'
            torch.save(state, save_path)

        # 결과 기록
        line = "\nTop 1: {}  ==> Top 10: {} ==> Top 50: {}, validation loss ==> {}".format(val_1, val_10, val_50, loss)
        fr = open('result.txt', 'a')
        fr.write(line)
        fr.close()
        print(line)


def train_mlp(batch_size=1024, lr=0.001, epochs=100, weight_decay=0, dropout_rate=0.3, saved_model='./model/saved_rollout_state'):
    """
    정책 네트워크 학습 메인 함수

    Args:
        batch_size: 배치 크기
        lr: 학습률
        epochs: 에포크 수
        weight_decay: 가중치 감쇠
        dropout_rate: 드롭아웃 비율
        saved_model: 모델 저장 경로 접두사
    """
    # 템플릿 로드
    with gzip.open('./prepare_data/uspto_template.pkl.gz', 'rb') as f:
        templates = pickle.load(f)
    num_of_rules = len(templates)

    # 모델 생성
    rollout = RolloutPolicyNet(n_rules=num_of_rules, dropout_rate=dropout_rate)
    print('mlp model training...')

    # 학습/검증 데이터 로드
    train_path = './prepare_data/policyTrain.pkl.gz'
    with gzip.open(train_path, 'rb') as f:
        trainData = pickle.load(f)

    test_path = './prepare_data/policyTest.pkl.gz'
    with gzip.open(test_path, 'rb') as f:
        testData = pickle.load(f)

    print('Training size:', len(trainData['smiles']))

    # 학습 실행
    train(rollout, dataTrain=trainData, dataTest=testData, batch_size=batch_size, lr=lr, epochs=epochs, wd=weight_decay, saved_model=saved_model)


def loadPolicyModel():
    """
    학습된 정책 모델 로드 (예제 함수)

    Returns:
        RolloutPolicyNet: 로드된 모델
    """
    with gzip.open('./prepare_data/uspto_template.pkl.gz', 'rb') as f:
        templates = pickle.load(f)
    num_of_rules = len(templates)

    rollout = RolloutPolicyNet(n_rules=num_of_rules, dropout_rate=0.4)
    rollout.load_state_dict(torch.load('./model/saved_rollout_state_2023-01-09_04:38:49.ckpt'))
    rollout.eval()
    return rollout


if __name__ == '__main__':
    """
    커맨드라인에서 실행 시 정책 네트워크 학습
    예: python policyNet.py --batch_size 2048 --learning_rate 0.001
    """
    import argparse
    parser = argparse.ArgumentParser(description="Policies for retrosynthesis Planner")
    parser.add_argument('--model_folder', default='./model', type=str, help='specify where to save the trained models')
    parser.add_argument('--batch_size', default=2056, type=int, help="specify the batch size")
    parser.add_argument('--dropout_rate', default=0.4, type=float, help="specify the dropout rate")
    parser.add_argument('--learning_rate', default=0.01, type=float, help="specify the learning rate")
    args = parser.parse_args()

    model_folder = args.model_folder
    batch_size = args.batch_size
    dropout_rate = args.dropout_rate
    lr = args.learning_rate

    # 모델 폴더 생성
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)

    # 학습 시작
    train_mlp(batch_size=batch_size, lr=lr, dropout_rate=dropout_rate)
