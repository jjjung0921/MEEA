"""
valueEnsemble.py
================
가치 함수 앙상블 모듈 - 분자 집합의 합성 난이도 평가

역할:
- 여러 분자들의 합성 난이도(비용)를 예측
- MCTS의 휴리스틱 함수로 사용됨
- 일관성 학습(Consistency Learning)으로 훈련

주요 개념:
- Fitting Loss: 분자 집합의 실제 합성 비용과 예측값 차이
- Consistency Loss: Bellman 방정식 기반 일관성 제약
  V(target) ≈ V(reactants) + cost(reaction)
"""

import os
import numpy as np
import torch
import pickle
import torch.nn as nn
from GraphEncoder import GraphModel
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def unpack_fps(packed_fps):
    """
    압축된 fingerprint 배열을 해제

    Args:
        packed_fps: 압축된 비트 배열

    Returns:
        numpy.ndarray: 해제된 float32 배열
    """
    packed_fps = np.array(packed_fps)

    # 이미 2048차원으로 풀린 경우 그대로 반환
    if packed_fps.shape[-1] == 2048:
        return packed_fps.astype(np.float32)

    # packbits 형태(길이 256 등)만 언팩
    packed_fps = packed_fps.astype(np.uint8)
    shape = (*(packed_fps.shape[:-1]), -1)
    fps = np.unpackbits(packed_fps.reshape((-1, packed_fps.shape[-1])), axis=-1).astype(np.float32).reshape(shape)
    return fps


class ValueEnsemble(nn.Module):
    """
    가치 함수 앙상블 모델

    구조:
    1. GraphModel: 여러 분자의 fingerprint를 집계
    2. Linear: 집계된 벡터를 스칼라 가치로 변환

    입력: (batch_size, num_molecules, fp_dim) + 마스크
    출력: (batch_size, 1) - 합성 난이도 예측값
    """
    def __init__(self, fp_dim, latent_dim, dropout_rate):
        """
        Args:
            fp_dim: Fingerprint 차원 (2048)
            latent_dim: 잠재 공간 차원 (128)
            dropout_rate: 드롭아웃 비율 (사용되지 않음, 호환성 유지용)
        """
        super(ValueEnsemble, self).__init__()
        self.fp_dim = fp_dim
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate

        # 그래프 신경망으로 분자 집합 인코딩
        self.graphNN = GraphModel(input_dim=fp_dim, feature_dim=latent_dim, hidden_dim=latent_dim)

        # 선형층으로 가치 예측 (바이어스 없음)
        self.layers = nn.Linear(latent_dim, 1, bias=False)

    def forward(self, fps, mask):
        """
        순전파

        Args:
            fps: 분자 fingerprint (batch_size, num_molecules, fp_dim)
            mask: 유효한 분자 마스크 (batch_size, num_molecules)

        Returns:
            torch.Tensor: 예측된 가치 (batch_size, 1)
        """
        x = self.graphNN(fps, mask)  # 분자 집합 인코딩
        x = self.layers(x)  # 가치 예측
        return x


class ConsistencyDataset(Dataset):
    """
    일관성 학습용 데이터셋

    데이터 구조:
    - reaction_costs: 반응 비용 (스칼라)
    - target_values: 목표 분자의 실제 가치
    - reactant_fps: 반응물 fingerprint들
    - reactant_masks: 유효한 반응물 마스크

    일관성 조건: V(target) ≈ V(reactants) + cost
    """
    def __init__(self, data):
        """
        Args:
            data: 일관성 학습 데이터 딕셔너리
        """
        self.reaction_costs = data['reaction_costs']
        self.target_values = data['target_values']
        self.reactant_fps = data['reactant_fps']
        self.reactant_masks = data['reactant_masks']

    def __len__(self):
        return len(self.reaction_costs)

    def __getitem__(self, item):
        """
        Args:
            item: 인덱스

        Returns:
            tuple: (반응 비용, 목표 가치, 반응물 fps, 반응물 마스크)
        """
        reaction_cost = self.reaction_costs[item]
        target_value = self.target_values[item]

        # 최대 5개 분자까지 패딩 (실제로는 최대 3개 사용)
        reactant_fps = np.zeros((5, 2048), dtype=np.float32)
        reactant_fps[:3, :] = unpack_fps(np.array(self.reactant_fps[item]))

        reactant_masks = np.zeros(5, dtype=np.float32)
        reactant_masks[:3] = self.reactant_masks[item]

        return reaction_cost, target_value, reactant_fps, reactant_masks


class FittingDatasetTest(Dataset):
    """
    Fitting 학습/평가용 데이터셋

    데이터 구조:
    - fps: 분자 집합의 fingerprint
    - values: 실제 합성 비용
    - masks: 유효한 분자 마스크
    """
    def __init__(self, data):
        """
        Args:
            data: fitting 데이터 딕셔너리
        """
        self.fps = data['fps']
        self.values = data['values']
        self.masks = data['masks']

    def __len__(self):
        return len(self.values)

    def __getitem__(self, item):
        """
        Args:
            item: 인덱스

        Returns:
            tuple: (fps, 가치, 마스크)
        """
        return self.fps[item], self.values[item], self.masks[item]



class Trainer:
    """
    가치 함수 학습 트레이너

    학습 방식:
    1. Consistency Loss: 반응 전후의 가치 일관성 유지
       Loss = max(0, -V(reactants) - cost + V(target) + 7)^2
    2. Fitting Loss: 실제 합성 비용과의 차이 최소화
       Loss = MSE(V_pred, V_actual)

    각 배치마다 두 손실을 동시에 최적화
    """
    def __init__(self, model, n_epochs, lr, batch_size, model_folder, device, test_epoch, log_interval=10):
        """
        Args:
            model: ValueEnsemble 모델
            n_epochs: 에포크 수
            lr: 학습률
            batch_size: 배치 크기
            model_folder: 모델 저장 폴더
            device: 연산 디바이스
            test_epoch: 평가 주기
            log_interval: 중간 로그 출력 주기(에포크 기준)
        """
        self.batch_size = batch_size
        self.log_interval = log_interval

        # 학습 데이터 로드 - 일관성 학습용
        file = './data/train_consistency.pkl'
        with open(file, 'rb') as f:
            self.train_consistency_data = pickle.load(f)
        self.train_consistency = ConsistencyDataset(self.train_consistency_data)
        self.train_consistency_loader = DataLoader(self.train_consistency, batch_size=self.batch_size, shuffle=True)
        self.train_consistency_iter = iter(self.train_consistency_loader)

        # 학습 데이터 로드 - Fitting용
        file = './data/train_fitting.pkl'
        with open(file, 'rb') as f:
            self.train_fitting_data = pickle.load(f)
        self.num_fitting_mols = len(self.train_fitting_data['values'])
        print('Train Data Loaded')

        # 검증 데이터 로드 - 일관성 평가용
        file = './data/val_consistency.pkl'
        with open(file, 'rb') as f:
            data = pickle.load(f)
        self.val_consistency = ConsistencyDataset(data)
        self.val_consistency_loader = DataLoader(self.val_consistency, batch_size=self.batch_size, shuffle=False)

        # 검증 데이터 로드 - Fitting 평가용
        file = './val_dataset/val_fitting.pkl'
        with open(file, 'rb') as f:
            data = pickle.load(f)
        self.val_fitting = FittingDatasetTest(data)
        self.val_fitting_loader = DataLoader(self.val_fitting, batch_size=self.batch_size, shuffle=False)
        print('Validation Data Loaded.')

        # 학습 파라미터
        self.n_epochs = n_epochs
        self.lr = lr
        self.model_folder = model_folder
        self.device = device
        self.model = model.to(device)
        self.test_epoch = test_epoch
        self.fitting_criterion = nn.MSELoss(reduction='none')  # 샘플별 손실 계산
        self.num_mols = 1  # 현재 배치의 분자 수

        # 모델 폴더 생성
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        # 옵티마이저
        self.optim = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)

    def sample(self, num_mols, num_samples):
        """
        Fitting 학습용 배치 샘플링

        무작위로 num_mols개의 분자를 선택하여 집합 생성
        가치는 개별 분자 가치의 합으로 계산

        Args:
            num_mols: 샘플링할 분자 개수 (1~5)
            num_samples: 생성할 샘플 수 (배치 크기)

        Returns:
            dict: fitting 데이터 배치
        """
        fps = np.zeros((num_samples, 5, 2048), dtype=np.float32)
        values = []
        masks = np.ones((num_samples, 5))
        masks[:, num_mols:] = 0  # num_mols개만 유효

        for n in range(num_samples):
            # 무작위로 num_mols개 분자 선택 (중복 없이)
            index = np.random.choice(self.num_fitting_mols, num_mols, replace=False)
            fp = [self.train_fitting_data['fps'][i] for i in index]
            value = np.sum([self.train_fitting_data['values'][i] for i in index])  # 가치 합산
            fps[n, :num_mols, :] = unpack_fps(np.array(fp))
            values.append(value)

        data = {
            'fps': list(fps.astype(np.float32)),
            'values': list(np.array(values).astype(np.float32)),
            'masks': list(masks.astype(np.float32))
        }
        return data

    def _pass(self, consistency_data, fitting_data):
        """
        학습 1회 패스 (순전파 + 역전파)

        두 가지 손실을 동시에 최적화:
        1. Fitting Loss: 분자 집합의 예측 가치와 실제 가치 차이
        2. Consistency Loss: Bellman 방정식 기반 일관성 제약

        Args:
            consistency_data: 일관성 학습 배치
            fitting_data: Fitting 학습 배치
        """
        self.optim.zero_grad()

        # === Fitting Loss 계산 ===
        fps, values, masks = fitting_data['fps'], fitting_data['values'], fitting_data['masks']
        fps = torch.tensor(np.asarray(fps, dtype=np.float32), device=self.device)
        values = torch.tensor(np.asarray(values, dtype=np.float32), device=self.device).reshape(-1)
        masks = torch.tensor(np.asarray(masks, dtype=np.float32), device=self.device)

        # 분자 개수에 따른 가중치 (적은 분자 = 높은 가중치)
        weight = torch.tensor(np.array([1 / self.num_mols] * len(values), dtype=np.float32), device=self.device)

        v_pred = self.model(fps, masks)
        fitting_loss = self.fitting_criterion(v_pred.reshape(-1), values)
        fitting_loss = (weight * fitting_loss).mean()  # 가중 평균

        # === Consistency Loss 계산 ===
        reaction_costs, target_values, reactant_fps, reactant_masks = consistency_data
        reaction_costs = torch.as_tensor(reaction_costs, dtype=torch.float32, device=self.device).reshape(-1)
        target_values = torch.as_tensor(target_values, dtype=torch.float32, device=self.device).reshape(-1)
        reactant_fps = torch.as_tensor(reactant_fps, dtype=torch.float32, device=self.device)
        reactant_masks = torch.as_tensor(reactant_masks, dtype=torch.float32, device=self.device)

        r_values = self.model(reactant_fps, reactant_masks)

        # Bellman 잔차: V(target) - V(reactants) - cost
        # 이상적으로는 0이어야 함. 여기에 7을 더해 양수로 만들고 hinge loss 적용
        r_gap = - r_values - reaction_costs + target_values
        r_gap = torch.clamp(r_gap, min=0)  # Hinge loss
        consistency_loss = (r_gap ** 2).mean()

        # === 전체 손실 및 역전파 ===
        loss = fitting_loss + consistency_loss
        loss.backward()
        self.optim.step()

        # 손실 기록
        fr = open('loss.txt', 'a')
        line = str(loss.item()) + '\t' + str(fitting_loss.item()) + '\t' + str(consistency_loss.item()) + '\n'
        fr.write(line)
        fr.close()
        return loss.item(), fitting_loss.item(), consistency_loss.item()

    def eval(self):
        """
        검증 세트에서 평가

        두 가지 손실 모두 평가:
        1. Consistency Loss
        2. Fitting Loss
        """
        self.model.eval()
        consistency_loss = []
        fitting_loss = []

        # Consistency Loss 평가
        for batch in self.val_consistency_loader:
            reaction_costs, target_values, reactant_fps, reactant_masks = batch
            reaction_costs = torch.as_tensor(reaction_costs, dtype=torch.float32, device=self.device).reshape(-1)
            target_values = torch.as_tensor(target_values, dtype=torch.float32, device=self.device).reshape(-1)
            reactant_fps = torch.as_tensor(reactant_fps, dtype=torch.float32, device=self.device)
            reactant_masks = torch.as_tensor(reactant_masks, dtype=torch.float32, device=self.device)

            r_values = self.model(reactant_fps, reactant_masks)
            r_gap = - r_values - reaction_costs + target_values + 7.
            r_gap = torch.clamp(r_gap, min=0)
            consistency_loss.append((r_gap ** 2).mean().item())

        # Fitting Loss 평가
        for batch in self.val_fitting_loader:
            fps, values, masks = batch
            fps = torch.as_tensor(fps, dtype=torch.float32, device=self.device)
            values = torch.as_tensor(values, dtype=torch.float32, device=self.device).reshape(-1)
            masks = torch.as_tensor(masks, dtype=torch.float32, device=self.device)

            # val_fitting은 단일 분자 fps(배치,2048) 형태가 올 수 있으므로 3차원으로 패딩
            if fps.dim() == 2:
                fps = fps.unsqueeze(1)  # (B,1,2048)
            if fps.size(1) < masks.size(1):
                bsz, target_mols, fp_dim = fps.size(0), masks.size(1), fps.size(2)
                padded = torch.zeros((bsz, target_mols, fp_dim), device=self.device, dtype=fps.dtype)
                padded[:, :fps.size(1), :] = fps
                fps = padded

            v_pred = self.model(fps, masks)
            fitting_loss.append(F.mse_loss(v_pred.reshape(-1), values).item())

        # 결과 기록
        fr = open('test.txt', 'a')
        fr.write(str(np.mean(consistency_loss)) + '\t' + str(np.mean(fitting_loss)) + '\n')
        fr.close()

    def sampleMols(self):
        """
        분자 개수 샘플링 (확률적)

        분포:
        - 1개: 69%
        - 2개: 25%
        - 3개: 5%
        - 4개: 0.5%
        - 5개: 0.5%

        Returns:
            int: 샘플링된 분자 개수
        """
        randomNumber = np.random.rand()
        if randomNumber < 0.69:
            return 1
        if randomNumber < 0.94:
            return 2
        if randomNumber < 0.99:
            return 3
        if randomNumber < 0.995:
            return 4
        return 5

    def train(self):
        """
        학습 메인 루프

        각 에포크마다:
        1. Consistency 배치 샘플링
        2. 분자 개수 샘플링
        3. Fitting 배치 생성
        4. 학습 패스 실행
        5. 주기적으로 평가 및 저장
        """
        self.model.train()
        for epoch in range(self.n_epochs):
            # Consistency 배치 가져오기 (에포크 끝나면 리셋)
            consistency_batch = next(self.train_consistency_iter)
            if len(consistency_batch) < self.batch_size:
                self.train_consistency_loader = DataLoader(self.train_consistency, batch_size=self.batch_size, shuffle=True)
                self.train_consistency_iter = iter(self.train_consistency_loader)
                consistency_batch = next(self.train_consistency_iter)

            # 분자 개수 샘플링 및 Fitting 배치 생성
            num_mols = self.sampleMols()
            self.num_mols = num_mols
            fitting_batch = self.sample(num_mols, self.batch_size)

            # 학습
            total_loss, fit_loss, cons_loss = self._pass(consistency_batch, fitting_batch)

            # 중간 로그 출력
            if (epoch + 1) % self.log_interval == 0:
                print(f"[Train] Epoch {epoch + 1} | total: {total_loss:.4f} | fit: {fit_loss:.4f} | cons: {cons_loss:.4f}", flush=True)

            # 주기적 평가 및 저장
            if (epoch + 1) % self.test_epoch == 0:
                save_file = self.model_folder + '/epoch_%d.pt' % (epoch + 1)
                torch.save(self.model.module.state_dict(), save_file)  # DataParallel 모델 저장
                self.eval()
                self.model.train()


if __name__ == '__main__':
    """
    가치 함수 학습 실행

    설정:
    - 4개 GPU 병렬 학습 (DataParallel)
    - 배치 크기: 1024
    - 학습률: 0.001
    - 100 에포크마다 평가 및 저장
    """
    # 학습 하이퍼파라미터
    n_epochs = 1000000000  # 매우 큰 값 (수동으로 중단할 때까지 학습)
    lr = 0.001
    batch_size = 1024
    model_folder = './model'
    device = 'cuda:0'
    test_epoch = 100  # 평가 주기
    log_interval = 100  # 중간 로그 출력 주기(에포크 기준)

    # 모델 생성 및 병렬화
    model = ValueEnsemble(2048, 128, dropout_rate=0.1)
    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])  # 4개 GPU 사용

    # 트레이너 생성 및 학습 시작
    trainer = Trainer(model, n_epochs, lr, batch_size, model_folder, device, test_epoch, log_interval=log_interval)
    trainer.train()
