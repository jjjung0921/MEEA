"""
GraphEncoder.py
===============
그래프 인코더 모듈 - 분자의 그래프 표현을 처리하는 신경망

역할:
- 여러 분자들의 fingerprint를 집계하여 단일 벡터로 인코딩
- ValueEnsemble 모델에서 사용됨
- 마스킹을 통해 가변 길이의 분자 집합 처리
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphLayer(nn.Module):
    """
    기본 그래프 레이어
    선형 변환 + ReLU 활성화 + 드롭아웃
    """
    def __init__(self, in_features, out_features):
        """
        Args:
            in_features: 입력 특징 차원
            out_features: 출력 특징 차원
        """
        super(GraphLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)  # 선형 변환
        self.drop = nn.Dropout(0.1)  # 과적합 방지를 위한 드롭아웃 (10%)

    def forward(self, x):
        """
        순전파

        Args:
            x: 입력 텐서

        Returns:
            변환된 텐서 (ReLU + 드롭아웃 적용)
        """
        x = self.drop(F.relu(self.linear(x)))
        return x


class GraphModel(nn.Module):
    """
    그래프 모델 - 여러 분자를 집계하여 하나의 표현으로 인코딩

    동작 방식:
    1. 각 분자의 fingerprint를 GraphLayer로 변환
    2. 마스크를 적용하여 유효한 분자만 선택
    3. 모든 분자를 합산하여 집합 표현 생성
    """
    def __init__(self, input_dim, feature_dim, hidden_dim):
        """
        Args:
            input_dim: 입력 차원 (fingerprint 크기, 보통 2048)
            feature_dim: 특징 차원 (사용되지 않음, 호환성 유지용)
            hidden_dim: 은닉층 차원
        """
        super(GraphModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.graphlayer = GraphLayer(input_dim, hidden_dim)  # 단일 그래프 레이어

    def forward(self, x, mask):
        """
        순전파: 여러 분자를 하나의 벡터로 집계

        Args:
            x: 입력 텐서 (batch_size, num_molecules, input_dim)
            mask: 마스크 텐서 (batch_size, num_molecules)
                  - 1: 유효한 분자, 0: 패딩

        Returns:
            집계된 벡터 (batch_size, hidden_dim)
        """
        x = self.graphlayer(x)  # (batch_size, num_molecules, hidden_dim)

        # 마스크를 hidden_dim만큼 복제하여 각 특징에 적용
        mask = mask[:, :, None].repeat(1, 1, self.hidden_dim)

        # 마스킹 후 합산 (유효한 분자들만 집계)
        x = torch.sum(x * mask, dim=1)  # (batch_size, hidden_dim)
        return x
