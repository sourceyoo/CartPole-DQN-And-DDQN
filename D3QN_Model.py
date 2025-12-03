from torch import nn
from torch.nn import functional

class Model(nn.Module):
    def __init__(self, input_features, output_values):
        super(Model, self).__init__()
        
        # 1. 공통 특징 추출 레이어 (Shared Feature Layer)
        # 기존 fc1은 그대로 유지하여 특징을 뽑아냅니다.
        self.fc1 = nn.Linear(in_features=input_features, out_features=32)

        # ---------------------------------------------------------
        # 여기서부터 두 갈래로 나뉩니다.
        # ---------------------------------------------------------

        # 2. Value Stream (상태 가치 V)
        # 상태 그 자체의 점수를 매기므로 출력(out_features)은 무조건 1개입니다.
        self.fc_value = nn.Linear(in_features=32, out_features=32)
        self.out_value = nn.Linear(in_features=32, out_features=1)

        # 3. Advantage Stream (행동 이점 A)
        # 각 행동 별 점수 차이를 매기므로 출력(out_features)은 행동의 개수(output_values)입니다.
        self.fc_adv = nn.Linear(in_features=32, out_features=32)
        self.out_adv = nn.Linear(in_features=32, out_features=output_values)

    def forward(self, x):
        # 공통 레이어 통과
        x = functional.selu(self.fc1(x))

        # Value 계산
        v = functional.selu(self.fc_value(x))
        v = self.out_value(v)

        # Advantage 계산
        a = functional.selu(self.fc_adv(x))
        a = self.out_adv(a)

        # 4. 결합 (Aggregation)
        # 공식: Q(s,a) = V(s) + (A(s,a) - A의 평균)
        # A의 평균을 빼주는 이유는 학습의 안정성을 위해서입니다.
        x = v + (a - a.mean(dim=1, keepdim=True))
        
        return x