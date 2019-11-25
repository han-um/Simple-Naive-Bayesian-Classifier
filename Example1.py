"""From Think Bayes(앨런 B. 다우니 지음, 권정민 옮김, 한빛미디어 펴냄)"""

import thinkbayes2 as tb

class Cookie(tb.Pmf):
    """A map from string bowl ID to probablity."""

    # 사전확률 정의
    def __init__(self, hypos):
        """Initialize self.

        hypos: sequence of string bowl IDs
        """
        tb.Pmf.__init__(self)
        for hypo in hypos:
            # 가설마다 동일한 사전확률(1) 부여
            self.Set(hypo, 1)
        # 총합이 1이 되도록(확률이 되도록) 정규화
        # 가설이 두 개이므로 정규화 결과 가설마다 0.5 사전확률 가짐
        self.Normalize()

    # 사후확률 구하기
    def Update(self, data):
        """Updates the PMF with new data.

        data: string cookie type
        """
        for hypo in self.Values():
            # 우도 구하기
            like = self.Likelihood(data, hypo)
            # 사전확률에 우도 업데이트(곱)
            self.Mult(hypo, like)
        # 사전확률 x 우도 모든 값의 합이 1이 되도록 정규화
        # 한정상수가 동일하므로 사후확률 계산에서 생략
        self.Normalize()

    mixes = {
        'Bowl 1': dict(vanilla=0.75, chocolate=0.25),
        'Bowl 2': dict(vanilla=0.5, chocolate=0.5),
    }

    # 우도 함수
    def Likelihood(self, data, hypo):
        """The likelihood of the data under the hypothesis.

        data: string cookie type
        hypo: string bowl ID
        """
        # mixes 데이터 자체가 우도를 나타내고 있으므로
        # 그냥 불러오기만 하면 우도함수 정의 끝
        mix = self.mixes[hypo]
        like = mix[data]
        return like


# 가설정의 (가설1=그릇1, 가설2=그릇2)
hypos = ['Bowl 1', 'Bowl 2']

# 가설에 따른 객체 선언
pmf = Cookie(hypos)

# 사후확률 구하기
# 사전확률에 우도 업데이트 + 정규화
pmf.Update('vanilla')

# 결과 출력
for hypo, prob in pmf.Items():
    print hypo, prob