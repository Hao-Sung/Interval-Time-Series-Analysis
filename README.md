# 區間式時間序列分析

本專案的目的，在於使用Python語言，實現區間資料常用的時間序列模型，以及視覺化方法。模型與方法的定義，可參閱「參考文獻」所條列的論文。

架構上，`ITSA`為專案模組，`example.ipynb`則為範例程式碼。


## 前置作業

進入專案後，以`pip`安裝所需套件
```
pip install -r requirements.txt
```
再匯入模組即可。

## 使用範例

```python
# 匯入自定義模組
import ITSA.simulation as sim
from ITSA.models.AIRMA import AIRMA

# 生成模擬資料
interval_ts = sim.airma(
    phi=[0.3], theta=[-0.7], sigma=1, nsamples=1000, nsimulations=250
)

# 模型配適
model = AIRMA(endog=interval_ts, order=(1,1), n_sample=1000)
model.fit(iters=1000)

# 配適結果一覽
model.summary()
```
其他範例程式碼，可參見`example.ipynb`。

## 參考文獻

1. Lin L-C, Sung H, Lee S. Comprehensive interval-valued time series model
with application to the S&P 500 index and PM2.5 level data analysis. Appl Stochastic Models Bus Ind. 2022;1-21.
doi: 10.1002/asmb.2733
2. Lin L-C, Chien H-L, Lee S. Symbolic interval-valued data analysis for time series based on auto-interval-regressive models. Stat Methods
Appl. 2021;30:295-315.
3. Zhang M, Lin DK. Visualization for interval data. J Comput Graph Stat. 2022. doi:10.1080/10618600.2022.2066678