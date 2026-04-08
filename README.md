量化 ETF 動量研究

1. Introduction:
本專案實作了一套系統性的 ETF 配置策略量化研究流程, 驗證動量指標在不同市場條件下的有效性.
This project implements a systematic quantitative research pipeline for ETF allocation strategies, focusing on evaluating the effectiveness of momentum-based factors under different market conditions.

核心框架:
    市場條件(Regime) → 因子驗證(Factor Validation) → 策略(Strategy) → 效能評估(Evaluation)

重要組成部分:
    資料樣本: 20檔不同種類的台灣 ETF + 5檔美國債券 ETF(由於台灣債券 ETF 數據有限所以以美國債券作為替代)
    市場條件: 趨勢 × 波動率 (4 regimes)
    因子: 橫截面(Cross-sectional) & 時間序列(Time-series) 動量.
    因子驗證: IC (Information Coefficient) by regime + IC robustness.
    策略: 每月調整權重, 選擇表現最好的前 N 檔(CS-MOM), 只做多.
    風險: 手續費, 換手率, 資產級止損.
    效能評估: 夏普值, 最大回撤, 風險價值, 交易統計.

2. Conclusion:
    動量指標只有弱與不穩定的預測能力.
    效益對特定市場條件高度依賴.
    回溯期大小的選擇對結果表現的影響顯著.
    交易成本與止損對結果產生重大影響.
    弱因子在通過調整策略後仍然可以產生可接受的結果.

3. Insight
I. 因子驗證 > 策略:
    直接設計啟發式策略做效能評估無法解釋策略的優劣勢.
    通過因子驗證降低過度擬合和假 alpha 的發生.

II. Alpha 一般都是條件性:
    缺少市場條件的建立與後續分析, 就只看到不穩定的結果, 而看不到策略為什麼優勢/失效.
    動量策略的有效性取決於市場條件, 所以並不是具有普遍性的 alpha 來源.

III. 風險管理:
    止損重塑了策略的報酬分佈, 在降低波動率的同時也壓縮了報酬率.
    止損同時也有機會讓最大回撤惡化, 因為限縮了回彈的潛在機會.
    高換手率會放大交易成本，讓原本僅有微弱 alpha 的策略在扣除成本後迅速失去獲利能力.

4. Future Work:
    通過本專案觀察到 ETF 的收益表現相對平緩, 隨著換手率的增加, 交易成本會顯著的侵蝕獲利能力, 尤其是只有微弱 alpha 的策略, 因此, ETF 可能不太適合短期交易策略.
    未來的研究對象將轉向股票市場，因為股票市場更大的價格波動和更高的潛在收益可能為 alpha 的產生和資產配置策略提供更有利的環境。
