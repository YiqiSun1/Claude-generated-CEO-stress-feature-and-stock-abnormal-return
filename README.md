## Project overview
This repository addresses a research question: Can CEOs’ internal stress predict future abnormal stock returns? The motivations are threefold. First, it’s important to model the effects of stress on performance-related activities. Traditionally, stress is proxied by questionnaire or dictionary approaches (Harvard IV-4 dictionary). Recently, the rise of LLM and ML models have been used to proxy stress. While these models might not be superior in this task, it provides an important research tool for researchers. Furthermore, LLM allows researchers to analyze large amounts of text without sifting through documents, increasing the scalability of studies. This research uses an approach of using LLM with psychology-backed prompts designed to proxy CEOs’ internal stress by analyzing their linguistic patterns in the quarterly earning calls. More discussion of this scoring methodology is explained later. By establishing a potential correlation between CEOs’ stress and stock abnormal returns, one can understand more about how stress affects performance in the work setting. 

Second, predicting stock return is an essential problem in finance. Analyzing if using LLM-based data features provides an “edge” is useful for understanding information processing in the financial market and the potential application of LLMs in the financial setting.

The stock abnormal return windows I tested are [0,+1] (the first trading day after the earnings call date), [+2,+7] (2-7 days after the earnings call), and [+2,+180]. I found several notable results. First, there exists a significant correlation between the stress score observed during the Q&A session of the earnings call and the stock abnormal return during the window [0, +1]. (-0.005, t=-5.275), meaning the market immediately incorporates the stress signal. Second, CEOs’ stress exhibited during prepared remarks has an equal correlation with stock abnormal return in the [0,+1] window as the Q&A session, suggesting both prepared remarks of the earnings call have unique information content. Finally, The stress signal has no predictive power in both the [+2,+7] and [+2,+180] windows. 

Following these findings, I constructed a long-short trading strategy with a one day holding period and conducted a backtest using data from 2020-2024. The entry rule is set on the day immediately after the earnings call, and the exit is the day after. The strategy only trades on earnings call days such that at least 2 longs and 2 shorts positions are available to ensure proper risk. The strategy allocates money equally across stocks to ensure profits are driven by the signal rather than by company size. This strategy fails to earn a positive return during the backtesting period from 2020 to 2024. Although this decision is made somewhat arbitrarily and further experiment is needed for the future. 

# Methodology
## Methodology overview

```
Earnings Call Transcripts
        ↓
Extract CEO speech (Prepared Remarks + Q&A)
        ↓
Score linguistic stress via Claude (1–5 scale)

Compute Cumulative Abnormal Returns (CAR)
        ↓
Merge with CRSP returns, Compustat fundamentals, IBES forecasts
        ↓
Regression analysis (OLS with clustered standard errors)
        
```

## Prompts used: 
```text
You are an expert in psycholinguistics and corporate financial communication.

Analyze the linguistic stress displayed by a CEO in the {section} of an earnings call.

Linguistic stress refers to HOW the CEO speaks, not WHAT topics are discussed.
Calibrate your scoring relative to a typical confident CEO earnings call — not everyday conversation.

Look for these markers:
1. HEDGING: "I think", "perhaps", "we hope", "roughly", "it's difficult to say"
2. PRONOUN DISTANCING: Shifting from "I" to "we", passive voice when addressing bad news
3. NEGATIVE AFFECT: concern, challenge, difficult, uncertainty, headwind, pressure
4. VAGUENESS: Avoiding specific numbers, deflecting questions, non-answers
5. EXCESSIVE QUALIFICATION: Unusual caveats, disclaimers, conditional statements
6. DISFLUENCY: Restarts, repetition, filler phrases ("you know", "sort of")
7. TEMPORAL ORIENTATION: Excessive focus on past problems vs forward guidance

Score 1 to 5 (integers only):
1 = No stress. Confident, direct, specific. Strong forward guidance.
2 = Minimal stress. Mostly confident with occasional hedging.
3 = Moderate stress. Noticeable hedging, some distancing or vagueness.
4 = High stress. Multiple markers present, evasiveness, strong negative affect.
5 = Very high stress. Pervasive markers throughout, highly evasive, emotionally loaded.
```
## Model
The model used for scoring is Claude Sonnet 4.6, a large language model developed by Anthropic. The model is prompted with the above instructions to evaluate the linguistic stress of CEOs during earnings calls. The scoring is done separately for the Prepared Remarks and the Q&A session, resulting in two stress scores per earnings call.

## Data Sources

| Source | Description |
|--------|-------------|
| earningscall API | S&P 500 earnings call transcripts (2020–2026) |
| CRSP | Daily stock returns and value-weighted market returns |
| Compustat | Quarterly firm-level financial fundamentals |
| IBES | Analyst EPS forecasts and actuals |
| Loughran-McDonald | Financial sentiment dictionary (1993–2025) |

Final samples include: 
| companies: 470 S&P 500 firms | quarters: 2020 Q1 – 2024 Q4 | observations: 8022 firm observations |

## Summary statistics: 
## Table 1: Summary Statistics

| Variable  | N     | Mean   | SD    | Min     | Median | Max    |
|-----------|-------|--------|-------|---------|--------|--------|
| CAR(0,1)  | 8,022 | 0.002  | 0.062 | −0.436  | 0.001  | 0.459  |
| Stress QA | 8,010 | 2.530  | 0.676 | 1.000   | 3.000  | 5.000  |
| Stress PR | 5,762 | 2.032  | 0.630 | 1.000   | 2.000  | 4.000  |
| VOL       | 7,866 | 0.021  | 0.010 | 0.006   | 0.018  | 0.105  |
| MOM       | 7,914 | 0.013  | 0.200 | −0.703  | −0.005 | 2.913  |
| LNMVE     | 7,758 | 10.502 | 1.039 | 7.220   | 10.349 | 15.075 |
| BM        | 7,758 | 0.340  | 0.324 | 0.000   | 0.248  | 5.338  |
| UE        | 7,764 | 0.133  | 0.529 | −17.010 | 0.070  | 11.000 |
| POSWORDS  | 8,022 | 0.020  | 0.006 | 0.000   | 0.019  | 0.150  |
| NEGWORDS  | 8,022 | 0.007  | 0.004 | 0.000   | 0.006  | 0.079  |

*This table reports summary statistics for the main variables used in the analysis. N denotes the number of observations. SD is the standard deviation.*

## Reproducing data: 
The exact reproducing step requires several different data sources and intermediate merging. A sample dataset is included in 'data/' for illustration. 

## Requirements
- EarningsCallAPI key
- Anthropic API key
- WRDS access (for CRSP, Compustat, IBES data)
- Loughran-McDonald dictionary (for sentiment analysis)

Because some of the merging steps are used through notebooks and require getting transcript data first, reproducing the exact data is difficult. The 'Code' repository provides core logic as guidance. Start with 'data/tickers.csv'

| Script | Purpose |
|:-------|:--------|
| `get_data.py` | Download earnings call transcripts from the EarningsCall API |
| `submit_batch.py` | Submit CEO speech segments to the Claude Batch API for stress scoring |
| `retrieve_results.py` | Retrieve stress scores from the Claude Batch API |
| `merging_everything.py` | Merge stress scores with fundamentals and analyst earnings forecasts |
| `compute_car_0_1.py` | Compute cumulative abnormal returns for the [0,+1] window |



## Regression form:
$$
\begin{aligned}
CAR_{i,t}(h_1, h_2) = \; & \alpha + \beta_1 \, STRESS_{i,t}
    + \beta_2 \, UE_{i,t} + \beta_3 \, LNMVE_{i,t} \\
    & + \beta_4 \, MOM_{i,t} + \beta_5 \, BM_{i,t}
    + \beta_6 \, VOL_{i,t} \\
    & + \beta_7 \, POSWORDS_{i,t} + \beta_8 \, NEGWORDS_{i,t}
    + \varepsilon_{i,t}
\end{aligned}
$$

### Dependent Variables
| Variable | Description |
|----------|-------------|
| `car_01` | Cumulative Abnormal Return, days 0 to +1 (2-day) |
| `car_0180` | Cumulative Abnormal Return, days 0 to +180 (6-month) | |
| `car_27` | Cumulative Abnormal Return, days +2 to +7 (6-day) |

### Independent Variables — Stress Scores
| Variable | Description |
|----------|-------------|
| `stress_pr` | CEO linguistic stress in Prepared Remarks (1–5 scale) |
| `stress_qa` | CEO linguistic stress in Q&A session (1–5 scale) |
| `stress_whole` | CEO linguistic stress across the full call (1–5 scale) |

### Control Variables
| Variable | Description |
|----------|-------------|
| `vol` | Historical volatility (std of returns, 125 trading days prior) |
| `mom` | Momentum (abnormal return, days −127 to −2) |
| `lnmve` | Log market value of equity (size) |
| `bm` | Book-to-market ratio (value factor) |
| `ue` | Unexpected earnings (Actual EPS − Analyst Median Forecast) |
| `POSWORDS` | Count of Loughran-McDonald positive words |
| `NEGWORDS` | Count of Loughran-McDonald negative words |




## Abnormal Returns Calculation

Abnormal Return (AR) for each trading day is computed as:

```
AR_t = R_t - R_market_t
```

where `R_market_t` is the CRSP value-weighted market return.

Cumulative Abnormal Returns:

```
CAR(t1, t2) = Σ AR_t  for t = t1 to t2
```






<!-- ```bash
pip install anthropic earningscall pandas numpy pyfixest python-dotenv
```

Additional dependencies for video analysis (exploratory):
```bash
pip install mediapipe opencv-python yt-dlp
``` -->

<!-- API credentials required:
- Anthropic API key (for Claude stress scoring)
- earningscall API key (for transcript access)
- WRDS access (for CRSP, Compustat, IBES data)

Store credentials in a `.env` file (never commit this file):
```
ANTHROPIC_API_KEY=your_key_here
EARNINGSCALL_API_KEY=your_key_here
```

--- -->

## Main results: 
1. 1 unit increase in stress level exhibited in the Q&A session is associated with a 0.5% decrease in the 2-day cumulative abnormal return holding common control variables including firm fundamentals, earnings surprise, and traditional dictionary-based sentiment measures. This means the market immediately reacts to the LLM stress signal.

**Dependent variable:** `car_01` · **Estimation:** OLS · **Inference:** clustered (CRV1) · **N:** 7,491 · **R²:**0.011 · **RMSE:** 0.042


| Coefficient | Estimate | Std. Error | t value | Pr(>\|t\|) | 2.5% | 97.5% |
|:------------|---------:|-----------:|--------:|-----------:|-------:|--------:|
| Intercept   |  0.027   | 0.008      |  3.469  | 0.002      |  0.011 |  0.044  |
| stress_qa   | −0.005   | 0.001      | −5.275  | 0.000      | −0.006 | −0.003  |
| vol         |  0.075   | 0.092      |  0.824  | 0.419      | −0.114 |  0.265  |
| mom         |  0.009   | 0.004      |  2.495  | 0.021      |  0.002 |  0.017  |
| lnmve       | −0.002   | 0.001      | −2.558  | 0.018      | −0.003 | −0.000  |
| bm          |  0.001   | 0.003      |  0.438  | 0.665      | −0.004 |  0.006  |
| UE          |  0.006   | 0.002      |  2.504  | 0.020      |  0.001 |  0.010  |
| POSWORDS    | −0.041   | 0.103      | −0.399  | 0.694      | −0.254 |  0.172  |
| NEGWORDS    | −0.024   | 0.157      | −0.155  | 0.878      | −0.350 |  0.301  |

2. Stress exhibited in the prepared remarks has an equal magnitude of effect on the 2-day cumulative abnormal return as the Q&A session. This means that both the prepared remarks and Q&A session are important for investors to understand the stress level of the CEO.

**Dependent variable:** `car_01` · **Estimation:** OLS · **Inference:** clustered (CRV1) · **N:** 5,342 · **R²:** 0.015 · **RMSE:** 0.039

| Coefficient | Estimate | Std. Error | t value | Pr(>\|t\|) | 2.5% | 97.5% |
|:------------|---------:|-----------:|--------:|-----------:|-------:|--------:|
| Intercept   |  0.039   | 0.011      |  3.571  | 0.002      |  0.016 |  0.061  |
| stress_qa   | −0.004   | 0.001      | −2.915  | 0.008      | −0.006 | −0.001  |
| stress_pr   | −0.004   | 0.001      | −2.656  | 0.014      | −0.007 | −0.001  |
| vol         |  0.082   | 0.089      |  0.925  | 0.365      | −0.102 |  0.266  |
| mom         |  0.006   | 0.005      |  1.265  | 0.219      | −0.004 |  0.017  |
| lnmve       | −0.002   | 0.001      | −2.475  | 0.022      | −0.004 | −0.000  |
| bm          |  0.001   | 0.003      |  0.445  | 0.661      | −0.005 |  0.008  |
| UE          |  0.006   | 0.003      |  2.051  | 0.052      | −0.000 |  0.012  |
| POSWORDS    | −0.105   | 0.099      | −1.057  | 0.302      | −0.311 |  0.101  |
| NEGWORDS    | −0.016   | 0.175      | −0.094  | 0.926      | −0.380 |  0.347  |
* one thing to note: This has a different sample size than the first regression result since in the initial data cleaning process, some prepared remarks are missing due to the quality of the transcript.


3. Stress signal has no predictive power on both the [2,7] and [2,180] CAR. This means the market adjust to the stress signal very quickly. 



<!-- ## Future:
Originally, I wanted to incorporate facial image of CEO or even specific characteristics of CEO face such as dark circles for proxy of stress. However, I had some difficulty thinking about how to get quality image of CEOs with universal lightings and angles. Furthermore, audio could be another useful things to add, but some research has done that already.  -->

<!-- ## Accessing the code and paper: 
You can access a paper version of this project in here. The paper does not include backtesting. (https://scholarship.claremont.edu/cmc_theses/4150/)

If you have questions or are interested in the code and methodology, contact me at ysun26@cmc.edu. -->

## Limitations: 
<!-- Here are several limitations and directions for future improvement. First, there is a risk of an omitted variable bias. There could be variables other than the variables specified in the regression that can "explain" the variation in stock abnormal return. Put in another word, there could be other variables that are correlated with the factors included. These could be company performance, market conditions, or other external factors. Therefore, this paper shouldn't conclude there is a causal relationship between internal stress score and future stock abnormal return. I want to point out this might not be as terrible as we think, as  -->

1. Data leakage remains the largest concern of this project. Claude Sonnet 4.6 was trained with data including 2020-2024, so it's possible the model already "knows" what happended in the stock market and generate stress score according to that knowledge. This is mitigated by the fact my prompts focuses on the scoring of text content without guiding the model to look at stock data. Several things are needed for improvement. First, use a model with cutoff date before the data being used.Second, stripping of the company name from the earning call and inspect if the correlation still exist. 
2. The scoring part requires further investigation. Future work would entail comparing the scoring of LLM with human scoring to see if LLM scoring makes sense. 3. CEOs' stress is hard to interpret in the current form. LLM generated stress could be related to internal stress or external events. Since the focus for this project is internal stress, future work could use exogeneous shocks for cleaner identification of CEOs stress. 



<!-- The stress produced by Claude could be correlated with these external factors, weakening the causal relationship between CEO stress and abnormal return.  -->

<!-- CEOs' stress is hard to interpret, as the stress could both come from things related to the company or internal stress. I do want to claim I don't emphasize this as much since I wanted to measure the stress level of CEO and didn't care too much about whehter it's internal caused or caused by external factors.  -->

<!-- Finally and the most important thing: The data leakage problem. The model I used (Claude Sonnet 4.6) might alrady "know" what happened since it might be trained on the the stock data during the period I conduct this research on. I admit this is a lack of consideration on my end at the beginning of the project and an issue with data limitation. Future research should deliberately use models that are trained on different data than the research data. However, this concern is mitigated by the fact that I prompt the model to only look at the transcript and output a stress score without guiding it to look at the stock data.  --> 

## Lesson learned: 
1. Data integrity is more essential than modeling. I realized very late that I only had TICKERs from my API-retrieved data, which is not as robust as identifiers such as PERMNO, which are available in other financial datasets. Furthermore, LLM data leakage problem is serious. Spending more time engineering data is worth the time. 
2. Implementing a strategy is a different topic from finding a signal. Even if a signal correlates with abnormal returns, that doesn’t mean it's a profitable strategy. Furthermore, thinking about the best way to trade is in itself an important and challenging problem.
3. One should be mindful of the project's structure and simplicity to facilitate future replication.


<!-- At the end of the day, this is a good exercise and shows LLM has huge potential in understanding human language and even psychology. I am both scared and excited about the future of LLM. -->

## Contact:
I am open to feedback on this project or collaboration on other things. ysun26@cmc.edu
