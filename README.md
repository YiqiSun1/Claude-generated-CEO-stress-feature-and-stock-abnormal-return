## Project overview
This repository addresses a research question: Can CEOs’ internal stress predict future abnormal stock returns? The motivations are threefold. First, it’s important to model the effects of stress on performance-related activities. Traditionally, stress is proxied by questionnaire or dictionary approaches (Harvard IV-4 dictionary). Recently, the rise of LLM and ML models have been used to proxy stress. While these models might not be superior in this task, it provides an important research tool for researchers. Furthermore, LLM allows researchers to analyze large amounts of text without sifting through documents, increasing the scalability of studies. This research uses an approach of using LLM with psychology-backed prompts designed to proxy CEOs’ internal stress by analyzing their linguistic patterns in the quarterly earning calls. More discussion of this scoring methodology is explained later. By establishing a potential correlation between CEOs’ stress and stock abnormal returns, one can understand more about how stress affects performance in the work setting. 

Second, predicting stock return is an essential problem in finance. Analyzing if using LLM-based data features provides an “edge” is useful for understanding information processing in the financial market and the potential application of LLMs in the financial setting.

The stock abnormal return windows I tested are [0,+1] (the first trading day after the earnings call date), [+2,+7] (2-7 days after the earnings call), and [+2,+180]. I found several notable results. First, there exists a significant correlation between the stress score observed during the Q&A session of the earnings call and the stock abnormal return during the window [0, +1]. (-0.005, t=-5.275), meaning the market immediately incorporates the stress signal. Second, CEOs’ stress exhibited during prepared remarks has a stronger correlation with stock abnormal return in the [0,+1] window than Q&A session, suggesting investors might pay more attention to the prepared remarks of the earnings call. Finally, The stress signal has no predictive power in both the [+2,+7] and [+2,+180] windows. 

Following these findings, I constructed a long-short trading strategy with a one day holding period and conducted a backtest using data from 2020-2024. The entry rule is set on the day immediately after the earnings call, and the exit is the day after. The strategy only trades on earnings call days such that at least 2 longs and 2 shorts positions are available to ensure proper risk. The strategy allocates money equally across stocks to ensure profits are driven by the signal rather than by company size. This strategy fails to earn a positive return during the backtesting period from 2020 to 2024. Although this decision is made somewhat arbitrarily and further experiment is needed for the future. 

# Methodology
## Methodology overview

```
Earnings Call Transcripts
        ↓
Extract CEO speech (Prepared Remarks + Q&A)
        ↓
Score linguistic stress via Claude (1–5 scale)
        ↓
Merge with CRSP returns, Compustat fundamentals, IBES forecasts
        ↓
Compute Cumulative Abnormal Returns (CAR)
        ↓
Regression analysis (OLS + Fixed Effects)
```

## Prompting used: 
'''text
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
'''

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


## Reproducing data: 

## Regression form:

<!-- ### Dependent Variables
| Variable | Description |
|----------|-------------|
| `car_01` | Cumulative Abnormal Return, days 0 to +1 (2-day) |
| `car_0180` | Cumulative Abnormal Return, days 0 to +180 (6-month) | -->


<!-- ### Independent Variables — Stress Scores
| Variable | Description |
|----------|-------------|
| `stress_pr` | CEO linguistic stress in Prepared Remarks (1–5 scale) |
| `stress_qa` | CEO linguistic stress in Q&A session (1–5 scale) |
| `stress_whole` | CEO linguistic stress across the full call (1–5 scale) | -->

### Control Variables
| Variable | Description |
|----------|-------------|
| `vol` | Historical volatility (std of returns, 125 trading days prior) |
| `mom` | Momentum (abnormal return, days −127 to −2) |
| `lnmve` | Log market value of equity (size) |
| `bm` | Book-to-market ratio (value factor) |
| `UE` | Unexpected earnings (Actual EPS − Analyst Median Forecast) |
| `POSWORDS` | Count of Loughran-McDonald positive words |
| `NEGWORDS` | Count of Loughran-McDonald negative words |




<!-- ## Abnormal Returns Calculation

Abnormal Return (AR) for each trading day is computed as:

```
AR_t = R_t - R_market_t
```

where `R_market_t` is the CRSP value-weighted market return.

Cumulative Abnormal Returns:

```
CAR(t1, t2) = Σ AR_t  for t = t1 to t2
``` -->






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
1. 1 unit increase in stress level exibited in the Q&A session is associated with a 0.9% decrease in the 2-day cumulative abnormal return holding common control variables including firm fundamentals, earnings surprise, and traditional dictionary-based sentiment measures. This means the market immediately react to the LLM stress signal. 

**Dependent variable:** `car_01` · **Estimation:** OLS · **Inference:** clustered (CRV1) · **N:** 7,491 · **R²:** 0.045 · **RMSE:** 0.057

| Coefficient | Estimate | Std. Error | t value | Pr(>\|t\|) |   2.5% |  97.5% |
|:------------|---------:|-----------:|--------:|----------:|-------:|-------:|
| Intercept   |    0.042 |      0.012 |   3.617 |     0.002 |  0.018 |  0.067 |
| **stress_qa** | **−0.009** |  **0.001** | **−6.277** | **0.000** | **−0.012** | **−0.006** |
| vol         |    0.351 |      0.099 |   3.562 |     0.002 |  0.147 |  0.556 |
| mom         |   −0.005 |      0.005 |  −1.114 |     0.277 | −0.016 |  0.005 |
| lnmve       |   −0.002 |      0.001 |  −2.395 |     0.026 | −0.004 | −0.000 |
| bm          |   −0.001 |      0.003 |  −0.403 |     0.691 | −0.008 |  0.005 |
| UE          |    0.031 |      0.004 |   8.048 |     0.000 |  0.023 |  0.039 |
| POSWORDS    |   −0.013 |      0.143 |  −0.089 |     0.930 | −0.309 |  0.283 |
| NEGWORDS    |   −0.764 |      0.237 |  −3.228 |     0.004 | −1.255 | −0.273 |

2. Stress exibited in the prepared remarks has a higher magnitude of effect on the 2-day cumulative abnormal return that the Q&A session. This is unexpected given my prior is that people would pay more attention to the Q&A session since CEO can't prepare the answer to spontaneous questions. 

**Dependent variable:** `car_01` · **Estimation:** OLS · **Inference:** clustered (CRV1) · **N:** 5,342 · **R²:** 0.062 · **RMSE:** 0.056

| Coefficient | Estimate | Std. Error | t value | Pr(>\|t\|) |   2.5% |  97.5% |
|:------------|---------:|-----------:|--------:|----------:|-------:|-------:|
| Intercept   |    0.058 |      0.014 |   3.977 |     0.001 |  0.028 |  0.088 |
| **stress_qa** | **-0.006** | **0.002** | **-3.757** | **0.001** | **-0.010** | **-0.003** |
| **stress_pr** | **-0.013** | **0.002** | **-6.870** | **0.000** | **-0.016** | **-0.009** |
| vol         |    0.472 |      0.109 |   4.352 |     0.000 |  0.247 |  0.697 |
| mom         |   -0.014 |      0.007 |  -1.880 |     0.073 | -0.029 |  0.001 |
| lnmve       |   -0.002 |      0.001 |  -1.951 |     0.064 | -0.005 |  0.000 |
| bm          |    0.000 |      0.004 |   0.003 |     0.997 | -0.009 |  0.009 |
| UE          |    0.031 |      0.004 |   7.676 |     0.000 |  0.023 |  0.040 |
| POSWORDS    |    0.025 |      0.135 |   0.187 |     0.853 | -0.255 |  0.305 |
| NEGWORDS    |   -0.322 |      0.310 |  -1.039 |     0.310 | -0.966 |  0.321 |

* one thing to note: This has a different sample size that the first regressio result since in the initial data cleaning process, some prepared remarks are missing due to the quality of the transcript.


3. Stress signal has no predictive power on both the [2,7] and [2,180] CAR. This means the market adjust to the stress signal very quickly. 



<!-- ## Future:
Originally, I wanted to incorporate facial image of CEO or even specific characteristics of CEO face such as dark circles for proxy of stress. However, I had some difficulty thinking about how to get quality image of CEOs with universal lightings and angles. Furthermore, audio could be another useful things to add, but some research has done that already.  -->

<!-- ## Accessing the code and paper: 
You can access a paper version of this project in here. The paper does not include backtesting. (https://scholarship.claremont.edu/cmc_theses/4150/)

If you have questions or are interested in the code and methodology, contact me at ysun26@cmc.edu. -->

## Limitations: 
Here are several limitations and directions for future improvement. First, there is a risk of an omitted variable bias. There could be variables other than the variables specified in the regression that can "explain" the variation in stock abnormal return. Put in another word, there could be other variables that are correlated with the factors included. These could be company performance, market conditions, or other external factors. Therefore, this paper shouldn't conclude there is a causal relationship between internal stress score and future stock abnormal return. I want to point out this might not be as terrible as we think, as 

<!-- The stress produced by Claude could be correlated with these external factors, weakening the causal relationship between CEO stress and abnormal return.  -->

Secondly, CEOs' stress is hard to interpret, as the stress could both come from things related to the company or internal stress. I do want to claim I don't emphasize this as much since I wanted to measure the stress level of CEO and didn't care too much about whehter it's internal caused or caused by external factors. 

Finally and the most important thing: The data leakage problem. The model I used (Claude Sonnet 4.6) might alrady "know" what happened since it might be trained on the the stock data during the period I conduct this research on. I admit this is a lack of consideration on my end at the beginning of the project and an issue with data limitation. Future research should deliberately use models that are trained on different data than the research data. However, this concern is mitigated by the fact that I prompt the model to only look at the transcript and output a stress score without guiding it to look at the stock data. 

<!-- At the end of the day, this is a good exercise and shows LLM has huge potential in understanding human language and even psychology. I am both scared and excited about the future of LLM. -->

## Critique and advice:
I am open to suggestions to improve this project or collaboration on other things. ysun26@cmc.edu
