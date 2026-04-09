# Linear Regression Workshop — Slide Content

---

## Slide 1 — Cover

**Title:** Linear Regression
**Subtitle:** From a straight line to a working ML model
**Footer:** AI Club · [Date]

---

## Slide 2 — Regression: Predict a Number

**Heading:** What is Regression?

**One-liner:** Regression = predicting a **continuous number** as output.

**Contrast box:**
- Classification → "Is this email spam?" *(a category)*
- Regression → "How much will this house cost?" *(a number)*

**Examples (icon + label format):**
- 🏠 Predict house prices
- 🌡️ Predict tomorrow's temperature
- 🎓 Predict a student's exam score
- 📈 Predict monthly sales

**Takeaway line:** *If the answer is a number on a scale, think regression.*

---

## Slide 3 — Simple Linear Regression

**Heading:** Simple Linear Regression

**Key idea:** Find a straight line that best describes the relationship between one input and one output.

**Equation (large, centered):**
```
ŷ = β₀ + β₁x
```

**Term breakdown:**
| Symbol | Meaning |
|--------|---------|
| ŷ | Predicted value (dependent variable) |
| x | Input feature (independent variable) |
| β₀ | Intercept — where the line crosses the y-axis |
| β₁ | Slope — how much ŷ changes per unit of x |

**Why "simple"?** → Only **one** independent variable.
**Why "linear"?** → The equation is a **linear function** of x.

> 📊 **Figure:** `fig_slide3_fitted_line.png`
> Scatter plot of observations with the best-fit line drawn through them.

---

## Slide 4 — Parameter Optimization & Residuals

**Heading:** How Do We Fit the Line?

**Setup:** There are infinite possible lines. We need a way to pick the *best* one.

**Introduce residuals:**
> A **residual** is the gap between the actual value and what the model predicted.
```
residual = y - ŷ
```

**The goal:**
> In supervised learning, the goal is always to **minimize the gap** between predicted and actual values.

**Loss function (MSE):**
```
MSE = (1/n) Σ (yᵢ - ŷᵢ)²
```
- We square residuals so negatives don't cancel out positives
- We want this number as **small as possible**

**This is called:** Residual Sum of Squares minimization → leads to the best fit line.

> 📊 **Figure:** `fig_slide4_residuals.png`
> Scatter plot with vertical gold lines connecting each point to the regression line — each gap is a residual.

---

## Slide 5 — Gradient Descent & OLS

**Heading:** How Do We Minimize the Loss?

**Two approaches — side by side:**

### Ordinary Least Squares (OLS)
- A mathematical formula that solves for β directly
- Closed-form, exact, fast for small datasets
```
β = (XᵀX)⁻¹Xᵀy
```
- ✅ No iterations needed
- ❌ Expensive when data is very large

### Gradient Descent
- An iterative algorithm — takes small steps downhill on the loss curve
- Used in most ML frameworks under the hood
- ✅ Scales to large datasets
- Requires choosing a **learning rate**

**Key message:** Both methods find the same answer. OLS is exact, gradient descent is approximate but scalable.

> 📊 **Figure:** `fig_slide5_loss_curve.png`
> Loss vs. iteration curve — the model "learns" as loss drops and flattens toward convergence.

---

## Slide 6 — Multiple Linear Regression

**Heading:** Multiple Linear Regression

**Key idea:** Same concept, more input features.

**Equation:**
```
ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
```

**Example:**
> Predicting house price using:
> - x₁ = size (m²)
> - x₂ = number of rooms
> - x₃ = distance from city center

**Interpreting coefficients:**
> β₁ = 200 means: *for every 1 m² increase in size, the predicted price increases by 200 — holding all other features constant.*

**Watch out for:**
- Adding more features ≠ always better
- More features → more risk of overfitting

> 📊 **Figure:** `fig_bonus_correlation_heatmap.png`
> Correlation heatmap showing which features are most related to the target — use to motivate feature selection.

---

## Slide 7 — Metrics: RMSE, MAE, R²

**Heading:** How Good Is Our Model?

**Three key metrics:**

### MSE → RMSE
```
RMSE = √MSE
```
- Same unit as the target variable (interpretable!)
- Penalizes **large errors** more heavily (due to squaring)
- *"On average, my predictions are off by X units"*

### MAE
```
MAE = (1/n) Σ |yᵢ - ŷᵢ|
```
- Average absolute error — simple and robust
- Less sensitive to outliers than RMSE
- *"On average, I'm off by X, treating all errors equally"*

### R²
```
R² = 1 - (SS_res / SS_tot)
```
- Ranges from 0 to 1 (higher = better)
- *"My model explains X% of the variance in the data"*
- ⚠️ R² alone doesn't tell the full story — always check residuals too

**Quick comparison table:**

| Metric | Unit | Sensitive to Outliers |
|--------|------|-----------------------|
| RMSE | Same as y | Yes |
| MAE | Same as y | Less |
| R² | Unitless (0–1) | Somewhat |

---

## Slide 8 — Visual Evaluation

**Heading:** Always Plot Your Results

**Three diagnostic plots:**

### 1. Actual vs. Predicted Plot
- x-axis: actual values, y-axis: predicted values
- A perfect model → all points on the diagonal line y = x
- Clusters away from the line reveal systematic errors

### 2. Residual Plot
- x-axis: predicted values, y-axis: residuals
- Good model → residuals scattered **randomly** around 0
- Patterns in this plot = your model is missing something

### 3. Residual Distribution Plot
- Histogram or KDE of residuals
- Should look roughly **normal, centered at 0**
- Skewed distribution → model has a bias

**Key message:** Numbers lie. Plots don't. Always visualize before concluding.

> 📊 **Figures (use all three, side by side or stacked):**
> - `fig_slide8a_actual_vs_pred.png` — Points hugging the diagonal = good model
> - `fig_slide8b_residual_plot.png` — Residuals scattered around zero
> - `fig_slide8c_residual_dist.png` — Bell-shaped residual histogram

---

## Slide 9 — Assumptions & Polynomial Transformations

**Heading:** When the Line Isn't Enough

**Two important concepts:**

### Linearity Assumption
- Linear regression assumes the relationship between x and y is **linear**
- If the scatter plot curves → a straight line won't capture it well

**Homoscedasticity**
- Residuals should have **constant variance** across all predicted values
- If residuals fan out → this assumption is violated
- Fix: log-transform the target, or use a different model

### Polynomial Transformations
- You can extend linear regression to model curves by adding **polynomial features**
```
ŷ = β₀ + β₁x + β₂x²
```
- Still "linear" in terms of the coefficients
- Lets you fit curved relationships without switching models

> 📊 **Figure:** `fig_slide9_poly_comparison.png`
> Side-by-side: curved data with a straight line (bad fit, red) vs. degree-2 polynomial (good fit, green).

---

## Slide 10 — Bias–Variance Tradeoff

**Heading:** The Core Tension in Every ML Model

**Key idea:** Every model makes two kinds of errors. The goal is to balance both.

### Bias
- Error from wrong assumptions — the model is **too simple**
- High bias → **underfitting**: misses the real pattern even on training data
- Example: fitting a straight line to data that curves

### Variance
- Error from sensitivity to training data — the model is **too complex**
- High variance → **overfitting**: memorizes noise, fails on new data
- Example: a high-degree polynomial that wiggles through every point

**The tradeoff:**
```
Total Error = Bias² + Variance + Irreducible Noise
```

| | High Bias | Low Bias |
|---|---|---|
| **High Variance** | Worst of both worlds | Overfit |
| **Low Variance** | Underfit | ✅ Sweet spot |

**How to detect:**
- Training error ≈ Test error (both high) → **Underfitting** → add complexity
- Training error low, Test error high → **Overfitting** → simplify or regularize

> 📊 **Figure:** `fig_bias_variance.png`
> Three curves showing: underfitting (high bias, degree-1 line on curved data), good fit (balanced), overfitting (high variance, degree-N polynomial snaking through every point). Training vs. test error curves vs. model complexity also shown.

---

## Slide 11 — Regularization: Ridge & Lasso

**Heading:** Controlling Complexity with Regularization

**Problem:** When a model overfits, its coefficients grow very large — they latch onto noise.

**Solution:** Add a **penalty term** to the loss function that discourages large coefficients.

### Ridge Regression (L2)
```
Loss = MSE + λ · Σ βᵢ²
```
- Penalizes the **square** of each coefficient
- Shrinks all coefficients toward zero — but never exactly to zero
- ✅ Great when many features contribute a little
- Controlled by **λ** (lambda): larger λ → stronger shrinkage

### Lasso Regression (L1)
```
Loss = MSE + λ · Σ |βᵢ|
```
- Penalizes the **absolute value** of each coefficient
- Can shrink coefficients **all the way to zero** → built-in feature selection
- ✅ Great when only a few features actually matter
- Produces sparse models — easier to interpret

**Side-by-side comparison:**

| | Ridge (L2) | Lasso (L1) |
|---|---|---|
| Penalty | β² | \|β\| |
| Zeroes out features? | ❌ No | ✅ Yes |
| Best when | Many useful features | Few relevant features |
| Interpretability | Moderate | High (sparse) |

**Key message:** λ = 0 → plain OLS. λ → ∞ → all coefficients shrink to zero. Find the sweet spot with cross-validation.

> 📊 **Figure:** `fig_regularization.png` *(placeholder — generate with coefficient paths vs. λ for Ridge and Lasso side-by-side)*
> Left: Ridge coefficient paths — all shrink smoothly. Right: Lasso paths — some drop to zero (feature elimination).

---

## Slide 12 — When NOT to Use It

**Heading:** Linear Regression Is Not Always the Answer

**Avoid it when:**

| Situation | Why it fails | Use instead |
|-----------|-------------|-------------|
| Output is a category | Regression predicts numbers | Logistic Regression, SVM |
| Relationship is highly non-linear | Line can't capture the pattern | Decision Trees, Neural Nets |
| Data has many outliers | OLS is sensitive to extreme values | Robust Regression, Tree-based models |
| Features are correlated with each other | Unstable coefficients | Ridge / LASSO Regression |
| Too many irrelevant features | Noise inflates error | Lasso, feature selection |

**Key message:** *Linear regression is a powerful baseline — but always ask: does the data actually look linear?*

---

## Slide 13 — Hands-On: scikit-learn

**Heading:** Let's Build One

**Dataset:** California Housing

**Steps shown as numbered flow:**

**1. Load & explore the data**
```python
import pandas as pd
df = pd.read_csv("housing.csv")
df.head()
```

**2. Define features and target**
```python
X = df[["median_income", "housing_median_age", "rooms_per_household"]]
y = df["median_house_value"]
```

**3. Split the data**
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

**4. Train the model**
```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso
model = LinearRegression()
# Try: model = Ridge(alpha=1.0) or Lasso(alpha=0.1)
model.fit(X_train, y_train)
```

**5. Evaluate**
```python
from sklearn.metrics import mean_squared_error, r2_score
preds = model.predict(X_test)
print(r2_score(y_test, preds))
```

**6. Inspect coefficients**
```python
print(model.coef_)       # β₁, β₂, ...
print(model.intercept_)  # β₀
```

**Callout:** *Try swapping in Ridge or Lasso. Does R² improve? Which coefficients shrink?*

> 📊 **Figures (from notebook run):**
> - `fig_slide8a_actual_vs_pred.png` — reuse here to show model output
> - `fig_slide8b_residual_plot.png` — diagnostic check after fitting

---

## Slide 14 — Q&A / Thank You

**Heading:** That's a Wrap!

**Recap — what we covered:**
- What regression is and when to use it
- How a line gets fitted (OLS & Gradient Descent)
- Simple and multiple linear regression
- How to evaluate: RMSE, MAE, R²
- Visual diagnostics
- Assumptions and polynomial features
- Bias–variance tradeoff
- Regularization: Ridge (L2) and Lasso (L1)
- Hands-on with scikit-learn

**Resources to go deeper:**
- StatQuest with Josh Starmer (YouTube) — visual, beginner-friendly
- Hands-On ML by Aurélien Géron (book)
- scikit-learn docs: sklearn.linear_model

**Closing line:** *Linear regression is the "hello world" of ML. Master this, and everything else builds on it.*

---

## Figure Index

| File | Used in slide | Description |
|------|--------------|-------------|
| `fig_slide3_fitted_line.png` | Slide 3 | Scatter + best-fit line |
| `fig_slide4_residuals.png` | Slide 4 | Residual gaps visualised |
| `fig_slide5_loss_curve.png` | Slide 5 | Gradient descent loss curve |
| `fig_bonus_correlation_heatmap.png` | Slide 6 | Feature correlation heatmap |
| `fig_slide8a_actual_vs_pred.png` | Slides 8, 13 | Actual vs. Predicted scatter |
| `fig_slide8b_residual_plot.png` | Slides 8, 13 | Residual vs. Predicted scatter |
| `fig_slide8c_residual_dist.png` | Slide 8 | Residual histogram + KDE |
| `fig_slide9_poly_comparison.png` | Slide 9 | Linear vs polynomial fit |
| `fig_bias_variance.png` | Slide 10 | ⚠️ **NEW** — Bias-variance tradeoff diagram |
| `fig_regularization.png` | Slide 11 | ⚠️ **NEW** — Ridge vs Lasso coefficient paths |

---
*Total slides: 14 | Estimated time: 75 min*
