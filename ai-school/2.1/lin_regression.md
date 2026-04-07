# Workshop Plan: Regression
**Open Minds Club | AI School**
---

## 0. Machine Learning
### What is machine learning?

### Hard vs Soft Computing

- Hard computing = classical programming. Precise, deterministic, rule-based. Works perfectly when the problem is well-defined and doesn't change. Example: calculating a square root, sorting a list.
- Soft computing = ML, fuzzy logic, neural networks. Tolerates uncertainty, approximates, learns from data. Works when problems are too complex, ambiguous, or dynamic for explicit rules.

### Supervised vs. Unsupervised Learning

### Regression vs. Classification

- Regression: label continuous
- Classification: discrete label 

## 1. What is Regression? (5 min)

- Supervised learning: predicting a **continuous** target value
- Real examples: predicting bike rentals, house prices, temperature
- Contrast with classification (continuous vs discrete output)

---

## 2. The Linear Regression Model (10 min)

### The equation
- Prediction as a **weighted sum** of features:
  ```
  ŷ = w₀ + w₁x₁ + w₂x₂ + ... + wₚxₚ
  ```
- `w₀` = intercept (bias)
- `w₁...wₚ` = learned weights (coefficients)
- `ε` = irreducible error (noise)

### What the model learns
- The weights — one per feature
- Higher weight = stronger influence on prediction
- Sign of weight = direction of influence (positive / negative)

### Quick interpretation (brief)
- Numerical feature: increasing x by 1 unit changes ŷ by `wᵢ`
- Categorical feature: weight = difference vs reference category

---

## 3. The Loss Function (10 min)

### Why we need one
- The model needs a way to measure how wrong it is
- We compare predicted `ŷ` vs actual `y`

### Mean Squared Error (MSE)
```
MSE = (1/n) Σ (yᵢ - ŷᵢ)²
```
- Squaring: penalizes large errors more than small ones
- Always positive
- We want to **minimize** this

### Why not absolute error?
- MSE is differentiable everywhere → easier to optimize
- MAE (Mean Absolute Error) is valid too, briefly mention it

### Visualizing the loss surface
- 1-feature case: loss is a **parabola** over weight values
- 2-feature case: a **bowl-shaped surface** in 3D
- Our goal: find the bottom of the bowl

---

## 4. Gradient Descent (15 min)

> *This is the engine behind almost all of ML — not just regression*

### The intuition
- We're standing on a hilly landscape (the loss surface)
- We want to reach the lowest point (minimum loss)
- Strategy: always step in the direction that goes downhill

### The algorithm
1. Initialize weights randomly (or at zero)
2. Compute the gradient of the loss w.r.t. each weight
3. Update weights in the **opposite** direction of the gradient:
   ```
   w := w - α · ∂L/∂w
   ```
4. Repeat until convergence

### The learning rate `α`
- Too large → overshoots, diverges
- Too small → converges very slowly
- Just right → smooth descent to minimum

### Variants
| Variant | How it works | Tradeoff |
|---|---|---|
| Batch GD | Uses all data per step | Stable but slow |
| Stochastic GD (SGD) | Uses 1 sample per step | Fast but noisy |
| Mini-batch GD | Uses small batches | Best of both (most common) |

### Why gradient descent over the closed-form solution?
- Closed-form (Normal Equation): `w = (XᵀX)⁻¹Xᵀy`
- Works perfectly for linear regression
- But: **doesn't scale** to large datasets or complex models (neural nets)
- Gradient descent generalizes to everything → that's why we learn it here

---

## 5. Evaluating the Model (8 min)

### R² (R-squared)
```
R² = 1 - SSE/SST
```
- 1.0 = perfect fit | 0.0 = no better than predicting the mean | <0 = worse than mean
- Limitation: increases with more features even if they add no value

### Adjusted R²
- Penalizes for number of features
- Use this when comparing models with different feature counts

### Train / Test split
- Never evaluate on the same data you trained on
- Typical split: 80% train, 20% test
- Cross-validation for more robust estimates

### Overfitting vs Underfitting
- Underfit: model too simple, high error on both train and test
- Overfit: model memorizes training data, fails on test data
- Linear regression tends to underfit when relationships are non-linear

---

## 6. Feature Types & Encoding (5 min)

### Numerical features
- Feed directly into the model
- Consider standardizing (mean=0, std=1) for stable training

### Categorical features — One-Hot Encoding
- Convert categories to binary columns
- L categories → L-1 columns (drop one to avoid multicollinearity)
- Example: `weather = {GOOD, MISTY, BAD}` → two binary columns

### Why encoding matters
- Model can only do math on numbers
- Wrong encoding = meaningless weights

---

## 7. Regularization (7 min)

### The problem: too many features
- More features → model can overfit
- Some features are noise, not signal

### Lasso (L1 regularization)
```
Loss = MSE + λ Σ|wᵢ|
```
- Forces many weights to exactly **zero** → automatic feature selection
- Larger λ = sparser model = fewer features

### Ridge (L2 regularization)
```
Loss = MSE + λ Σwᵢ²
```
- Shrinks all weights but keeps them all non-zero
- Better when all features are somewhat relevant

### The λ tradeoff
- λ = 0 → plain linear regression
- λ → ∞ → all weights shrink to zero
- Choose λ via cross-validation

---

## 8. Beyond Linear: A Peek Forward (5 min)

### When linear regression fails
- Non-linear relationships between features and target
- Feature interactions (e.g., temperature × season)

### What comes next
- **Polynomial features**: add x², x³ as explicit features
- **Decision Trees & Random Forests**: non-linear, no assumptions
- **Neural Networks**: stack of linear layers + non-linearities — gradient descent still powers them all

> *The intuition from linear regression + gradient descent carries directly into deep learning*

---

## Key Takeaways

1. Regression = predicting a continuous value as a weighted sum of features
2. The model learns by minimizing a loss function (MSE)
3. Gradient descent is the algorithm that does the learning — update weights step by step in the direction that reduces loss
4. Evaluate with R² on a held-out test set
5. Regularization (Lasso/Ridge) prevents overfitting by penalizing large weights
6. Linear regression is a building block — understand it deeply and the rest of ML follows naturally

---

## Hands-on (if time allows)

- Fit a linear regression on the bike rental dataset
- Plot the loss curve during gradient descent
- Tune λ for Lasso, observe which features drop to zero
- Compare train vs test R² to diagnose overfitting
