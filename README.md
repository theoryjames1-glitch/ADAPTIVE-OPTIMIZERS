
## 1. What is an optimizer?

In machine learning, an **optimizer** is the algorithm that updates model parameters (weights) so as to minimize a loss function. The simplest is **stochastic gradient descent (SGD):**

$$
\theta_{t+1} = \theta_t - \eta \, g_t
$$

where:

* $\theta_t$ = parameters at step $t$,
* $g_t = \nabla_\theta L(\theta_t)$ = gradient,
* $\eta$ = learning rate (fixed).

---

## 2. What makes an optimizer *adaptive*?

An **adaptive optimizer** automatically adjusts the **effective learning rate per parameter**, based on the history of gradients.
This lets it:

* take **larger steps** on infrequently updated parameters,
* take **smaller steps** on frequently updated parameters,
* reduce the need for manual tuning of a global learning rate.

---

## 3. How it works (examples)

### AdaGrad

Accumulates squared gradients to scale learning rate:

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t} + \epsilon} g_t
$$

where $G_t = \sum_{i=1}^t g_i^2$.

### RMSProp

Uses an **exponential moving average** (like an IIR filter) of squared gradients:

$$
v_t = \beta v_{t-1} + (1-\beta) g_t^2
$$

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t} + \epsilon} g_t
$$

### Adam

Adds a moving average of the raw gradient (momentum) on top of RMSProp:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
$$

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$

$$
\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
$$

Here $m_t, v_t$ are IIR-style recursions (feedback), so the optimizer “remembers” past gradients.

---

## 4. Key difference vs. adaptive filters

* **Adaptive filter:** changes coefficients to shape an input signal to minimize error (signal processing).
* **Adaptive optimizer:** changes *update rules* (effective step sizes) to adapt parameter search in optimization (learning).

Both rely on **recursions with feedback** (like IIR filters), but their goals differ:

* Adaptive filter → signal estimation / cancellation.
* Adaptive optimizer → faster, more stable convergence in training.

---

## 5. Why it matters

Adaptive optimizers (Adam, RMSProp, Adagrad, etc.) are popular because they:

* reduce sensitivity to initial learning rate,
* handle sparse features well,
* often converge faster in practice.

But they can sometimes **overfit or generalize worse** than simple SGD, so practitioners often combine them (e.g. train with Adam, then fine-tune with SGD).

---

✅ **In short:**
An **adaptive optimizer** is an algorithm that automatically tunes how much each parameter should be updated at each step, based on gradient history, rather than using a single fixed learning rate.

