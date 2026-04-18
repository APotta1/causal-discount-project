# General idea of this project

This note explains **what problem you are solving** and **how the repo is organized around that story**. For install and commands, see the main [README.md](README.md).

---

## The business question

**Does giving a discount actually cause someone to buy, or were they going to buy anyway?**

That matters if you want to know whether discounts are “worth it”: you need the **incremental** effect of the discount, not just “buyers who had a discount look better than buyers who didn’t.”

### Same idea, shorter

Using **fake data**, we build customers with traits you can explore in the notebooks—**age**, **income**, **prior purchases**, **sessions**, mobile vs not, and so on. In EDA you see how **treated vs control** differ on those dimensions (not only age, but age is one of the “groups” you can look at).

- The **naive comparison** (average purchase rate with a discount minus without) is the quick story: “Did the discount work?” In this setup it often **overstates** success, because many discounted people **already looked like buyers**.
- The **better methods** (regression adjustment, DiD, PSM) are there to answer a stricter question: **Did the discount actually change purchase probability**, after accounting for who tends to get targeted? They line up **much closer to the known true effect** in the simulation.
- **Uplift (T-learner)** pushes that one step further: **for whom** does the discount really add lift—the people who only buy *because* of the deal, vs people who would have bought anyway? That supports **targeting** discounts toward responsive customers instead of blanketing everyone.

So: fake customers, a **misleading** naive “it worked” headline, and **causal tools** that testify to what **really** happened for purchase intent–driven targeting.

---

## Why a simple comparison lies

In practice, **discounts are not random**. Marketing often targets **high-intent** customers—people who already look likely to purchase. So:

- The **discount group** can look better on average.
- Part of that gap is **selection**: those people were already more likely to buy **before** the discount.
- A naive “treated vs control” average **mixes** the real effect of the discount with **who got targeted**.

So the “idea of discounts” here is: **you cannot trust a raw comparison** when treatment assignment depends on customer traits.

---

## What this project does about it

1. **Simulate** that world: fake customers, some get a discount, assignment is **confounded** (correlated with who would buy anyway).

2. **Know the truth** inside the simulation: the code defines the **true** average effect of the discount on purchase probability. That lets you **measure bias** of each estimator.

3. **Compare methods**
   - **Naive difference-in-means** — simple treated minus control; usually **biased** here.
   - **Regression adjustment, Difference-in-Differences, Propensity Score Matching** — adjust for confounding in different ways; typically **closer** to the true effect in this DGP.
   - **Uplift (T-learner)** — estimates **who** responds more to a discount, which supports **targeted** vs **blanket** discount strategies (still about incremental conversion in this repo, not full profit-per-item economics).

4. **Uncertainty** — bootstrap confidence intervals for some estimators, and explicit **bias vs ground truth** in the causal-methods notebook.

---

## How the two notebooks fit

| Notebook | Role in plain words |
|----------|---------------------|
| **`01_eda.ipynb`** | **Look and understand** the simulated data: group sizes, covariate differences between treated and control, naive ATE vs truth, pre/post trends, propensity overlap. |
| **`02_causal_methods.ipynb`** | **Run the estimators** (naive, regression, DiD, PSM), compare to truth, show bias, then uplift and targeted vs blanket discount simulation. |

Run **01** first if you want the intuition; **02** is where the full causal pipeline runs.

---

## What this project is *not* (yet)

The outcome is **purchase yes/no** (and covariates), not **margin or cost** per product. So you are learning **causal lift on conversion**, not a full **ROI** model for “discounts on the exact items in the basket.” That would add pricing, margins, and costs on top of this causal layer.

---

## One-sentence summary

**Fake data where discounts target eager buyers, known true effect, then show naive comparisons exaggerate success and causal methods + uplift get closer to reality and smarter targeting.**
