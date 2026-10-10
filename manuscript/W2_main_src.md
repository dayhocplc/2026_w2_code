---
title: "Monotonicity by construction in two-layer fuzzy aggregation: a stochastic-dominance condition for clipped Mamdani inference and convex learning of monotone 2-additive Choquet integrals"
---

*Target journal: International Journal of Approximate Reasoning (Elsevier). Article type: Research paper.*

**Authors:** *[author list, order, affiliations and ORCID identifiers to be completed]*

**Corresponding author:** *[corresponding author and e-mail to be added]*

# Highlights

- Bounded-sum rule aggregation makes Mamdani firing vectors stochastically monotone.
- Output monotonicity then reduces to one check on the defuzzifier over a simplex.
- An exact cell-enumeration certificate settles the check for clipped centroids.
- Monotone 2-additive Choquet learning becomes a convex bound-constrained problem.
- It matches the choquistic-regression optimum one to two orders of magnitude faster.

# Abstract

Screening instruments that aggregate symptom scores should be monotone: a worse answer must never lower the estimated risk. For a two-layer fuzzy model, in which a Mamdani layer maps domain scores to domain risks and a 2-additive Choquet layer fuses them, we show how to guarantee monotonicity by construction in both layers. For the Mamdani layer with clipping implication and centroid defuzzification, we prove that if every input is described by an ordered Ruspini partition, antecedents are combined by the product t-norm and rules sharing a consequent are aggregated by the bounded sum, then the vector of consequent strengths is a probability vector that increases in first-order stochastic dominance for any monotone rule base, smooth or not, and any number of inputs. Output monotonicity then reduces to a single condition on the defuzzifier over the probability simplex, which depends only on the consequent sets; we give an exact certificate for it by enumerating the cells of a finite line arrangement. For the Choquet layer we prove that the Grabisch monotonicity constraints of a 2-additive capacity are equivalent to nonnegativity bounds under a simple reparametrization, so that penalized logistic learning becomes a convex bound-constrained problem solvable by L-BFGS-B, and that its unpenalized version has the same optimum as choquistic regression restricted to 2-additive capacities. Experiments on an autism-screening cohort and three clinical data sets show that each hypothesis is needed, reproduce the choquistic optimum at a small fraction of its cost, and find no significant loss of discrimination from monotonicity.

**Keywords:** Monotone fuzzy inference; Mamdani–Assilian model; Ruspini partition; First-order stochastic dominance; Choquet integral; 2-additive capacity

# 1. Introduction

A screening questionnaire turns a caregiver's answers into a risk estimate and, eventually, into a referral decision. Whatever model sits between the answers and the decision, one property is non-negotiable: if every answer stays the same or moves towards the more concerning end of its scale, the risk must not go down. A model that violates this property can be accurate on average and still produce individual outputs that no clinician can defend. Monotonicity is therefore a safety requirement rather than a modelling preference, and it is best guaranteed by construction rather than checked after the fact on a finite set of inputs [[cano2019]]. Machine learning offers several monotone-by-construction model classes, from lattice models [[gupta2016, you2017]] to gradient boosting with monotone constraints [[ke2017]], and fuzzy modelling has long used constrained model structures for the same purpose [[lindskog2000]]; here we are concerned with the interpretable fuzzy aggregation models on which screening instruments are often built.

This paper is about a specific, and in our experience typical, architecture for such scores: a two-layer fuzzy aggregation model. In the first layer, one Mamdani–Assilian fuzzy inference system [[mamdani1975]] per symptom domain maps a domain deficit score and an age-lag index to a domain risk. In the second layer, a Choquet integral [[choquet1954]] with respect to a 2-additive capacity [[grabisch1997]] fuses the domain risks, so that pairs of domains may reinforce or substitute for one another. Both layers are popular because they are interpretable: the first through linguistic rules, the second through Shapley values and interaction indices [[grabisch2010]]. Both layers can also silently violate monotonicity. Fig. 1 shows the architecture and where each of our results applies.

For Mamdani systems, monotonicity has been studied in depth. Van Broekhoven and De Baets characterized monotone Mamdani–Assilian models under mean-of-maxima [[vbdb2008]] and centre-of-gravity (COG) defuzzification [[vbdb2009]], for trapezoidal fuzzy partitions and the minimum, product and Łukasiewicz t-norms. Under COG they showed that with more than one input only *smooth* rule bases, in which the consequents of adjacent cells differ by at most one label, can generate monotone models, that the product t-norm guarantees monotonicity for every smooth monotone rule base with two inputs, and that with three inputs the output sets must in addition share a shape. Sufficient conditions are also available for Takagi–Sugeno–Kang systems [[won2002, seki2010, kerk2025]], for hierarchical sum–product systems [[kouikoglou2009]] and for Gaussian membership functions [[tay2011]], and rule bases that violate monotonicity can be relabelled [[pang2016]]. In practice, however, many systems are built with the configuration that common toolboxes offer by default: product or minimum conjunction, *clipping* (minimum) implication and maximum aggregation. We found that in this configuration a screening model with a monotone and smooth rule base produced domain risks that fell by up to {{num(MAXDROP['max'],1)}} points on a 100-point scale while the deficit score rose, even with properly partitioned inputs, and that its original membership functions, which passed the standard design check that membership degrees cover the input range, made the total violation two to three times larger.

For Choquet integrals, monotonicity of the capacity guarantees monotonicity of the integral, and learning a monotone capacity from data is a constrained optimization problem [[grabisch2008, miranda1999, beliakov2009, beliakov2019, kojadinovic2007]]. Choquistic regression [[tehrani2011, tehrani2012]] made this problem a probabilistic classifier, P(Y = 1 | x) = σ(γ(C_μ(x) − β)), fitted by sequential quadratic programming with the full set of p·2^(p−1) monotonicity constraints; later work reduced the complexity for 2-additive capacities [[hullermeier2013]] and trained 2-additive models inside neural architectures with a softplus reparametrization [[bresson2020]]. What is missing, to our knowledge, is an explicit statement that for 2-additive capacities the problem is *convex after a change of variables*, together with a solver that needs only simple bounds.

**Contributions.** The paper makes four contributions.

1. *A stochastic-dominance lemma for the Mamdani layer (Lemma 1).* With ordered Ruspini partitions [[ruspini1969]], product conjunction and bounded-sum aggregation of rules that share a consequent, the vector of consequent strengths is a probability distribution over the ordered output labels, and it increases in first-order stochastic dominance (FOSD) in every input for *any* monotone rule base. The proof is a two-line coupling argument; it needs neither smoothness of the rule base nor a bound on the number of inputs.
2. *A reduction and a certificate (Theorem 1, Propositions 1–2).* Output monotonicity then holds whenever the defuzzifier, viewed as a function on the probability simplex, is monotone along the directions that move mass to the next label. This condition involves only the consequent sets. For the additive (SAM) defuzzifier it has a closed form; for clipped COG on a finite output universe we give an exact certificate by enumerating the cells of a finite line arrangement, on which the defuzzifier is linear-fractional.
3. *A convex learner for the Choquet layer (Theorem 2, Propositions 3–4).* We prove that the Grabisch monotonicity constraints of a 2-additive capacity, a closed-form node condition, and a nonnegativity reparametrization are equivalent; that penalized logistic learning in the reparametrized variables is a bound-constrained convex problem whose solutions are exactly those of the original problem; and that its unpenalized version has the same optimum as choquistic regression restricted to 2-additive capacities.
4. *Evidence.* On the grid-scale behaviour of the Mamdani layer, on a Polish autism-screening cohort and on three clinical benchmark data sets, we show that each hypothesis of Lemma 1 is needed, that the reparametrized solver reaches the choquistic optimum at a fraction of the cost, and that imposing monotonicity costs no discrimination where the monotone direction is valid.

Section 2 fixes notation. Section 3 treats the Mamdani layer and Section 4 the Choquet layer; Section 5 combines them. Section 6 describes the experiments, Section 7 reports them and Section 8 discusses the results in relation to prior work.

# 2. Preliminaries

## 2.1 Ordered Ruspini partitions

Let X = [a, b] ⊂ ℝ. A family A_1, …, A_K of fuzzy sets on X is a *Ruspini partition* if Σ_k A_k(x) = 1 for every x ∈ X [[ruspini1969]]. We call it *ordered* if, in addition, every cumulative sum
$$F_k(x)=\sum_{l\le k}A_l(x),\qquad k=1,\dots,K-1,$$
is nonincreasing in x. The ordering formalizes the requirement that the labels are listed from "low" to "high". It holds for the usual trapezoidal partitions in which only consecutive sets overlap: on the overlap of A_k and A_(k+1) all sets A_l with l < k vanish, so F_k = A_k = 1 − A_(k+1), which is nonincreasing because A_k decreases there; to the left of the overlap F_k = 1 and to the right F_k = 0. A partition that merely *covers* X, i.e. Σ_k A_k(x) > 0 everywhere, need not be Ruspini, and a Ruspini partition need not be ordered.

## 2.2 The Mamdani layer

Consider n inputs x = (x_1, …, x_n) ∈ X_1 × ⋯ × X_n, and for input i a family A_(i,1), …, A_(i,K_i). A *complete* rule base assigns to every multi-index j = (j_1, …, j_n) ∈ 𝒥 = [K_1] × ⋯ × [K_n] one consequent label r(j) ∈ [M] = {1, …, M}; the consequent labels are fuzzy sets B_1, …, B_M on an output universe Z. The rule base is *monotone* if j ≤ j′ componentwise implies r(j) ≤ r(j′), and *smooth* if r changes by at most one label between adjacent cells [[vbdb2009]]. The system computes

- the firing strength of rule j with product conjunction, τ_j(x) = ∏_i A_(i,j_i)(x_i);
- the strength of each consequent label, w_c(x) = S(τ_j(x) : r(j) = c), where S is a t-conorm — the maximum S_M, the probabilistic sum S_P or the bounded (Łukasiewicz) sum S_L(u, v) = min(1, u + v);
- the output fuzzy set with clipping implication, B′_x(z) = max_c min(w_c(x), B_c(z));
- the crisp output by COG, y(x) = Σ_z z B′_x(z) / Σ_z B′_x(z), with Z a finite grid (the continuous case replaces sums by integrals).

With S = S_M this is the classical Mamdani–Assilian model with product conjunction. The only change we make is to how rules that *share a consequent* are combined; rules with different consequents are still combined by the maximum. The output depends on x only through the vector w(x) = (w_1(x), …, w_M(x)), and we write
$$y(x)=D\big(w(x)\big),\qquad D(w)=\frac{\sum_{z\in Z} z\,\max_c \min\{w_c,B_c(z)\}}{\sum_{z\in Z}\max_c \min\{w_c,B_c(z)\}}.\tag{1}$$
We call D the *defuzzifier map*. It depends only on the consequent sets and on Z.

## 2.3 First-order stochastic dominance on the simplex

Let Δ_M = {w ∈ ℝ^M : w ≥ 0, Σ_c w_c = 1}. For w, w′ ∈ Δ_M we write w ⪯ w′ (w′ dominates w) if every upper tail satisfies T_k(w) = Σ_(c≥k) w_c ≤ T_k(w′), k = 2, …, M. We write v_k = e_k − e_(k−1) for the direction that moves mass from label k − 1 to label k. A function D on Δ_M is *FOSD-isotone* if w ⪯ w′ implies D(w) ≤ D(w′).

## 2.4 Capacities and the 2-additive Choquet integral

Let N = {1, …, p}. A set function μ : 2^N → ℝ with μ(∅) = 0 is a *capacity* if it is monotone, A ⊆ B ⇒ μ(A) ≤ μ(B); it is *normalized* if μ(N) = 1. Its Möbius transform m satisfies μ(A) = Σ_(B⊆A) m(B), and μ is *2-additive* if m(B) = 0 whenever |B| > 2 [[grabisch1997]]. We write m_i = m({i}) and m_ij = m({i, j}). The Choquet integral of x ∈ [0, 1]^p with respect to μ is C_μ(x) = Σ_k (x_(k) − x_(k−1)) μ({(k), …, (p)}), where x_(1) ≤ ⋯ ≤ x_(p) and x_(0) = 0; for a 2-additive μ it takes the Möbius form [[grabisch1997]]
$$C_m(x)=\sum_{i}m_i x_i+\sum_{i<j}m_{ij}\min(x_i,x_j).\tag{2}$$
If μ is a capacity, C_μ is nondecreasing in every argument. The same formulas define C_m for an unnormalized m, which we use below.

# 3. The Mamdani layer

## 3.1 A stochastic-dominance lemma

**Lemma 1.** *Assume that (i) the sets of every input form an ordered Ruspini partition, (ii) the conjunction is the product, (iii) the rule base is complete and (iv) rules sharing a consequent are aggregated by the bounded sum S_L. Then for every x:*

*(a) w(x) ∈ Δ_M, and w_c(x) = Pr[r(J(x)) = c], where J(x) = (J_1, …, J_n) has independent coordinates with Pr[J_i = k] = A_(i,k)(x_i);*

*(b) if the rule base is monotone and x ≤ x′ componentwise, then w(x) ⪯ w(x′).*

*Proof.* (a) Since every family is Ruspini, Σ_(j∈𝒥) τ_j(x) = ∏_i Σ_k A_(i,k)(x_i) = 1, so the numbers τ_j(x) are the probabilities of a product distribution on 𝒥; this is the law of J(x). For each c the partial sum Σ_(r(j)=c) τ_j(x) is at most one, so the bounded sum equals the ordinary sum, and w_c(x) = Pr[r(J(x)) = c]. By completeness Σ_c w_c(x) = 1.

(b) Couple the two distributions. Let U_1, …, U_n be independent uniform variables on (0, 1] and set J_i(x_i) = min{k : F_(i,k)(x_i) ≥ U_i}, with F_(i,K_i) ≡ 1. Then Pr[J_i(x_i) ≤ k] = F_(i,k)(x_i), so J(x) has the law in (a). If x_i ≤ x′_i, the ordering gives F_(i,k)(x′_i) ≤ F_(i,k)(x_i) for all k, hence J_i(x′_i) ≥ J_i(x_i) for every realization of U_i. Monotonicity of r then gives r(J(x′)) ≥ r(J(x)) pointwise, and therefore Pr[r(J(x′)) ≥ k] ≥ Pr[r(J(x)) ≥ k] for every k, which is w(x) ⪯ w(x′). ∎

The lemma is the whole reason for replacing the maximum by the bounded sum. With S_M, w_c(x) is the largest of the rule strengths that share label c, Σ_c w_c(x) falls below one wherever several rules share a label, and the tails of w(x) can decrease when x increases; the same happens with S_P and with partitions that are not Ruspini. Section 7.1 shows each failure numerically.

## 3.2 Reduction to a condition on the defuzzifier

**Lemma 2.** *A function D : Δ_M → ℝ is FOSD-isotone if and only if, for every w ∈ Δ_M and every k = 2, …, M, the map t ↦ D(w + t v_k) is nondecreasing on {t ≥ 0 : w + t v_k ∈ Δ_M}.*

*Proof.* Necessity: w + t v_k differs from w only in the tail T_k, which increases by t, so w ⪯ w + t v_k. Sufficiency: let w ⪯ w′ and G_k = T_k(w′) − T_k(w) ≥ 0. A direct computation gives w′ − w = Σ_(k=2)^M G_k v_k. Move from w to w′ in the order k = 2, 3, …, M. After step k the point has coordinates w′_c for c < k, w_k + G_k for c = k and w_c for c > k, all nonnegative, so each step is a move along a single direction v_k inside Δ_M, along which D does not decrease. ∎

**Theorem 1.** *Under the assumptions of Lemma 1 with a monotone rule base, if the defuzzifier map D of the system is FOSD-isotone, then y(x) = D(w(x)) is nondecreasing in every input; more precisely, x ≤ x′ componentwise implies y(x) ≤ y(x′).*

*Proof.* Lemma 1 gives w(x) ⪯ w(x′); FOSD-isotonicity of D gives D(w(x)) ≤ D(w(x′)). ∎

We refer to FOSD-isotonicity of the defuzzifier map as *condition (D)*.

Three features of Theorem 1 matter in practice. First, it separates the *input side* of the system (partitions, conjunction, rule base, aggregation), which Lemma 1 handles for any number of inputs and any monotone rule base, from the *output side* (consequent sets and defuzzifier), which enters only through the finite-dimensional map D. Second, the condition on D does not depend on the rule base, so it is checked once per choice of consequent sets. Third, the theorem applies to any pipeline whose output is a function of w(x) alone; the clipped COG of (1) is one example, the additive model below is another.

## 3.3 Checking the defuzzifier condition

**Proposition 1 (additive defuzzifier).** *Suppose the output is the standard additive model (SAM) [[kosko1997]], in which clipped consequents are replaced by scaled ones and combined by summation, so that D(w) = Σ_c w_c V_c ζ_c / Σ_c w_c V_c, where V_c > 0 and ζ_c are the area and the centroid of B_c and ζ_1 < ⋯ < ζ_M. If, for every k = 2, …, M,*
$$V_k(\zeta_k-y)-V_{k-1}(\zeta_{k-1}-y)\ \ge 0\quad\text{for }y\in\{\zeta_1,\zeta_M\},\tag{3}$$
*then D is FOSD-isotone. In particular, (3) holds whenever all consequents have the same area.*

*Proof.* The derivative of D along v_k equals [V_k(ζ_k − D(w)) − V_(k−1)(ζ_(k−1) − D(w))] / Σ_c w_c V_c. The bracket is affine in D(w) and D(w) ∈ [ζ_1, ζ_M], so it is nonnegative on that interval if it is nonnegative at both end points. Lemma 2 concludes. ∎

For clipped COG, D in (1) has no closed form, but it has a structure that makes an exact check possible.

**Proposition 2 (certificate for clipped COG).** *Let Z be finite and let 𝒯 = {B_c(z) : c ∈ [M], z ∈ Z} ∩ [0, 1]. Consider the arrangement 𝒜 of the hyperplanes {w_c = t} (c ∈ [M], t ∈ 𝒯) and {w_c = w_c′} (c ≠ c′) in the affine hull of Δ_M. On every open cell K of 𝒜 intersected with Δ_M,*
$$D(w)=\frac{a_K^\top w+b_K}{c_K^\top w+e_K}\tag{4}$$
*for constants a_K, b_K, c_K, e_K, and the sign of the directional derivative of D along v_k equals the sign of the affine function*
$$N_{K,k}(w)=(a_K^\top v_k)(c_K^\top w+e_K)-(c_K^\top v_k)(a_K^\top w+b_K).$$
*Consequently D is FOSD-isotone if N_(K,k)(u) ≥ 0 for every cell K, every vertex u of its closure and every k.*

*Proof.* Inside a cell, for every z the comparison of w_c with B_c(z) and the ranking of the terms min{w_c, B_c(z)} are fixed, so the aggregated value at z is either a constant B_c(z) or a coordinate w_c. Numerator and denominator of (1) are therefore affine in w, which gives (4), and the formula for the derivative of a ratio gives N_(K,k). An affine function on a polytope attains its minimum at a vertex, so N_(K,k) ≥ 0 on the closure of K. The map D is continuous on Δ_M, and a segment in direction v_k crosses finitely many cells; on each piece D is nondecreasing, so it is nondecreasing on the whole segment, and Lemma 2 concludes. ∎

For M = 3 labels the arrangement lives in a triangle and consists of lines in six directions. Algorithm 1 enumerates it.

**Algorithm 1 (certificate for M = 3).** *Input:* consequent values B_c(z) on the grid Z. (1) Form the threshold set 𝒯 and the 3|𝒯| + 3 lines of 𝒜 in the coordinates (w_1, w_2), w_3 = 1 − w_1 − w_2. (2) Compute every pairwise intersection of non-parallel lines inside the triangle; together with the triangle corners these are the vertices of 𝒜. (3) At every vertex u, evaluate the cell coefficients (a_K, b_K, c_K, e_K) at the points u + δ(cos θ_s, sin θ_s), s = 1, …, 24, with θ_s = (s − ½)·15° and δ = 10⁻⁹; since the minimum angle between two of the six line directions is about 18.4°, every cell incident to u contains at least one of these points. (4) Evaluate N_(K,k)(u) for k = 2, 3 and return the minimum of N_(K,k)(u)/(c_K^⊤u + e_K)². *Output:* the certificate is positive if this minimum is nonnegative.

The quantity returned is the smallest directional derivative of D over the whole simplex along either direction. For trapezoidal output partitions, exact closed-form COG computations are also available [[vbdb2006]]; the arrangement argument does not need them because it works directly on the output grid. The cost is quadratic in |𝒯|, which is at most M·|Z|.

# 4. The Choquet layer

## 4.1 Three equivalent monotonicity conditions

**Theorem 2.** *For a 2-additive m = (m_i, m_ij), the following statements are equivalent.*

*(G) μ is monotone; equivalently [[chateauneuf1989, miranda1999]], m_i + Σ_(j∈T) m_ij ≥ 0 for every i and every T ⊆ N \ {i} (p·2^(p−1) inequalities).*

*(C) m_i + Σ_(j≠i) min(m_ij, 0) ≥ 0 for every i (p inequalities).*

*(R) There exist s ∈ ℝ^p_+ and m⁺, m⁻ ∈ ℝ^(p(p−1)/2)_+ such that m_ij = m⁺_ij − m⁻_ij and m_i = s_i + Σ_(j≠i) m⁻_ij for every i.*

*Proof.* A set function is monotone if and only if it does not decrease when one element is added, and for a 2-additive μ, μ(T ∪ {i}) − μ(T) = m_i + Σ_(j∈T) m_ij, which gives the Möbius form of (G). (G) ⇒ (C): take T = {j : m_ij < 0}. (C) ⇒ (G): Σ_(j∈T) m_ij ≥ Σ_(j∈T) min(m_ij, 0) ≥ Σ_(j≠i) min(m_ij, 0). (C) ⇒ (R): take the minimal decomposition m⁺_ij = max(m_ij, 0), m⁻_ij = max(−m_ij, 0) and s_i = m_i + Σ_j min(m_ij, 0) ≥ 0. (R) ⇒ (C): min(m⁺_ij − m⁻_ij, 0) ≥ −m⁻_ij, so m_i + Σ_j min(m_ij, 0) ≥ m_i − Σ_j m⁻_ij = s_i ≥ 0. ∎

Substituting (R) into (2) and using x_i + x_j = min(x_i, x_j) + max(x_i, x_j) gives
$$C_m(x)=\sum_i s_i x_i+\sum_{i<j}\big[m^+_{ij}\min(x_i,x_j)+m^-_{ij}\max(x_i,x_j)\big],\tag{5}$$
a nonnegative combination of the features x_i, min(x_i, x_j) and max(x_i, x_j). The form (5) with nonnegative weights appears in Bresson et al. [[bresson2020]], who attribute it to [[grabisch1997]] and enforce the sign constraints by a softplus map inside a neural network trained by stochastic gradient descent. Our use of it differs in two respects: Theorem 2 makes the equivalence with the original constraints explicit, including the correspondence of the variables, and Propositions 3–4 below use it to obtain a convex problem that is solved to global optimality with simple bounds.

## 4.2 Penalized logistic learning as a bound-constrained convex problem

Given data (x^(t), y^(t)), t = 1, …, n, with x^(t) ∈ [0, 1]^p and y^(t) ∈ {0, 1}, let ℓ(a, m) be the mean logistic loss of η^(t) = a + C_m(x^(t)). For λ ≥ 0 and γ ≥ 0 consider

(P) minimize ℓ(a, m) + λ Σ_(i<j) |m_ij| + (γ/2) Σ_i m_i² subject to (G);

(P′) minimize ℓ(a, m(s, m⁻), m⁺ − m⁻) + λ Σ_(i<j) (m⁺_ij + m⁻_ij) + (γ/2) Σ_i m_i(s, m⁻)² subject to s, m⁺, m⁻ ≥ 0,

where m_i(s, m⁻) = s_i + Σ_j m⁻_ij.

**Proposition 3.** *(a) (P) and (P′) have the same optimal value, the image of every minimizer of (P′) is a minimizer of (P), and the minimal decomposition of every minimizer of (P) is a minimizer of (P′). (b) If λ > 0, every minimizer of (P′) satisfies m⁺_ij m⁻_ij = 0. (c) (P′) is a convex problem with bound constraints only.*

*Proof.* (a) By Theorem 2 the image of a feasible point of (P′) is feasible for (P), with objective no larger because |m⁺_ij − m⁻_ij| ≤ m⁺_ij + m⁻_ij; conversely the minimal decomposition of a feasible point of (P) is feasible for (P′) with the same objective. (b) If m⁺_ij and m⁻_ij are both positive, subtract δ = min(m⁺_ij, m⁻_ij) from both and add δ to s_i and s_j. Then m_ij, m_i and m_j are unchanged, the point stays feasible, and the objective falls by 2λδ. (c) ℓ is convex in (a, m) as a logistic loss of an affine function; (a, s, m⁺, m⁻) ↦ (a, m) is linear, so the composition is convex, and so are the penalty terms. ∎

By (5), (P′) is an L1/L2-penalized logistic regression with nonnegative coefficients on the features (x_i, min, max) and a free intercept. We solve it with L-BFGS-B [[byrd1995, zhu1997]] using the analytic gradient, along a decreasing path of λ values with warm starts. Because L-BFGS-B can stop on its relative-reduction criterion before the projected gradient vanishes, we check the Karush–Kuhn–Tucker condition after every run and restart from the last iterate while the largest projected-gradient component exceeds 10⁻⁵. The small ridge term γ > 0 is not needed for the theory; it keeps the scale of m bounded when the training data are nearly separable.

## 4.3 Relation to choquistic regression

Choquistic regression [[tehrani2012]] models P(Y = 1 | x) = σ(γ(C_μ(x) − β)) with a normalized capacity μ, a scale γ > 0 and a threshold β ∈ [0, 1], and fits (μ, γ, β) by penalized maximum likelihood with a sequential quadratic programming solver.

**Proposition 4.** *Restricted to 2-additive capacities, the model σ(γ(C_μ(x) − β)) with normalized monotone μ and γ > 0 coincides with the model σ(a + C_m(x)) with monotone unnormalized m and C_m(1, …, 1) > 0, through m = γμ, a = −γβ, with inverse γ = C_m(1, …, 1), μ = m/γ, β = −a/γ. Consequently, whenever the maximum-likelihood threshold of the former lies in [0, 1], the unpenalized versions of the two problems have the same optimal likelihood; the former is bilinear in (γ, μ) and nonconvex, the latter is (P′) with λ = γ = 0 and convex.*

*Proof.* The Möbius transform, and hence C_m, is linear in the capacity, so γ C_μ = C_(γμ); monotonicity is invariant under positive scaling, and C_m(1, …, 1) = μ_m(N). The map and its inverse are bijections between the two parameter sets, apart from the threshold constraint. ∎

Proposition 4 does not say that the two procedures give the same model in practice: the penalties differ. The L1 penalty of [[tehrani2012]] acts on the normalized Möbius coefficients, and because Σ m = 1 with m_i ≥ 0 under (C), the penalty on a 2-additive capacity equals 1 + 2 Σ_(i<j) m⁻_ij, so it shrinks only *negative* interactions, whereas our penalty shrinks all interactions towards a sparse graph. The proposition does say that the *optimization* problems are the same up to a change of variables, which we use in Section 7.4 to compare solvers on identical optima.

# 5. The two-layer system

**Corollary 1.** *Let every domain-level Mamdani system satisfy the hypotheses of Theorem 1, so that the domain risk r_i = y_i(d_i, LAG_i) is nondecreasing in the deficit score d_i and in the age-lag index LAG_i, and let the fusion layer be P(Y = 1 | r) = σ(a + C_m(r/100)) with m satisfying any condition of Theorem 2. Then P(Y = 1) is nondecreasing in every d_i and every LAG_i.*

*Proof.* A composition of maps that are nondecreasing in each argument is nondecreasing in each argument, and σ is increasing. ∎

When the Choquet layer is fitted directly on item scores rather than on domain risks, the direction of each input must be fixed before fitting. For risk scores produced by the first layer it is known by construction. For raw items the direction has to be supplied: Fallah Tehrani et al. [[tehrani2012]] reverse attributes known to act in the decreasing direction, and here we estimate it within each training fold, reversing a feature whose training-fold AUC is below 0.5; monotonicity then holds with respect to the oriented features. Section 8 returns to this choice.

# 6. Experimental design

## 6.1 The Mamdani layer

The domain systems of the screening model have two inputs, the domain deficit score d ∈ [0, 1] and the age-lag index LAG ∈ [0, 3] (in standard deviations), each with three labels, and three output labels on [0, 100] with a grid step of 0.5. In the revised design the labels of d are NONE = tri(0, 0, 0.45), MILD = tri(0, 0.45, 0.80) and CLEAR = trap(0.45, 0.80, 1, 1); those of LAG are ON-TIME = tri(0, 0, 1), MILD = tri(0, 1, 2.5) and SEVERE = trap(1, 2.5, 3, 3); and the output labels are LOW = tri(0, 0, 35), MEDIUM = tri(25, 50, 75) and HIGH = tri(65, 100, 100). The nine rules map (d, LAG) label pairs to LOW for (NONE, ON-TIME), (NONE, MILD) and (MILD, ON-TIME); to MEDIUM for (NONE, SEVERE), (MILD, MILD) and (CLEAR, ON-TIME); and to HIGH otherwise. The rule base is complete, monotone and smooth. The original design used membership functions that covered the input ranges but summed to as little as {{num(MT['partitions']['MF_D_V1']['min_sum'],2)}} for d and {{num(MT['partitions']['MF_LAG_V1']['min_sum'],2)}} for LAG; the revised design uses ordered Ruspini partitions. We measured monotonicity violation on a regular grid with step 0.01 in both inputs (101 × 301 points) as the total amount by which the output *decreases* between neighbouring grid points when one input increases. The measure is zero if and only if the output is nondecreasing on the grid; it grows with grid resolution when violations exist.

We evaluated five configurations that cross the partition type (original or Ruspini) with the aggregation of rules sharing a consequent (S_M, S_P, S_L), six variants that cross the conjunction (product or minimum) and the implication (clipping or product scaling) with the aggregation (S_M or S_L), each with the original smooth rule base and a non-smooth monotone rule base in which the label jumps from LOW to HIGH between adjacent cells. For every configuration we also computed the largest decrease of the tails T_2 and T_3 of w(x) along the grid, a direct check of the conclusion of Lemma 1. We then ran Algorithm 1 for the project's consequents and for a deliberately skewed set (a narrow LOW, a symmetric MEDIUM and a wide HIGH), and a stress test on {{MT['stress']['bounded_sum']['n']}} random systems with two or three inputs, two to four labels per input, random ordered trapezoidal Ruspini partitions and random monotone rule bases, {{MT['stress']['n_nonsmooth_rulebases']}} of which were non-smooth, evaluated on a 41 × 41 or 17 × 17 × 17 grid.

## 6.2 The Choquet layer: data and protocol

{{TABLE1}}

We used four binary classification data sets (Table 1). POL252 is a published case–control cohort of Polish toddlers with the 25-item Quantitative Checklist for Autism in Toddlers (Q-CHAT) [[allison2008, niedzwiecka2022, niedzwiecka_data]], items scored 0–4 in the direction of autistic traits; the diagnosis was assigned in the community, independently of the questionnaire. The three benchmark data sets come from outside autism: the UCI Parkinson's voice data [[little2009]], the original Wisconsin breast-cancer data with nine ordinal cytology items [[wolberg1990]] and the Cleveland heart-disease data [[detrano1989]]. The last one contains nominal attributes (chest-pain type, resting ECG, ST slope, thallium scan), for which no monotone direction exists; we kept it as a case in which the monotonicity assumption is partly wrong.

Inputs were mapped to [0, 1] by fixed affine maps for the ordinal instruments (POL252: x/4; BCW683: (x − 1)/9) and by the empirical distribution function of the training fold for continuous attributes; all maps are monotone. Features were oriented within the training fold as described in Section 5. We used 5-fold stratified cross-validation repeated 10 times (seed 20260925). Within each training fold, λ ∈ {10⁻², 3·10⁻³, 10⁻³, 3·10⁻⁴} was chosen by an inner 3-fold cross-validation maximizing the AUC, with γ = 10⁻³.

**Methods compared.** (i) *Monotone 2-additive Choquet* (proposed; (P′) solved by L-BFGS-B). (ii) *Unconstrained 2-additive Choquet*: the same model and penalty with λ taken from (i) and no sign constraints, which isolates the effect of the constraints. (iii) *Additive monotone*: (i) with all interactions fixed at zero. (iv) *Choquistic regression, maximum likelihood*: the 2-additive choquistic model σ(γ(C_μ(x) − β)) with normalized μ, fitted without penalty. For the AUC comparison we computed its optimum with the reparametrized solver (Proposition 4); on the first two repetitions (10 folds per data set; 15 on POL252, where five further folds had been fitted in an interrupted earlier run of the same code) we also fitted it in its original normalized parametrization with SLSQP [[kraft1988]], as an independent implementation, to confirm that both reach the same optimum and to time the original formulation. (v) *LightGBM with monotone constraints* on all oriented features [[ke2017]] (300 trees, learning rate 0.03, 7 leaves, minimum 10 samples per leaf, row and column subsampling 0.8, "advanced" constraint method). (vi) *L1 logistic regression* on standardized features and (vii) *L1 logistic regression with all pairwise products*, the unconstrained interaction model closest to the Choquet layer, both with the penalty chosen by inner cross-validation.

**Statistics.** For each data set we compared (i) with each of (ii)–(vii) by the corrected resampled t-test of Nadeau and Bengio [[nadeau2003]] on the 50 fold-level AUC differences and controlled the false discovery rate over the six comparisons with the Benjamini–Hochberg procedure [[benjamini1995]]. We also report how many nodes of the unconstrained model violate (C), and how often the constraints of the monotone model are active.

## 6.3 Computational cost

We compared three solvers for the penalized convex problem with λ = 3·10⁻³ and γ = 10⁻³: the reparametrized bound-constrained L-BFGS-B, and the same problem in the original variables (m, m⁺, m⁻) with the p linear constraints m_i − Σ_j m⁻_ij ≥ 0, solved with the interior-point trust-region method of SciPy (trust-constr) [[byrd1999]] and with SLSQP [[kraft1988]]. For the unpenalized maximum-likelihood problem we compared L-BFGS-B in the reparametrized variables with choquistic regression in its original parametrization solved by SLSQP. Test problems were POL252 at item level (p = 25) and data simulated from sparse monotone 2-additive capacities with n = 500 and p ∈ {5, 10, 15, 20, 30, 40}. We report wall-clock time on one thread of a shared workstation (the minimum of three runs when a run took less than 5 s), the objective value and the smallest node margin of (C). A solver was not run at larger p once it exceeded 900 s.

## 6.4 Software

All computations used Python 3.12 with NumPy 2.5.2, SciPy 1.18.1, scikit-learn 1.9.0 and LightGBM 4.7.0. The code that reproduces every number, table and figure is available (Section "Code availability").

# 7. Results

{{RESULTS}}

# 8. Discussion

{{DISCUSSION}}

# 9. Conclusions

{{CONCLUSIONS}}

# Relationship to prior work

The screening model studied here was developed in the first author's doctoral thesis *[thesis reference to be completed]*, which reported the grid measurements of the Mamdani layer (Table 2, first and last rows) and a first version of the monotone Choquet learner on POL252. This paper adds the theory (Lemmas 1–2, Theorems 1–2, Propositions 1–4, Corollary 1), the certificate, the implication and stress-test analyses, the three external data sets, the comparison with choquistic regression and LightGBM, and the cost study. Companion papers use the same Polish cohort for different questions — whether flexible architectures outperform additive scores, whether labels and evaluation protocols can be trusted, and which items to ask *[companion manuscripts in preparation — update status before submission]*; no table in this paper repeats their results.

# Declarations

**Funding.** *[Funding statement, or: This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors — to be confirmed]*

**Declaration of competing interest.** The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper. *[to be confirmed]*

**Data availability.** POL252 is publicly available from Mendeley Data [[niedzwiecka_data]]. The UCI Parkinson's, Wisconsin breast-cancer (original) and Cleveland heart-disease data are publicly available from the UCI Machine Learning Repository. No new data were collected.

**Code availability.** The code is released as a self-contained archive (w2_code.zip) with a README describing how to reproduce every table and figure. *[public repository URL and DOI to be added]*

**Ethics.** Only publicly available, de-identified secondary data were analysed. *[Confirm with the authors' institution whether a formal exemption is required]*

**CRediT authorship contribution statement.** *[To be completed using the CRediT taxonomy]*

**Declaration of generative AI and AI-assisted technologies in the writing process.** During the preparation of this work the authors used an AI assistant (Claude, Anthropic) to draft code, run analyses and draft the text. The authors reviewed and edited all content and take full responsibility for the publication. *[authors to confirm or revise]*

# References

{REFERENCES}

# Figure legends

**Fig. 1** Two-layer screening model and where each result applies. Layer 1 maps each domain's deficit score and age-lag index to a domain risk; Layer 2 fuses the domain risks with a 2-additive Choquet integral inside a logistic link. Theorem 1 guarantees that Layer 1 is nondecreasing in both inputs, Theorem 2 that Layer 2 is nondecreasing in every risk, and Corollary 1 that the composite is nondecreasing in every input.

**Fig. 2** Domain risk of the Mamdani layer as a function of the deficit score d. (a) Ruspini partitions with maximum aggregation of rules sharing a consequent: the risk decreases on parts of the range. (b) Ruspini partitions with bounded-sum aggregation: the risk is nondecreasing. (c, d) Tails of the consequent-strength vector at LAG = 0.5 (w_2 + w_3 and w_3) and the total Σ_c w_c: with the maximum the total drops below one and the tails are not monotone; with the bounded sum the total is one and both tails are nondecreasing, as Lemma 1 states.

**Fig. 3** Smallest directional derivative of the defuzzifier map D along v_2 = e_2 − e_1 and v_3 = e_3 − e_2 over the probability simplex (finite differences on a grid, for display; the exact values come from Algorithm 1). (a) Consequent sets of the screening model: positive everywhere. (b) Skewed consequent sets (narrow LOW, wide HIGH): a region adjacent to the LOW–HIGH edge, bounded by the black zero contour, has negative derivative, so condition (D) fails and Theorem 1 cannot be applied.

**Fig. 4** Difference in cross-validated AUC between the monotone 2-additive Choquet model and each comparator, with Nadeau–Bengio 95% confidence intervals (5-fold cross-validation repeated 10 times). Orange: significant after Benjamini–Hochberg correction within the data set; grey: not significant.

**Fig. 5** Wall-clock time against the number of parameters on simulated data. (a) Penalized convex problem: reparametrized bound-constrained L-BFGS-B versus the same problem with linear constraints solved by trust-constr and SLSQP. (b) Unpenalized maximum likelihood: reparametrized L-BFGS-B versus choquistic regression in its original normalized parametrization solved by SLSQP. Missing points: solver not run after exceeding the 900 s budget at a smaller size.
