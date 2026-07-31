---
layout: post
title: "understanding kimi k3's attention"
date: 2026-07-31
---

## introduction

The release of Kimi K3 has made waves in the ML community and the media (for reasons beyond just the model itself of course). However, amidst this discourse, we perhaps miss out on the, for lack of better word, coolness of Kimi and it’s design. When reading the newly released tech report from Moonshot AI<sup>1</sup>, I was particularly drawn to the unique choices with regards to attention mechanisms that are utilized to make K3 a frontier level model with up to 1M tokens of context. In this post I’ll dive into the multiple innovations and tricks used by the Kimi team specifically with respect to attention.

## some context on attention variants

To start I think it's important to understand what recent trends in open models have led up to the architecture we see in Kimi K3. 

Attention is a notoriously inefficient calculation, with the traditional attention á la “Attention is All You Need”<sup>2</sup> being quadratic in its cost. To understand that cost we need only consider that in a sequence of N tokens, each token attends to every other token resulting in a time complexity of the order of N^2. Since its inception, however, researchers have put forward countless alternative methods to “improve” attention by trying to make it faster (e.g Linear Attention)<sup>3</sup> and by reducing the memory footprint (e.g Group Query Attention)<sup>4</sup>. 

K3 follows a newer trend in frontier open models, starting from MiniMax’s M1 which released last year in 2025<sup>5</sup>, in using a hybrid of attention variants to enable efficient and effective performance at massive scale. The particular flavors of attention used in K3 are Kimi Delta Attention (KDA) which builds upon Kimi Linear<sup>6</sup>, and Gated Multi-Head Latent Attention (Gated MLA) which builds on the original MLA introduced by Deepseek<sup>7</sup>. Kimi uses these two in a 3:1 ratio and also utilizes Attention Residuals<sup>8</sup> (more on that later).

<!-- VIZ:lineage -->
<figure class="viz" data-viz="lineage">
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 860 380" role="img" aria-labelledby="lineage-title"><title id="lineage-title">Two attention tracks converge in Kimi K3</title><style>.l{font-family:system-ui,sans-serif;fill:currentColor}.c{fill:none;stroke:currentColor;stroke-opacity:.35;stroke-width:2}.a{fill:none;stroke:currentColor;stroke-opacity:.45;stroke-width:3}.t{fill:#0F8A80}.i{fill:#3B3486}.n{font-size:19px;font-weight:700}.d{font-size:14px;fill-opacity:.7}.s{font-size:13px;font-weight:700;letter-spacing:1px}</style><g class="l"><text x="28" y="35" class="s">TWO ATTENTION TRACKS</text><text x="28" y="90" class="s t">LINEAR / RECURRENT</text><rect class="c" x="28" y="108" width="185" height="78" rx="12"/><text class="n" x="48" y="140">Kimi Linear</text><text class="d" x="48" y="166">recurrent state</text><path class="a" d="M213 147h75"/><rect class="c" x="300" y="108" width="205" height="78" rx="12"/><text class="n" x="320" y="140">KDA</text><text class="d" x="320" y="166">delta-rule memory</text><text x="28" y="250" class="s i">LATENT / GLOBAL</text><rect class="c" x="28" y="268" width="185" height="78" rx="12"/><text class="n" x="48" y="300">DeepSeek MLA</text><text class="d" x="48" y="326">compressed KV latent</text><path class="a" d="M213 307h75"/><rect class="c" x="300" y="268" width="205" height="78" rx="12"/><text class="n" x="320" y="300">Gated MLA</text><text class="d" x="320" y="326">input-dependent gate</text><path class="a" d="M505 147c50 0 45 45 95 45M505 307c50 0 45-45 95-45"/><rect x="600" y="170" width="230" height="110" rx="16" fill="none" stroke="#E0912B" stroke-width="2"/><text class="n" x="622" y="207">Kimi K3</text><text class="d" x="622" y="237">3 × KDA : 1 × Gated MLA</text><text class="d" x="622" y="260">+ attention residuals</text></g></svg>
  <figcaption>Attention-variant lineage: two efficient attention tracks converge in K3's hybrid recipe.</figcaption>
</figure>

## kimi delta attention

Kimi Delta Attention was a big motivator for me to make this post. It’s certainly not instantly intuitive at least not to a mere mortal like myself, but digging into the linear algebra we can see how powerful it is as the primary attention utilized in K3. KDA is a SSM-style attention calculation so it is helpful to liken it to Mamba<sup>9</sup>, where we have a state equation and observation equation and each is updated at each time step of recurrence. 

The recurrence, here, is over each token index so when looking at the update equations the tradition state-space axis of time is actually just the tokens in a sequence, meaning that the attention calculations performed are done, as you might guess, in linear time with respect to the number of tokens processed. For clarity here is an appendix of each term used in the calculations:

| Symbol | Meaning |
| --- | --- |
| $\mathbf{x}_t \in \mathbb{R}^{d}$ | layer input at token $t$ ($d$ = model dimension) |
| $\mathbf{q}_t,\ \mathbf{k}_t \in \mathbb{R}^{d_k}$ | query and key: $\operatorname{L2Norm}\!\big(\operatorname{Swish}(\operatorname{ShortConv}(\mathbf{W}_{q/k}\,\mathbf{x}_t))\big)$ |
| $\mathbf{v}_t \in \mathbb{R}^{d_v}$ | value: $\operatorname{Swish}(\operatorname{ShortConv}(\mathbf{W}_v\,\mathbf{x}_t))$ |
| $\mathbf{S}_t \in \mathbb{R}^{d_k \times d_v}$ | recurrent state — the associative memory |
| $\tilde{\mathbf{o}}_t \in \mathbb{R}^{d_v}$ | ungated readout, $\tilde{\mathbf{o}}_t = \mathbf{S}_t^\top \mathbf{q}_t$ |
| $\boldsymbol{\alpha}_t \in (0,1)^{d_k}$ | channel-wise retention (forget gate), $\boldsymbol{\alpha}_t = \exp(g_t)$ |
| $\beta_t \in (0,1)$ | write strength / step size: $\operatorname{Sigmoid}(\mathbf{W}_\beta\,\mathbf{x}_t)$ |
| $\operatorname{Diag}(\boldsymbol{\alpha}_t)$ | diagonal matrix formed from $\boldsymbol{\alpha}_t$ |
| $g_t = \log \boldsymbol{\alpha}_t$ | per-channel log-decay (bounding shown below) |
| $z_t$ | decay logit: $\mathbf{W}^{\uparrow}_{\alpha}\mathbf{W}^{\downarrow}_{\alpha}\,\mathbf{x}_t + \mathbf{b}_\alpha$ |
| $A_h$ | learnable per-head log-scale |
| $g_{\min} = -5$ | lower bound on the log-decay |
| $\mathbf{W}_g,\ \mathbf{W}_o$ | gate and output projection matrices |
| $\odot$ | elementwise (Hadamard) product |
| $\mathbf{y}_t$ | gated layer output |

The state update equation in its full form is:

$$\mathbf{S}_t = \left(\mathbf{I} - \beta_t\,\mathbf{k}_t \mathbf{k}_t^\top\right)\operatorname{Diag}(\boldsymbol{\alpha}_t)\,\mathbf{S}_{t-1} \;+\; \beta_t\,\mathbf{k}_t \mathbf{v}_t^\top$$

and the paired observation (readout) equation, which produces the layer's ungated output from the state, is:

$$\tilde{\mathbf{o}}_t = \mathbf{S}_t^\top \mathbf{q}_t$$

In this equation, I think its important to decompose the various affine components in order to not only understand their individual purpose, but also the progression of linear attention architectures. First we consider the shrinkage component $$\operatorname{Diag}(\boldsymbol{\alpha}_t)$$ which serves a simple purpose in decaying the previous state’s contribution to the current state. We use this to reduce the correlation of state tensors and allow for the memory writes at token t to be fairly represented.

Now the delta component<sup>10</sup>: $$\left(\mathbf{I} - \beta_t\,\mathbf{k}_t \mathbf{k}_t^\top\right)$$ which at face value to me seems to be more of the same. After all, the sight of transforming $$\operatorname{Diag}(\boldsymbol{\alpha}_t)\,\mathbf{S}_{t-1}$$ seems like its continuing to reduce the “impact” of the previous state on the current one. However, to better understand the role of this component we can look at the following equivalent equation:

$$\mathbf{S}_t = \operatorname{Diag}(\boldsymbol{\alpha}_t)\,\mathbf{S}_{t-1} \;+\; \beta_t\,\mathbf{k}_t\left(\mathbf{v}_t - \big(\operatorname{Diag}(\boldsymbol{\alpha}_t)\,\mathbf{S}_{t-1}\big)^{\!\top}\mathbf{k}_t\right)^{\!\top}$$

Now consider regression problem

$$\mathcal{L}_t(\mathbf{S}) = \tfrac{1}{2}\left\|\,\mathbf{S}^\top \mathbf{k}_t - \mathbf{v}_t\,\right\|_2^2$$

which we can find the gradient of with some simple calculus:

$$\nabla_{\mathbf{S}}\,\mathcal{L}_t = \mathbf{k}_t\left(\mathbf{S}^\top \mathbf{k}_t - \mathbf{v}_t\right)^{\!\top}$$

Taking a single gradient-descent step of size $$\beta_t$$ from the decayed state $$\operatorname{Diag}(\boldsymbol{\alpha}_t)\,\mathbf{S}_{t-1}$$ then recovers exactly the rewritten update above:

$$\mathbf{S}_t = \operatorname{Diag}(\boldsymbol{\alpha}_t)\,\mathbf{S}_{t-1} - \beta_t\,\nabla_{\mathbf{S}}\,\mathcal{L}_t$$

We can now see that the update rule looks like our old friend gradient descent, and this provides the power of the delta update. We can imagine this update as getting our current $$\mathbf{S}_t$$ to be taking a decayed version of the previous state $$\mathbf{S}_{t-1}$$ and updating it towards being a “better” operator for approximating $$\mathbf{v}_t$$. Based on how strong our $$\beta_t$$ is, our memory write for $$\mathbf{S}_t$$ steers the information in memory more towards associative recall.

<!-- VIZ:kda-memory -->
<figure class="viz">
  <iframe class="viz-frame" data-viz="kda-memory"
          src="{{ '/assets/viz/kda-memory.html' | relative_url }}"
          title="KDA state as an editable associative memory"
          loading="lazy" height="640"></iframe>
  <figcaption>The KDA state, edited one token at a time — forget, delete, write, read. Token 3 reuses token 1's key, so watch the delete step scrub the old value before the new one is written.</figcaption>
</figure>

One of the small tweaks the Kimi team made to improve KDA was to also introduce a lower bound on $$\boldsymbol{\alpha}_t$$.

$$g_t = g_{\min}\,\operatorname{Sigmoid}\!\left(e^{A_h}\,z_t\right) \in (g_{\min},\,0), \qquad \boldsymbol{\alpha}_t = \exp(g_t) \in \big(e^{g_{\min}},\,1\big)$$

The introduction of the $$g_{\min}$$ term allows for a limit on the amount of decay applied, and this provides an example of an architectural innovation more driven by hardware constraints than pure algorithmic improvement. The chosen $$g_{\min}$$ of -5 ensures that the log decay over a single token change is bounded below at $$-5$$, so the per-step retention $$\boldsymbol{\alpha}_t$$ stays above $$e^{-5}$$; accumulated over a 16-token tile the log decay stays above $$-80$$, i.e. the cumulative retention stays above $$e^{-80}$$. A problem with unbounded log decay is that since it grows with token position (recall our states here are built recurrently), we get tiny multiplications that result in very small numbers which require very high precision to track.

Kimi Linear<sup>6</sup> attempted to solve this problem by using 16 token increments as “tiles” in which relative log decay could be calculated, but even with this optimization explicit tensor multiplication calculations are required. With the new bounding, the decay is guaranteed to stay within what is known as the “bf16 dynamic range”. The dynamic range corresponds to values that the bf16 format can represent losslessly, and since decays all are within the dynamic range the Kimi team was able to entirely implement the related operations as efficient TensorCore matrix multiplications which operate on “blocks” of data as opposed to acting on individual positions.

<!-- VIZ:bounded-decay -->
<figure class="viz" data-viz="bounded-decay">
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 860 390" role="img" aria-labelledby="decay-title"><title id="decay-title">Bounded log decay in KDA</title><style>.x{font-family:system-ui,sans-serif;fill:currentColor}.a{stroke:currentColor;stroke-opacity:.35;stroke-width:2}.g{stroke:currentColor;stroke-opacity:.13}.c{fill:none;stroke:#0F8A80;stroke-width:4}.b{stroke:#E0912B;stroke-width:2;stroke-dasharray:7 5}.h{font-size:14px;font-weight:700;letter-spacing:1px}.d{font-size:15px;fill-opacity:.72}.m{font:17px ui-monospace,monospace}</style><g class="x"><text x="30" y="36" class="h">BOUNDED CHANNEL DECAY</text><text x="30" y="62" class="d">A sigmoid constrains log-decay before it becomes retention.</text><rect x="30" y="90" width="500" height="265" rx="14" fill="none" class="a"/><path class="a" d="M94 126v190h390"/><path class="g" d="M94 144h390M94 238h390M94 318h390"/><text x="52" y="150" class="d">0</text><text x="40" y="244" class="d">−2.5</text><text x="48" y="324" class="d">−5</text><path class="b" d="M94 318h390"/><path class="c" d="M108 315c95 0 130-11 169-48 43-41 67-117 195-122"/><text x="150" y="119" class="m">gₜ = −5 · sigmoid(eᴬʰzₜ)</text><rect x="560" y="90" width="270" height="265" rx="14" fill="none" class="a"/><text x="588" y="132" class="h">THE GUARANTEE</text><text x="588" y="181" class="m">−5 &lt; gₜ &lt; 0</text><text x="588" y="209" class="d">log-decay per token</text><text x="588" y="257" class="m">e⁻⁵ &lt; αₜ &lt; 1</text><text x="588" y="285" class="d">per-step retention</text><text x="588" y="326" class="d">16-token tile: g ≥ −80</text></g></svg>
  <figcaption>Bounding $g_t$ keeps decay within the bf16 dynamic range while retaining channel-wise control.</figcaption>
</figure>

The final output

$$\mathbf{y}_t = \mathbf{W}_o\left[\operatorname{Sigmoid}(\mathbf{W}_g\,\mathbf{x}_t) \odot \operatorname{RMSNorm}(\tilde{\mathbf{o}}_t)\right]$$

utilizes a gating weight $$\mathbf{W}_g$$ which improves expressiveness and can counteract attention sinks<sup>11</sup>. One of the unique parts of KDA which also helps explain the use of NoPE in Kimi K3 is that it enables implicit encoding of positional information. Fundamentally each $$\mathbf{S}_t$$ is a function of previous token states and as such it allows for positional information and relationships to be retained.

## gated multi-head latent attention

To compliment the linear attention from KDA, the K3 recipe also includes global attention layers in the form of Gated Multi-Head Latent Attention. The name is a mouthful but we can start with the general problem that the researchers hoped to solve using Gated MLA. For long context agentic tasks (what models like Kimi K3 are designed to support), the KV cache grows large, and this creates a massive memory bottleneck.

For clarity here is a new appendix for the variables and terms used:

| Symbol | Meaning |
| --- | --- |
| $n_h$ | number of attention heads; head index $i \in \{1,\dots,n_h\}$ |
| $d_h,\ d_h^v$ | per-head key/query and value dimensions |
| $d_c \ll n_h d_h$ | KV latent (compression) dimension |
| $\mathbf{c}^{KV}_t \in \mathbb{R}^{d_c}$ | cached KV latent — the only per-token tensor stored |
| $\mathbf{W}^{DKV},\ \mathbf{W}^{UK},\ \mathbf{W}^{UV}$ | KV down-projection, and key / value up-projections |
| $\mathbf{k}^{C}_t \in \mathbb{R}^{n_h d_h}$ | reconstructed keys (all heads), split head-wise into $\mathbf{k}^{C}_{t,i} \in \mathbb{R}^{d_h}$ |
| $\mathbf{v}^{C}_t \in \mathbb{R}^{n_h d_h^v}$ | reconstructed values, split into $\mathbf{v}^{C}_{t,i} \in \mathbb{R}^{d_h^v}$ |
| $\mathbf{q}_{t,i} \in \mathbb{R}^{d_h}$ | per-head query at token $t$ (from the query projection, split by head) |
| $\bar{\mathbf{o}}_{t,i}$ | per-head attention output; concatenated to $\bar{\mathbf{o}}_t = [\bar{\mathbf{o}}_{t,1};\dots;\bar{\mathbf{o}}_{t,n_h}]$ |
| $\tau \le t$ | past and current token positions attended over |
| $\mathbf{W}_g,\ \mathbf{W}_o$ | input-dependent gate and output projections |

Compression along the sequence dimension is not really a feasible option in this case. Sequence length will always vary and as such any sort of compression (read: matrix multiplication) would need operators with a variable sized dimension. A far more within reach method is to compress the keys and values themselves on a per-token basis, and this is the core idea that lead to MLA in Deepseek-V2<sup>7</sup>.

$$\mathbf{c}^{KV}_t = \mathbf{W}^{DKV}\mathbf{x}_t \in \mathbb{R}^{d_c}, \qquad \mathbf{k}^{C}_t = \mathbf{W}^{UK}\mathbf{c}^{KV}_t \in \mathbb{R}^{n_h d_h}, \qquad \mathbf{v}^{C}_t = \mathbf{W}^{UV}\mathbf{c}^{KV}_t \in \mathbb{R}^{n_h d_h^v}$$

Critically this compression is also along the attention head axis, and as such MLA can reduce the space complexity of per-token KV cache memory from $2\,n_h d_h$ to $d_c$. The Kimi team innovated a bit more with their implementation of MLA in K3, adding a gating mechanism:

<!-- VIZ:kv-footprint -->
<figure class="viz">
  <iframe class="viz-frame" data-viz="kv-footprint"
          src="{{ '/assets/viz/kv-footprint.html' | relative_url }}"
          title="Per-token KV-cache footprint comparison"
          loading="lazy" height="560"></iframe>
  <figcaption>MLA stores a compact per-token KV latent rather than separate keys and values for every attention head.</figcaption>
</figure>

$$\mathbf{y}_t = \mathbf{W}_o\left[\operatorname{Sigmoid}(\mathbf{W}_g\,\mathbf{x}_t) \odot \bar{\mathbf{o}}_t\right]$$

where each head $i$ runs an ordinary causal softmax attention over its slice of the reconstructed keys and values, and the heads are concatenated to form the ungated output $\bar{\mathbf{o}}_t$:

$$\bar{\mathbf{o}}_{t,i} = \sum_{\tau \le t}\operatorname{softmax}_{\tau \le t}\!\left(\frac{\mathbf{q}_{t,i}^\top \mathbf{k}^{C}_{\tau,i}}{\sqrt{d_h}}\right)\mathbf{v}^{C}_{\tau,i}, \qquad \bar{\mathbf{o}}_t = \big[\bar{\mathbf{o}}_{t,1};\ \dots;\ \bar{\mathbf{o}}_{t,n_h}\big]$$

Output gating is used liberally throughout the Kimi K3 attention variants, and this is in line with literature coming from fellow open model innovators AliBaba who detailed the benefits of gating for introducing sparsity and counteracting attention sinks<sup>11</sup>.

<!-- VIZ:division-of-labor -->
<figure class="viz" data-viz="division-of-labor">
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 860 400" role="img" aria-labelledby="labor-title"><title id="labor-title">KDA and Gated MLA divide the work in Kimi K3</title><style>.x{font-family:system-ui,sans-serif;fill:currentColor}.c{fill:none;stroke:currentColor;stroke-opacity:.32;stroke-width:2}.w{fill:none;stroke:currentColor;stroke-opacity:.45;stroke-width:3}.h{font-size:14px;font-weight:700;letter-spacing:1px}.n{font-size:22px;font-weight:700}.o{font-size:16px;font-weight:700}.d{font-size:15px;fill-opacity:.72}.m{font:14px ui-monospace,monospace}.t{fill:#0F8A80}.i{fill:#3B3486}</style><g class="x"><text x="30" y="35" class="h">COMPLEMENTARY MIXERS IN KIMI K3</text><rect x="30" y="78" width="360" height="235" rx="16" class="c"/><circle cx="62" cy="112" r="9" class="t"/><text x="82" y="120" class="n">KDA</text><text x="62" y="153" class="d">recurrent associative memory</text><text x="62" y="193" class="h t">WHERE / WHEN</text><text x="62" y="236" class="m">Sₜ ← decay(Sₜ₋₁) + delta write</text><text x="62" y="272" class="d">Carries order effects without a</text><text x="62" y="294" class="d">growing KV cache.</text><rect x="470" y="78" width="360" height="235" rx="16" class="c"/><circle cx="502" cy="112" r="9" class="i"/><text x="522" y="120" class="n">Gated MLA</text><text x="502" y="153" class="d">compressed global attention</text><text x="502" y="193" class="h i">WHAT</text><text x="502" y="236" class="m">cᴷⱽ → reconstructed K, V</text><text x="502" y="272" class="d">Recalls content across the context</text><text x="502" y="294" class="d">from a compact KV latent.</text><path class="w" d="M210 313v18h206M650 313v18H444"/><rect x="350" y="338" width="160" height="46" rx="23" fill="#E0912B" fill-opacity=".2" stroke="#E0912B"/><text x="385" y="367" class="o">gated output</text></g></svg>
  <figcaption>KDA carries where and when information occurs; Gated MLA retrieves what is relevant across the context.</figcaption>
</figure>

## attention residuals

The Kimi team also uses a unique residual connection method (of course reinforcing that at the end of the day everything really is just ResNet<sup>13</sup>). The formulation here takes on residual connections at the attention level and uses the familiar Q-K-V trio as a means of controlling the residual information that is passed.

$$\mathbf{q}_l = \mathbf{w}_l \in \mathbb{R}^{d}, \qquad \mathbf{k}_i = \mathbf{v}_i = \begin{cases} \mathbf{h}_1 & i = 0 \\[2pt] f_i(\mathbf{h}_i) & 1 \le i \le l-1 \end{cases}$$

These are used to compute:

$$\alpha_{i\to l} = \frac{\phi(\mathbf{q}_l,\ \mathbf{k}_i)}{\sum_{j=0}^{l-1}\phi(\mathbf{q}_l,\ \mathbf{k}_j)}, \qquad \mathbf{h}_l = \sum_{i=0}^{l-1}\alpha_{i\to l}\,\mathbf{v}_i, \qquad \phi(\mathbf{q},\mathbf{k}) = \exp\!\left(\mathbf{q}^\top\operatorname{RMSNorm}(\mathbf{k})\right)$$

I like to think of the value of $\alpha_{i\to l}$ for each given i as the fractional “learned importance” of the corresponding layer and we see that this importance is used to weight the contribution of each layer’s values in the input to the current layer. This adds the dimension of “attention over layers” which is certainly a compelling narrative although even considering it simply as a learned residual connection method like Deepseek’s “hyperconnections”<sup>12</sup> the value add is clear.

## wrapping up

The attention recipe used to create Kimi provided me a great opportunity to reinforce my own understanding of different attention variants as well as the different ways in which architecture can be optimized and co-designed with the model’s intended use as well as the the way in which it is trained. I hope this post is helpful to you and of course all credit goes to the Kimi team for both an incredible effort in frontier level open models as well as one of the clearest and most insightful technical reports I’ve gotten to read.

## references

1.  [Kimi K3: Open Frontier Intelligence](https://arxiv.org/pdf/2607.24653) — Moonshot AI / Kimi Team (2026).
2.  [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al. (2017).
3.  [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236) — Katharopoulos et al. (2020).
4.  [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) — Ainslie et al. (2023).
5.  [MiniMax-M1](https://arxiv.org/abs/2506.13585) — MiniMax (2025).
6.  [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692) — Kimi Team (2025).
7.  [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434) — DeepSeek-AI (2024).
8.  [Attention Residuals](https://arxiv.org/abs/2603.15031) — Kimi Team (2026).
9.  [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) — Gu & Dao (2023); see also Mamba-2, Dao & Gu (2024).
10.  [Parallelizing Linear Transformers with the Delta Rule over Sequence Length](https://arxiv.org/abs/2406.06484) — Yang et al. (2024).
11.  [Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free](https://arxiv.org/abs/2505.06708) — Qiu et al. (2025).
12.  [Hyper-Connections](https://arxiv.org/abs/2409.19606) — Zhu et al. (2024).
13.  [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) — He et al. (2015).
