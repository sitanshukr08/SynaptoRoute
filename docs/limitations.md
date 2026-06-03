# Known Limitations

SynaptoRoute is highly optimized for performance and concurrency, but like any semantic system based on dense vector representations, it operates within strict structural and algorithmic limits. 

This document outlines the known failure modes of the system based on empirical adversarial testing.

---

## 1. Directional Semantics

Dense embeddings (like `BGE-small`) convert text into high-dimensional space based on contextual distribution. They are exceptionally good at identifying topic, but struggle heavily with logical direction and polarity.

**SynaptoRoute natively struggles with:**

### Double Negation
- **Input:** *"I do not want to not cancel this"*
- **Result:** Will likely map incorrectly to an opposite intent or fail thresholding.

### Negation / Logical Contradictions
- **Input:** *"Book a flight but under no circumstances charge my credit card"*
- **Result:** The presence of strong domain keywords ("Book", "flight", "credit card") often overpowers the logical negation ("under no circumstances"). The router may confidently assign this to `book_flight`.

## 2. Mixed Intent & Multi-Action Queries

SynaptoRoute operates under a strict Single-Intent Assumption (returning Top-1 or Top-K for a single semantic block).

- **Input:** *"Cancel my flight to Paris and order a taxi to the hotel instead"*
- **Result:** The embedding averages the vectors of "cancel flight" and "order taxi". It will typically route to whichever half of the sentence had structurally stronger keywords, completely discarding the second intent.
- **Solution:** You must use an upstream chunking mechanism or a downstream LLM to split multi-action prompts before passing them to the router.

## 3. Threshold Calibration

Because dense vector spaces are naturally clustered, queries that are completely Out-Of-Distribution (OOD) will still trigger non-zero (and sometimes relatively high) cosine similarities simply because they share language artifacts.

- **The Problem:** A fixed global threshold (e.g., `0.60`) is a blunt instrument. It will either accidentally filter out valid niche queries (False OOD) or allow completely irrelevant queries to map to a random intent (Failed Rejection).
- **The Solution:** Pure vector routers cannot reliably say "I don't know" without aggressive per-route threshold calibration.

## 4. Hardware Limitations (Encoder Lock)

While SynaptoRoute manages its Faiss index in memory perfectly, the underlying `FastEmbed` / `ONNX` runtime is incredibly CPU-hungry.
- **The Problem:** Supplying 5,000 requests per second to SynaptoRoute will queue successfully, but if your CPU cannot process ONNX inferences fast enough, the async queue will eventually hit its `10,000` limit and start shedding load.
- **The Solution:** We mitigate this heavily via the asynchronous batch worker (which groups encode requests), but raw compute is still a hard bottleneck for the embedding phase.

---

## Recommended Fallbacks

For use cases where the limitations above are unacceptable, you should employ:

1. **LLM Verification:** Use SynaptoRoute to cheaply identify the correct subsystem, but force that subsystem's LLM to double-check the raw prompt for negations before executing destructive actions.
2. **Explicit Workflows:** For multi-intent chaining, rely on orchestration frameworks (like LangChain or LangGraph) layered on top of SynaptoRoute.
3. **Cross-Encoder Reranking (Delivered in v0.3.0):** We built an optional reranker pipeline to apply cross-attention models to the Top-K candidates. Benchmarks show this resolves keyword traps, but still struggles with deep logical negation.
