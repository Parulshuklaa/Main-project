# InspectIQ: interview preparation

## A defensible 60-second explanation

I built a Flask quality-inspection workbench connecting image classification with source-backed inspection guidance. I implemented a trainable convolutional network in NumPy so I could explain the forward pass and gradient flow. I evaluated it on an independent synthetic test seed and made its real-world limitations visible. The text pipeline uses hybrid sparse retrieval, session-isolated document uploads and optional LLM generation. My next milestone is real-data validation and comparison against transfer learning and an anomaly detection baseline.

Do not claim you trained on industrial data, deployed a GPU model, implemented Grad-CAM, or demonstrated production accuracy. Without an LLM key, say the live application demonstrates retrieval and the code supports generative RAG.

## Questions you should be able to answer

1. **Why this problem?** It joins perception with operational knowledge; a predicted defect needs a documented human review workflow.
2. **Why Flask?** Python model integration, a small deployment footprint and simple HTTP routing. Discuss worker/thread limits and blocking LLM calls.
3. **What does convolution learn?** Local filters shared across locations; gradient descent updates each filter using all overlapping image patches.
4. **CNN architecture?** 32×32 grayscale input → 12 valid 3×3 filters → 30×30×12 activations → ReLU → mean/max pooling → 24 features → 3 logits → softmax.
5. **Loss and backprop?** Cross-entropy; derivative of logits is `(probability - one_hot)/batch_size`. Propagate through dense weights, pooling, ReLU and convolution patch contractions. At tied maxima this implementation distributes gradient evenly.
6. **Why NumPy rather than PyTorch?** Educational transparency and small CPU deployment. PyTorch is better for larger models, GPU acceleration and mature optimization tooling.
7. **Why both mean and max pooling?** Mean captures broad texture; max emphasizes small localized patterns. Test this choice through an ablation before claiming improvement.
8. **Why is 100% not impressive here?** The synthetic classes are easy and train/test share the generator. Independent seeds do not eliminate distribution similarity.
9. **How would you improve generalization?** Real consented/licensed data, varied lighting and camera geometry, grouped splits, augmentation, transfer learning, representative negatives and OOD evaluation.
10. **What is the activation map?** Maximum ReLU response across channels, normalized for display; it does not explain a specific class or establish causality.
11. **What is RAG?** Retrieve relevant passages, supply them as evidence to a generator, return an answer with provenance. The default key-free mode only retrieves excerpts.
12. **Why word and character TF-IDF?** Word n-grams capture terms; character n-grams tolerate spelling and morphology. Weights .7/.3 are heuristic, not optimized on a benchmark.
13. **Why a similarity threshold?** To abstain when support is weak. .09 is a demo heuristic; tune against labeled relevant/irrelevant queries.
14. **What does chunk overlap do?** Reduces loss of context at boundaries, at the cost of duplicate hits and increased index size. Here 150 words with a 120-word stride.
15. **How do you prevent hallucination?** Restricted evidence prompt, abstention and valid source IDs. These do not prove that a cited statement is supported; groundedness evaluation is still needed.
16. **How do you handle prompt injection?** Separate trusted instructions from untrusted document text, no agent tools, and explicit instructions not to follow evidence commands. Further adversarial tests are needed.
17. **How are uploads isolated?** Random signed cookie session identifier, parameterized queries scoped by owner. It is not user authentication and is not appropriate for sensitive production documents.
18. **What fails on free hosting?** Cold starts, ephemeral uploads, CPU constraints, provider timeouts and a single process. UI handles errors and retrieval fallback.
19. **How would you scale?** Postgres, durable object store, workers for PDF/OCR ingestion, dedicated model servers, retrieval index caching, tenant-aware authentication and gateway rate limiting.
20. **What are meaningful tests?** Image predictions on generator examples, malformed input, CSRF rejection, upload isolation, document clearing and retrieval abstention. Add real-image held-out benchmarks and RAG evaluation separately.

## Hinglish: kaise banaya aur challenges

Pehle ek coherent problem choose ki: defect detect karne ke baad user ko inspection manual ka relevant evidence bhi chahiye. Frontend se image Flask API ko jaati hai; grayscale resize ke baad CNN class probabilities deta hai. Convolution aur classifier dono train hote hain. Activation map ko explanation ka substitute nahi bola.

Free CPU deployment ke liye heavyweight deep learning framework ki jagah NumPy implementation rakhi. Isse small model deploy ho sakta hai aur gradients samajhna easy hai, lekin real dataset par accuracy prove nahi hoti. Synthetic evaluation ka 100% score UI mein clear limitation ke saath diya.

Document ko chunks mein split karke word aur character TF-IDF se retrieve kiya. API key na ho toh exact evidence excerpts dikhte hain. Key ho toh retrieved context LLM ko bheja jaata hai. Invalid citation IDs ya provider failure par retrieval fallback hota hai. Isse generation ki factual correctness automatically guarantee nahi hoti.

Deployment mein major constraints memory, cold start aur temporary filesystem hain. SQLite data restart par lost ho sakta hai. Scale karne ke liye authentication, Postgres, durable storage, background jobs, separate inference workers aur retrieval evaluation add karne honge.

## Next milestones worth putting on your resume after completing them

- Benchmark a real inspection dataset with leakage-resistant splits.
- Add transfer learning baseline, calibration and per-class error analysis.
- Label a retrieval evaluation set and measure recall@k and answer groundedness.
- Add authenticated users and durable storage.
- Record actual load-test latency and throughput; do not invent improvement percentages.
