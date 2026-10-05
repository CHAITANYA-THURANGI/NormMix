# Research Notes: Telugu-English Code-Mixed Text Normalization

This document compiles research notes, related work, resources, and future directions for the Telugu-English code-mixed text normalization project.

## 1. Key Research Papers on Code-Mixed Text Normalization

*   **"Normalization of Code-Mixed Text"**: General overview of the challenges in normalizing code-mixed text, focusing on spelling variations, phonetic typing, and structural mixing.
*   **"Language Identification and Normalization in Code-Mixed Data"**: Discusses pipelined approaches where language identification precedes normalization.
*   **"Deep Learning Models for Code-Mixed Sentiment Analysis"**: While focused on sentiment, these papers often contain valuable preprocessing and normalization strategies for code-mixed input, utilizing subword tokenization (BPE, SentencePiece).
*   **"Transliteration and Normalization for Indian Languages"**: Research specifically targeting the phonetic transliteration (e.g., typing Telugu in Latin script) and the associated spelling variations. Focus on sequence-to-sequence models for this task.

*(Note: Specific citations to be added as literature review progresses).*

## 2. Telugu NLP Resources

*   **Corpora:**
    *   **AI4Bharat Samanantar:** Large-scale parallel corpora for Indian languages, useful for pre-training or transfer learning.
    *   **Telugu Wikipedia Dump:** Useful for creating monolingual language models.
    *   **Social Media Scrapes (Twitter, YouTube comments):** Essential for obtaining authentic code-mixed and non-standard text for training and evaluation.
*   **Tools:**
    *   **IndicNLP Library:** Tools for tokenization, normalization, and script conversion for Indian languages.
    *   **iNLTK:** Natural Language Toolkit for Indic Languages.
*   **Pre-trained Models:**
    *   **MuRIL (Multilingual Representations for Indian Languages):** BERT model pre-trained on Indian languages, including Telugu and transliterated text.
    *   **IndicBERT:** Another strong baseline for multilingual tasks in Indian languages.

## 3. Related Work Comparison

| Approach | Strengths | Weaknesses | Relevance to NormMix |
| :--- | :--- | :--- | :--- |
| **Rule-based / Dictionary lookup** | High precision for known words, fast. | Low recall, fails on novel variations and spelling errors. | Can be used as a fast fallback or for initial bootstrapping. |
| **Statistical Machine Translation (SMT)** | Handles character-level mappings well. | Requires aligned training data, struggles with context. | Less relevant given the dominance of Neural models. |
| **Seq2Seq with Attention (RNN/LSTM)** | Good at character/subword transformations, captures context. | Slower training, may struggle with long dependencies. | Strong baseline model. |
| **Transformer-based (e.g., mBART, ByT5)** | State-of-the-art performance, handles context and cross-lingual signals excellently. | Computationally expensive, requires large datasets. | Primary focus for NormMix architecture, specifically exploring character-level or subword models. |

## 4. Open Research Questions

1.  **Handling Ambiguity:** How to effectively resolve ambiguous transliterations (e.g., where a Latin spelling could map to multiple Telugu words depending on context)?
2.  **Zero-shot/Few-shot Normalization:** Can large multilingual language models (LLMs) perform code-mixed normalization effectively with minimal task-specific training data?
3.  **Evaluation Metrics:** Beyond simple Word Error Rate (WER) or BLEU, what are better metrics for evaluating the semantic preservation of normalization in code-mixed contexts?
4.  **Real-time On-device Inference:** How to compress state-of-the-art Transformer models for efficient on-device normalization (e.g., mobile keyboards) without significant loss in accuracy?

## 5. Potential Conference Venues for Publication

If the project yields novel architectural improvements or a significant new dataset, consider submitting to:

*   **ACL (Association for Computational Linguistics):** Top tier venue for all NLP research.
*   **EMNLP (Empirical Methods in Natural Language Processing):** Strong focus on empirical results and novel applications.
*   **NAACL (North American Chapter of the ACL):** Similar to ACL and EMNLP.
*   **COLING (International Conference on Computational Linguistics):** Broad coverage of NLP topics.
*   **LREC (Language Resources and Evaluation Conference):** Excellent venue if a new, substantial dataset for Telugu-English code-mixing is created and released.
*   **Workshops:** Look for specific workshops attached to these major conferences, such as:
    *   Workshop on Computational Approaches to Linguistic Code-Switching (CALCS).
    *   Workshops focusing on low-resource or Indian language NLP.
