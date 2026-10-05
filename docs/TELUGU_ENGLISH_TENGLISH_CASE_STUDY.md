# Evolution and Current Landscape of Telugu–English–Tenglish Translation: A Study of Translation, Transliteration, Code-Mixing, and Modern AI Systems

**Academic Case Study and Research Monograph**  
**Course Context:** Advanced Artificial Intelligence, Natural Language Processing, and Neural Networks  
**Target Audience:** Undergraduate and Graduate Scholars in AI, Data Science, and Computational Linguistics  
**Document Classification:** Academic Research & Ecosystem Case Study  
**Publication Date:** October 2026  

---

## Abstract

Natural Language Processing (NLP) for South Asian languages has undergone a profound paradigm shift over the past four decades, transitioning from symbolic rule-based expert systems to statistical modeling, sequence-to-sequence neural architectures, dense multilingual transformers, and contemporary generative Large Language Models (LLMs). While standardized machine translation between formal Telugu script (తెలుగు) and English is now considered a technologically mature discipline—demonstrated by state-of-the-art production systems such as Google Translate, Microsoft Translator, and AI4Bharat’s IndicTrans2—a critical technological gap persists in the processing of informal, digital vernacular communication. In real-world social media, instant messaging (WhatsApp, Telegram), and web platforms, native speakers predominantly communicate in **Tenglish** (Romanized Telugu) characterized by extensive intra-sentential **code-mixing**, unstandardized phonetic orthography, chat abbreviations, and English loanword insertions. 

This study provides a rigorous, academically grounded investigation of the Telugu–English linguistic computing ecosystem. We first trace the chronological evolution of machine translation across six distinct technological epochs: Rule-Based MT (RBMT) and Paninian grammar frameworks (e.g., Anusaaraka), Statistical MT (SMT), recurrent Neural MT (NMT), Transformer-based cross-lingual modeling, neural transliteration, and modern multilingual LLMs. Second, we evaluate the current technological landscape across major commercial platforms (Google Translate, Microsoft Translator, DeepL, Gboard, SwiftKey) and national research initiatives (AI4Bharat, Sarvam AI, Meta SeamlessM4T, IIIT Hyderabad). Third, we mathematically and linguistically deconstruct the language space into a **Six-Directional Translation Matrix**, defining explicit boundaries between *translation*, *transliteration*, *Romanization*, *code-switching*, *code-mixing*, and *normalization*. Fourth, we formalize the fundamental research bottleneck: the failure of standard subword tokenization and canonical MT decoders on informal Romanized inputs. Fifth, we analyze the positioning of **NormMix AI**—a locally deployed research prototype comprising a FastAPI microservice, Web Translation Studio, and Manifest V3 Chrome Extension utilizing a 2-Layer Bidirectional GRU with Scaled Dot-Product Attention and a Pointer-Generator Copy Mechanism. Finally, we establish an objective multi-dimensional comparative benchmark, detail ten central conclusions, and outline seventeen concrete future research directions.

**Keywords:** Telugu Natural Language Processing, Tenglish, Machine Translation, Neural Transliteration, Intra-sentential Code-Mixing, Pointer-Generator Networks, IndicTrans2, Dakshina Dataset, Aksharantar, Sequence-to-Sequence Modeling.

---

## Table of Contents

1. [Foundational Pedagogical Primer: Understanding Language, Script, and Computation](#1-foundational-pedagogical-primer-understanding-language-script-and-computation)
   - 1.1 Script vs. Language: The Abugida Architecture of Telugu
   - 1.2 Defining the Linguistic Boundaries: Translation, Transliteration, and Romanization
   - 1.3 The Sociolinguistic Phenomena: Code-Switching vs. Code-Mixing
   - 1.4 Text Normalization: The Bridge from Vernacular to Canonical
2. [Historical Background and Technological Evolution](#2-historical-background-and-technological-evolution)
   - 2.1 Epoch 1: Rule-Based Machine Translation (RBMT) & Indian MT Initiatives (1980s–1990s)
   - 2.2 Epoch 2: The Paninian Grammar Framework and Anusaaraka (1995–2005)
   - 2.3 Epoch 3: Statistical Machine Translation (SMT) and Phrase-Based Alignment (2000s–2014)
   - 2.4 Epoch 4: The Neural Revolution—Recurrent Seq2Seq and Attention Mechanisms (2014–2017)
   - 2.5 Epoch 5: The Transformer Era and Massively Multilingual NMT (2017–2022)
   - 2.6 Epoch 6: Modern Multilingual LLMs, Foundation Systems, and Multimodal Speech AI (2023–2026)
   - 2.7 Comprehensive 20-Milestone Evolutionary Timeline Table
   - 2.8 The Theoretical "Why": Drivers of Generational Transitions
3. [Existing Technology and Ecosystem Study](#3-existing-technology-and-ecosystem-study)
   - 3.1 Commercial Translation & Input Ecosystem (Google Translate, Microsoft Translator, DeepL, Gboard, SwiftKey, Google Input Tools)
   - 3.2 Indian National & Academic Research Ecosystem (AI4Bharat, IIIT Hyderabad, IIT Bombay, Sarvam AI, Meta AI)
   - 3.3 Comparative Ecosystem Matrix
4. [Six-Direction Telugu–English–Tenglish Analysis](#4-six-direction-teluguenglishtenglish-analysis)
   - 4.1 Direction A: Telugu (Script) $\to$ English (Translation)
   - 4.2 Direction B: English $\to$ Telugu (Script) (Translation)
   - 4.3 Direction C: Telugu (Script) $\to$ Tenglish (Romanization / Transliteration)
   - 4.4 Direction D: Tenglish $\to$ Telugu (Script) (Script Normalization / Back-Transliteration)
   - 4.5 Direction E: Tenglish $\to$ English (Code-Mixed Normalization + Translation)
   - 4.6 Direction F: English $\to$ Tenglish (Translation + Romanization / Colloquialization)
   - 4.7 Synthesis: The Six-Direction Maturity & Difficulty Continuum
5. [Problem Statement and Research Gap](#5-problem-statement-and-research-gap)
   - 5.1 Deconstructing the Fallacy: What is Solved vs. What Remains Broken
   - 5.2 Anatomy of Real-World Tenglish: Why Canonical MT Fails
   - 5.3 The Tokenizer Failure Mode: Subword Fragmentation and Out-of-Vocabulary Explosions
   - 5.4 The English Loanword Corruption Dilemma
   - 5.5 Capability vs. Maturity vs. Remaining Gap Matrix
6. [Positioning of the Student's Implementation (NormMix AI)](#6-positioning-of-the-students-implementation-normmix-ai)
   - 6.1 System Conceptualization: An Experimental Case-Study Prototype
   - 6.2 Architectural Topology: 2-Layer BiGRU + Scaled Dot-Product Attention
   - 6.3 The Pointer-Generator Copy Mechanism: Guaranteeing English Loanword Fidelity
   - 6.4 Omni-Directional Processing Pipeline (`omni_processor.py`)
   - 6.5 The User Interface Layer: Web Translation Studio and Manifest V3 Chrome Extension
   - 6.6 Empirical Benchmarks and Ablation Analysis
7. [Comprehensive Comparative Evaluation](#7-comprehensive-comparative-evaluation)
   - 7.1 Cross-System Feature Matrix
   - 7.2 What Existing Systems Already Do Exceptionally Well
   - 7.3 Where Existing Commercial and Research Systems Fall Short
   - 7.4 What the Student's Implementation Specifically Solves
   - 7.5 What Remains Unresolved: Academic Limitations of the Current Prototype
8. [Conclusion and Future Scope](#8-conclusion-and-future-scope)
   - 8.1 Summary of Findings Across Ten Architectural Dimensions
   - 8.2 Future Research Roadmap (Seventeen Concrete Vectors)
   - 8.3 Central Academic Conclusion
9. [References](#9-references)

---

# 1. Foundational Pedagogical Primer: Understanding Language, Script, and Computation

To rigorously evaluate natural language processing systems for Telugu, English, and their hybrid variants, one must begin with foundational linguistic and computational principles. In computational linguistics, confusing *language* (a cognitive, semantic, and grammatical communication system) with *script* (a graphic orthographic system used to transcribe spoken utterances) produces severe architectural flaws. This section builds the conceptual foundation from scratch, bridging basic linguistic definitions to advanced computational modeling.

```
+---------------------------------------------------------------------------------------------------+
|                                 LINGUISTIC DIMENSIONS MATRIX                                      |
+------------------------------------+--------------------------------------------------------------+
| Dimension                          | Core Technical Definition & Concrete Example                 |
+------------------------------------+--------------------------------------------------------------+
| Language vs. Script                | Language: Telugu (Dravidian grammar, semantics).            |
|                                    | Script: Telugu Lipi (తెలుగు) vs. Latin/Roman (Tenglish).    |
+------------------------------------+--------------------------------------------------------------+
| Translation                        | Semantic transfer across languages:                          |
|                                    | "నేను కాలేజీకి వెళ్తున్నాను" -> "I am going to college."     |
+------------------------------------+--------------------------------------------------------------+
| Transliteration                    | Phonetic mapping across scripts without semantic change:     |
|                                    | "కాలేజీ" (Telugu script) <-> "college / kaaleeji" (Latin).  |
+------------------------------------+--------------------------------------------------------------+
| Romanization                       | Transliteration specifically into the Latin/Roman script:    |
|                                    | "నమస్కారం" -> "namaskaram".                                  |
+------------------------------------+--------------------------------------------------------------+
| Code-Switching (Inter-sentential)  | Alternating languages across complete clause boundaries:     |
|                                    | "I will call you later. నాకు ఇప్పుడు పని ఉంది."             |
+------------------------------------+--------------------------------------------------------------+
| Code-Mixing (Intra-sentential)     | Embedding morphemes/words of Language B into grammar of A:   |
|                                    | "nenu clg lo project submit chesanu."                       |
+------------------------------------+--------------------------------------------------------------+
| Text Normalization                 | Standardizing non-canonical/noisy forms to standard tokens:  |
|                                    | "clg ki velthunna bro" -> "కాలేజీకి వెళ్తున్నాను, bro"       |
+------------------------------------+--------------------------------------------------------------+
```

### 1.1 Script vs. Language: The Abugida Architecture of Telugu

Telugu is a South-Central Dravidian language spoken natively by over 95 million people primarily in Andhra Pradesh, Telangana, and the Yanam district of Puducherry. Historically and typologically, Telugu features an **agglutinative morphology** and a canonical **Subject-Object-Verb (SOV)** word order, contrasting sharply with the analytic, **Subject-Verb-Object (SVO)** structure of English.

Typologically, scripts fall into three primary categories:
1. **Alphabets (e.g., Latin, Greek):** Consonants and vowels have independent, equal graphical status (e.g., in English, 'c-a-t' consists of three discrete glyphs of equal rank).
2. **Abjads (e.g., Arabic, Hebrew):** Only consonants are typically written; vowels are omitted or indicated by optional diacritical marks.
3. **Abugidas / Alphasyllabaries (e.g., Telugu, Devanagari, Tamil):** Consonant-vowel sequences are written as a cohesive syllabic unit centered around a base consonant glyph possessing an inherent vowel (typically the short neutral vowel $/a/$ or $/ʌ/$ in Telugu). If a different vowel follows, the base consonant is modified with a mandatory vowel diacritic (**Mātrā** or *గుణింతం* / Guṇintaṁ). When two or more consonants appear without an intervening vowel (a consonant cluster), a **Virama** (or *హలంత్* / *Polu*) suppresses the inherent vowel, forming subscripted or conjoined ligatures (**Vaṭṭu** / *వత్తు*).

```
                      +-----------------------------+
                      | Base Consonant: క (/ka/)    |
                      +-----------------------------+
                                     |
           +-------------------------+-------------------------+
           |                                                   |
           v                                                   v
   Add Vowel Diacritic:                       Conjoin Consonant Cluster (Vattu):
   క + ా (aa) = కా (/kaa/)                    క + ్ (Virama) + య (ya) = క్య (/kya/)
```

In Unicode computing, Telugu script spans the block `U+0C00` to `U+0C7F`. When a user types Telugu in its native script, text engines render complex grapheme clusters via OpenType glyph tables. However, when users type on a standard Latin QWERTY keyboard, they do not enter Unicode Telugu code points; rather, they construct an ad-hoc, phonetic approximation using ASCII characters (e.g., typing `nenu` for `నేను`). This leads to the fundamental distinction between language and script: **Tenglish is not a new language; it is the Telugu language transcribed through the Latin script, frequently seasoned with English lexicon.**

### 1.2 Defining the Linguistic Boundaries: Translation, Transliteration, and Romanization

Precision in terminology is crucial for academic analysis:

*   **Machine Translation (MT):** The computational task of taking a text sequence $S_A$ in Source Language $L_A$ and producing an equivalent text sequence $T_B$ in Target Language $L_B$ such that the *semantic meaning*, *discourse pragmatics*, and *grammatical integrity* are preserved.
    $$\text{Translation: } \mathcal{M}_{\text{MT}}(S_{L_A}) \to T_{L_B}, \quad \text{Semantics}(S) \equiv \text{Semantics}(T)$$
    *Example:* Telugu native `నేను కాలేజీకి వెళ్తున్నాను` $\to$ English `I am going to college.`

*   **Transliteration:** The task of converting text from Source Script $\mathcal{S}_1$ to Target Script $\mathcal{S}_2$ such that the *phonetic realization* (pronunciation) is preserved as accurately as possible, without translating semantic meaning.
    $$\text{Transliteration: } \mathcal{M}_{\text{Xlit}}(W_{\mathcal{S}_1}) \to W_{\mathcal{S}_2}, \quad \text{Phonetics}(W_{\mathcal{S}_1}) \approx \text{Phonetics}(W_{\mathcal{S}_2})$$
    *Example:* Telugu script `కాలేజీ` $\to$ Latin script `kaaleeji` or `college`.

*   **Romanization:** A specific subtype of transliteration where the target script is explicitly restricted to the Latin (Roman) alphabet. Standardized Romanization schemes exist (e.g., ISO 15919, IAST, National Library at Kolkata romanization, or Harvard-Kyoto), which use precise diacritical marks (e.g., $\bar{a}, \bar{i}, \bar{u}, \text{ṭ}, \text{ḍ}, \text{ṇ}, \text{ś}$). In contrast, **informal Romanization** (colloquial Tenglish) dispenses with all formal diacritics and relies on rough ASCII approximations.

### 1.3 The Sociolinguistic Phenomena: Code-Switching vs. Code-Mixing

In multilingual speech communities, bilingual speakers effortlessly transition between languages:

*   **Code-Switching (Inter-Sentential):** The alternation between two languages at syntactic clause or sentence boundaries. Speaker A completes a full clause or thought in Telugu, then switches to English for the next clause.
    *Example:* `I will call you later. నాకు ఇప్పుడు అర్జెంట్ పని ఉంది.` (Clear clause separation; each clause adheres strictly to the monolingual grammatical rules of its respective language).

*   **Code-Mixing (Intra-Sentential):** The embedding of linguistic units (morphemes, words, phrases, compound verbs) from one language into the syntactic framework of another language *within the same clause*.
    *Example:* `naku ivala college lo important interview undi bro.`
    Here, the syntactic matrix is Telugu (indicated by the dative pronoun `naku`, postposition `lo`, existential copula `undi`), but the content words (`college`, `important`, `interview`, `bro`) are English nouns and adjectives. Furthermore, English loanwords undergo morphological integration via Telugu case markers, e.g.:
    - `college` + `కి` (dative suffix `-ki`) $\to$ `college-ki` (to college)
    - `submit` + `చేశావా` (past interrogative auxiliary `chesava`) $\to$ `submit chesava?` (did you submit?)

### 1.4 Text Normalization: The Bridge from Vernacular to Canonical

**Text Normalization** in code-mixed computing is the transformation of non-standard, noisy, colloquial, or Romanized orthography into a standardized, canonical orthographic representation suitable for downstream consumption by natural language understanding (NLU), machine translation (MT), or speech synthesis (TTS) models.

In Telugu–English computing, normalization involves two distinct operations:
1. **Phonetic Script Restitution:** Converting Romanized Telugu morphemes into canonical Telugu Unicode script (`nenu` $\to$ `నేను`, `velthunna` $\to$ `వెళ్తున్నాను`).
2. **Loanword Isolation & Orthographic Canonicalization:** Detecting embedded English lexical tokens, correcting phonetic typing noise (`clg` $\to$ `college`, `intrvw` $\to$ `interview`), and either preserving them in pristine Latin script or transcribing them into standard Telugu loanword conventions.

---

# 2. Historical Background and Technological Evolution

The progression of Telugu computational linguistics spans over four decades. To understand why modern AI systems operate as they do, one must trace the evolutionary lineage of Natural Language Processing (NLP) from symbolic manipulation to deep representation learning.

```
+---------------------------------------------------------------------------------------------------+
|                                  THE 40-YEAR NLP EVOLUTION                                        |
+---------------------------------------------------------------------------------------------------+
|  1985-1995: Rule-Based MT (RBMT)                                                                  |
|  - TDIL, Expert Systems, Morphological Analyzers, Bilingual Lexicons                              |
|  - Limitation: Combinatorial rule explosion; fragile on exceptions and colloquialisms             |
|                                    |                                                              |
|                                    v                                                              |
|  1995-2005: Paninian Grammar & Anusaaraka                                                         |
|  - IIIT Hyderabad, IIT Kanpur, University of Hyderabad (Sangal, Chaitanya, Kulkarni)              |
|  - Karaka relations, Information transfer without loss, Shallow parsing                           |
|  - Limitation: Constrained domain coverage; dependency on handcrafted parsers                     |
|                                    |                                                              |
|                                    v                                                              |
|  2005-2014: Statistical Machine Translation (SMT)                                                 |
|  - IBM Models 1-5, Moses Toolkit, GIZA++, Word Alignment, n-gram Language Models                  |
|  - Limitation: Severe data sparsity; agglutinative Telugu morphology created OOV explosion        |
|                                    |                                                              |
|                                    v                                                              |
|  2014-2017: Recurrent Neural MT (NMT) & Attention                                                 |
|  - Seq2Seq (Sutskever et al.), RNN/LSTM/GRU (Cho et al.), Bahdanau & Luong Attention              |
|  - Google GNMT (2016-2017), Microsoft Translator Telugu expansion                                 |
|  - Limitation: Vanishing gradients over long distances; sequential bottleneck prevents scaling    |
|                                    |                                                              |
|                                    v                                                              |
|  2017-2022: Dense Multilingual Transformers                                                       |
|  - Transformer (Vaswani et al.), mBART, IndicTrans, Samanantar, Google Dakshina, Aksharantar      |
|  - Limitation: High compute costs; subword BPE tokenizers fail on noisy Romanized scripts         |
|                                    |                                                              |
|                                    v                                                              |
|  2023-2026: Generative LLMs, Foundation Models & Multimodal Speech                                |
|  - IndicTrans2, Sarvam-1, Meta SeamlessM4T, GPT-4, Llama-3, IndicConformer, Live Speech          |
|  - Current Challenge: Context-aware Tenglish code-mixing and real-time on-device inference        |
+---------------------------------------------------------------------------------------------------+
```

### 2.1 Epoch 1: Rule-Based Machine Translation (RBMT) & Indian MT Initiatives (1980s–1990s)

In the early decades of computational linguistics in India, the primary paradigm was Rule-Based Machine Translation (RBMT). Recognizing the immense linguistic diversity of the Indian subcontinent (22 scheduled languages across Indo-Aryan, Dravidian, Austroasiatic, and Tibeto-Burman families), the Government of India’s Department of Electronics (now the Ministry of Electronics and Information Technology, MeitY) launched the **Technology Development for Indian Languages (TDIL)** initiative in 1991.

Early MT prototypes—such as **Anglabharti** (developed at IIT Kanpur under Prof. R.M.K. Sinha, targeting English-to-Indian translation using a rule-based pseudo-interlingua approach) and **MaTra** (developed by the National Centre for Software Technology, NCST)—focused predominantly on Indo-Aryan languages (Hindi, Bengali). Telugu, being a Dravidian language with an exceptionally complex agglutinative inflectional structure, posed severe challenges to standard transfer-rule systems. 

**Architectural Mechanics of RBMT:**
1. **Morphological Analysis:** A source word was stripped of its inflectional affixes using handcrafted Finite State Automata (FSA) or rule engines to identify the lemma and grammatical features (e.g., `వెళ్తున్నాను` $\to$ Lemma: `వెళ్లు` [go] + Aspect: Progressive + Tense: Present + Person: 1st + Number: Singular).
2. **Syntactic Parsing:** Context-Free Grammars (CFGs) parsed the sentence into hierarchical syntax trees.
3. **Structural Transfer:** Rules reordered the parsed constituents from Source Language syntax to Target Language syntax (e.g., transforming SVO in English to SOV in Telugu).
4. **Morphological Generation:** The target language surface forms were synthesized from lemmas using bilingual dictionaries.

*Why RBMT Stalled:* Natural language exhibits an astronomical number of grammatical edge cases, idioms, polysemy, and colloquial permutations. Handcrafting rules for every syntactic interaction required hundreds of person-years of linguistic expertise. The rules were brittle; a single unexpected token or typo broke the parser entirely.

### 2.2 Epoch 2: The Paninian Grammar Framework and Anusaaraka (1995–2005)

A major intellectual breakthrough in Indian MT emerged from the **Akshar Bharati** research group, led by Prof. Rajeev Sangal (IIIT Hyderabad / IIT BHU), Prof. Vineet Chaitanya, and Prof. Amba Kulkarni. Rather than borrowing Western grammatical formalisms (Chomskyan Generative Grammar), they formulated computational parsers based on the ancient Indian grammatical treatise of **Pāṇini** (the *Ashtādhyāyī*).

This endeavor gave birth to **Anusaaraka** (translating from Sanskrit as "one that follows" or "facilitates"). Anusaaraka was not merely a translation system; its philosophical goal was to allow a reader of Language B to access text in Language A *without loss of information*, preserving semantic nuances through transparent lexical mapping and structural annotations.

**Core Paninian Concepts Applied to Telugu:**
*   **Kāraka Theory:** Instead of treating syntax as purely structural phrase trees (NP, VP), the Paninian framework models the semantic relationships between the verb (action / *kriyā*) and the nominal arguments participating in that action. The six primary Kārakas (Kartā [agent], Karma [theme/object], Karaṇa [instrument], Sampradāna [recipient], Apādāna [source], Adhikaraṇa [locus]) map directly to Telugu postpositional case markers (*విభక్తులు* / Vibhaktulu):
    - `తో` (-tho) indicates Karaṇa kāraka (instrumental: `కలంతో రాశాడు` = wrote with a pen).
    - `కొరకు / కోసం` (-koraku / -kosam) indicates Sampradāna (purposive/dative).
    - `నుంచి` (-nunchi) indicates Apādāna (ablative: `ఇంటి నుంచి` = from home).
    - `లో` (-lo) indicates Adhikaraṇa (locative: `కాలేజీలో` = in college).
*   **Information Distribution:** If a morphological feature in Telugu (e.g., respectful plural distinction in second-person pronouns: `నువ్వు` [informal singular] vs. `మీరు` [polite/plural]) had no direct equivalent in English, Anusaaraka retained explicit grammatical tags so the reader was never misled.

Following Anusaaraka, the multi-institutional consortium project **Sampark** was funded by TDIL, uniting IIIT Hyderabad, University of Hyderabad, IIT Kharagpur, and CDAC to build MT systems across 18 Indian language pairs, including Telugu–Hindi and Telugu–Tamil. While Sampark achieved remarkable linguistic precision on formal news text, it remained constrained to formal literary Telugu and could not process informal, spoken, or Romanized language.

### 2.3 Epoch 3: Statistical Machine Translation (SMT) and Phrase-Based Alignment (2000s–2014)

The dawn of the 21st century revolutionized NLP through empirical, data-driven approaches. Pioneered by IBM's Candide project and formalized in the mathematical foundations of Brown et al. (1993), Machine Translation was reframed as a noisy-channel optimization problem:

$$\hat{E} = \arg\max_{E} P(E|F) = \arg\max_{E} P(F|E) \cdot P(E)$$

Where $F$ is the foreign source sentence (Telugu), $E$ is the English target sentence, $P(F|E)$ is the **Translation Model** (estimating lexical and phrasal correspondence), and $P(E)$ is the **Language Model** (typically an $n$-gram model trained on massive target monolingual text using SRILM or KenLM to ensure fluent output).

With the release of the open-source **Moses** toolkit (Koehn et al., 2007) and **GIZA++** for word alignment (Och and Ney, 2003), researchers could automatically train Phrase-Based Statistical Machine Translation (PB-SMT) models from bilingual parallel corpora without handcrafting grammatical rules.

**Why SMT Failed to Fully Solve Telugu–English:**
1. **Agglutinative Morphological Sparsity:** Telugu attaches tense, aspect, mood, person, number, gender, and postpositions as continuous suffixes to base roots. A single Telugu verb root can generate over 200 distinct surface forms (e.g., `చేయడం`, `చేశాను`, `చేశావా`, `చేస్తున్నాను`, `చేయగలరు`, `చేయకపోతే`, `చేయాల్సిందే`). In standard SMT, words are treated as atomic tokens. Thus, each inflected variant was treated as an entirely separate vocabulary item, causing severe data sparsity and high Out-Of-Vocabulary (OOV) rates.
2. **Long-Distance Reordering:** English is rigidly SVO (`[Subject: The student] [Verb: submitted] [Object: the assignment]`), whereas Telugu is SOV (`[Subject: విద్యార్థి] [Object: అసైన్‌మెంట్‌ను] [Verb: సమర్పించాడు]`). In long, complex sentences, the main verb appears at the very end of the Telugu clause. Phrase-based SMT models utilized a distortion penalty parameter that penalized long-distance phrasal jumps, causing scrambled and ungrammatical English outputs.

### 2.4 Epoch 4: The Neural Revolution—Recurrent Seq2Seq and Attention Mechanisms (2014–2017)

Between 2014 and 2016, deep learning completely supplanted SMT. The breakthrough was the **Sequence-to-Sequence (Seq2Seq)** architecture introduced by Sutskever, Vinyals, and Le (2014) at Google and Cho et al. (2014) at NYU.

Rather than relying on explicit phrase tables and $n$-gram counting, Seq2Seq mapped a variable-length source sentence $(x_1, \dots, x_{T_x})$ to continuous vector representations using a **Recurrent Neural Network (RNN)** Encoder, and generated target tokens $(y_1, \dots, y_{T_y})$ conditioned on that vector using an RNN Decoder. To mitigate the vanishing gradient problem over long sequences, vanilla RNNs were replaced with **Long Short-Term Memory (LSTM)** units (Hochreiter & Schmidhuber, 1997) or **Gated Recurrent Units (GRU)** (Cho et al., 2014).

```
Vanilla Seq2Seq Bottleneck:
Source Tokens: [x_1] -> [x_2] -> [x_3] -> [x_4] ===> Fixed Vector [c] ===> Decoder [y_1] -> [y_2]
(Information about x_1 is severely compressed or lost when c is passed to the decoder)
```

**The Attention Mechanism (Bahdanau et al., 2014; Luong et al., 2015):**
The fundamental flaw of the original Seq2Seq model was the **information bottleneck**: compressing an entire sentence of arbitrary length into a single fixed-size context vector $c$. Bahdanau et al. (2014) eliminated this bottleneck by introducing **additive attention**, allowing the decoder to dynamically compute an alignment distribution over all encoder hidden states at each decoding time-step:

$$e_{t,i} = v_a^T \tanh(W_a s_{t-1} + U_a h_i)$$
$$\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_{k=1}^{T_x} \exp(e_{t,k})}$$
$$c_t = \sum_{i=1}^{T_x} \alpha_{t,i} h_i$$

Where $s_{t-1}$ is the previous decoder state, $h_i$ is the $i$-th encoder hidden state, $\alpha_{t,i}$ is the attention weight, and $c_t$ is the dynamic context vector. Luong et al. (2015) subsequently introduced **multiplicative (dot-product) attention**:

$$e_{t,i} = s_t^T W_a h_i$$

**Commercial Milestones:**
*   **Google Neural Machine Translation (GNMT) (Wu et al., 2016):** Google deployed deep LSTM encoder-decoders with residual connections. In 2017, Google transitioned Telugu from statistical models to GNMT, delivering dramatic improvements in fluency and semantic cohesion for formal Telugu–English translation.
*   **Microsoft Translator (2017–2018):** Microsoft integrated deep neural models for Telugu into its Azure Cognitive Services, enabling real-time neural translation across enterprise applications.

### 2.5 Epoch 5: The Transformer Era and Massively Multilingual NMT (2017–2022)

Despite their success, recurrent neural networks suffered from an inherent computational limitation: **sequential computation**. Because state $h_i$ strictly depends on $h_{i-1}$, training could not be parallelized across the time dimension on modern GPUs.

In 2017, Vaswani et al. published the seminal paper *"Attention Is All You Need"*, introducing the **Transformer** architecture. The Transformer dispensed with recurrence entirely, relying exclusively on **Multi-Head Self-Attention** and Positional Encodings:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

This architecture unlocked unprecedented scaling laws. Large multilingual models could now be pre-trained on billions of tokens across dozens of languages simultaneously.

**Key Indian-Language NLP Research Milestones:**
*   **Google Dakshina Dataset (Roark et al., 2020):** Google Research released Dakshina, a foundational open benchmark for 12 South Asian languages, including Telugu. It provided 8,000 native script sentences paired with human-annotated Latin Romanizations, alongside a validated transliteration lexicon of 58,550 Telugu word pairs. This dataset established the modern standard for evaluating Romanized Indic text.
*   **AI4Bharat Samanantar (Ramesh et al., 2022):** Researchers at AI4Bharat (IIT Madras) released Samanantar, the largest public parallel corpus for Indian languages. For Telugu–English alone, Samanantar compiled **10.5 million parallel sentence pairs**, expanding the public data frontier tenfold and enabling open academic models to compete with proprietary commercial translation APIs.
*   **AI4Bharat Aksharantar & IndicXlit (Madhani et al., 2023):** AI4Bharat developed Aksharantar, the largest publicly available transliteration dataset for 21 Indian languages, containing over **26 million transliteration pairs**. Concurrently, they released **IndicXlit**, a sequence-to-sequence transformer model fine-tuned for high-accuracy phonetic script conversion between Latin script and native Indic scripts.
*   **AI4Bharat IndicTrans (Kakwani et al., 2020) & IndicTrans2 (Gala et al., 2023):** IndicTrans2 achieved state-of-the-art translation accuracy across all 22 scheduled Indian languages and English, outperforming commercial systems on public academic benchmarks (IN22-Conv, FLORES-200) by training on curated bitext with language indicator tokens.

### 2.6 Epoch 6: Modern Multilingual LLMs, Foundation Systems, and Multimodal Speech AI (2023–2026)

By 2023, NLP entered the generative foundation model epoch:
1. **Decoder-Only Multilingual LLMs:** Models such as OpenAI’s GPT-4, Meta’s LLaMA 3, Google’s PaLM 2 and Gemini, and Anthropic’s Claude demonstrated remarkable zero-shot translation capabilities. However, their internal Byte-Pair Encoding (BPE) tokenizers were overwhelmingly trained on English text, leading to severe token fragmentation when processing Telugu script (often requiring 4 to 8 tokens to encode a single Telugu word), drastically increasing latency and API inference costs.
2. **Indian Sovereign AI Initiatives:** 
   - **Sarvam AI (2023–2026):** Founded by Pratyush Kumar and Vivek Raghavan, Sarvam released **Sarvam-1** (a 2-billion parameter model optimized for 10 Indian languages including Telugu, featuring an efficient custom Indic tokenizer), **Bulbul** (state-of-the-art speech synthesis), and **Saaras** (speech-to-text).
   - **Digital India Bhashini (MeitY):** A government-backed initiative democratizing language technology APIs for public governance, enabling real-time voice translation across rural communities.
3. **Multimodal & Live Speech Translation:**
   - **Meta SeamlessM4T (Barrault et al., 2023) & SeamlessM4T v2:** The first unified massively multilingual multimodal model capable of direct Speech-to-Speech Translation (S2ST), Speech-to-Text (S2TT), and Text-to-Speech (T2ST) across 100 languages, including Telugu.
   - **Whisper & IndicConformer:** End-to-end conformer architectures processing spoken conversational audio, paving the way for real-time speech interpretation.

---

### 2.7 Comprehensive 20-Milestone Evolutionary Timeline Table

The following table summarizes the chronological breakthroughs in Telugu and Indian language computational linguistics:

| # | Period / Year | Technology / System | Organization / Researchers | Main Technical Contribution | Direct Relevance to Telugu / Tenglish |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **1991** | TDIL Initiative | Dept. of Electronics (Govt. of India) | Formalized national R&D framework for Indian NLP, character encoding standards (ISCII). | Laid institutional groundwork for all Telugu computational linguistic tools. |
| 2 | **1995** | Anglabharti & MaTra | IIT Kanpur (R.M.K. Sinha) / NCST | Rule-based pseudo-interlingua and tree-adjoining grammars for English-to-Indian MT. | First structured attempt at algorithmic English-to-Dravidian structural transfer. |
| 3 | **1998–2003** | Anusaaraka | Akshar Bharati / IIIT Hyderabad (Sangal, Chaitanya, Kulkarni) | Paninian grammar model, Kāraka dependency structures, information-lossless parsing. | Established semantic Kāraka mappings for Telugu case markers (`తో`, `లో`, `కొరకు`). |
| 4 | **2006–2009** | Sampark Consortium MT | IIIT-H, Univ. of Hyderabad, CDAC, IIT-KGP | Multi-engine machine translation consortium across 18 Indian language pairs. | Produced production-grade morphological analyzers and POS taggers for Telugu. |
| 5 | **2007** | Moses Toolkit & SMT | Koehn et al. / Academic Open Source | Phrase-based statistical translation using log-linear models and heuristic phrase extraction. | Allowed first data-driven Telugu-English translation; suffered from agglutinative OOV. |
| 6 | **2010** | Google Input Tools & Transliteration | Google Inc. | Statistical dictionary and character $n$-gram transliteration for Latin-to-Indic scripts. | Enabled consumers to type Tenglish on web forms to produce Telugu script. |
| 7 | **2014** | Sequence-to-Sequence (Seq2Seq) | Sutskever et al. (Google) / Cho et al. (NYU) | End-to-end recurrent neural translation using LSTM/GRU encoder-decoder topology. | Proved neural representations could outperform discrete statistical phrase tables. |
| 8 | **2014–2015** | Attention Mechanisms | Bahdanau et al. (Additive) / Luong et al. (Multiplicative) | Solved the information bottleneck by aligning decoder states to all encoder hidden vectors. | Made long Telugu sentence translation grammatically coherent. |
| 9 | **2016–2017** | Google Neural MT (GNMT) Telugu | Google Research (Wu et al., 2016) | Large-scale deep LSTM models with residual connections deployed to Google Translate. | Represented the inflection point where Telugu–English MT became commercially viable. |
| 10 | **2017** | The Transformer Architecture | Vaswani et al. (Google Brain / Research) | Replaced recurrence with Multi-Head Self-Attention, enabling massive parallelization. | The universal structural backbone for all modern LLMs, IndicTrans, and Whisper. |
| 11 | **2018** | Microsoft Translator Neural Telugu | Microsoft Cognitive Services | Production neural machine translation integrated into Microsoft Office, Bing, Azure. | High-quality enterprise translation for Telugu administrative and business documents. |
| 12 | **2020** | Google Dakshina Dataset | Roark et al. (Google Research) | Released gold benchmark of Latin-transliterated text for 12 Indic languages (8k sentences, 58k lexicon). | First standardized academic benchmark for evaluating Romanized Telugu (Tenglish). |
| 13 | **2020** | IndicTrans (v1) | AI4Bharat (Kakwani et al., 2020) | First 16-language multilingual transformer trained exclusively on public Indian corpora. | Validated transfer learning between Dravidian languages (Tamil $\leftrightarrow$ Telugu). |
| 14 | **2022** | Samanantar Corpus | AI4Bharat (Ramesh et al., 2022) | Curated largest public parallel dataset for Indic MT: 10.5 million Telugu–English pairs. | Demolished the low-resource data bottleneck for academic Telugu MT research. |
| 15 | **2023** | Aksharantar & IndicXlit | AI4Bharat (Madhani et al., 2023) | 26M word-pair transliteration corpus and sequence-to-sequence transformer model. | State-of-the-art phonetic Latin-to-Telugu word transliteration model. |
| 16 | **2023** | IndicTrans2 | AI4Bharat (Gala et al., 2023) | SOTA open-source multilingual MT for 22 languages; outscored commercial APIs on FLORES. | Currently the premier open-source translation model for formal Telugu $\leftrightarrow$ English. |
| 17 | **2023** | Meta SeamlessM4T (v1/v2) | Meta AI (Barrault et al., 2023) | Massively multilingual, multimodal model for Speech-to-Speech and Speech-to-Text. | Direct spoken Telugu $\to$ spoken English translation without intermediate text steps. |
| 18 | **2024** | Sarvam AI Sovereign Models | Sarvam AI (Sarvam-1, Bulbul, Saaras) | 2B-parameter LLM with custom Indic tokenization, low latency, and native Indian voice models. | Dramatically lowered latency and token cost for Telugu text generation. |
| 19 | **2024–2025** | Generative Multilingual LLMs | OpenAI (GPT-4o), Google (Gemini 1.5/2.0), Anthropic | Advanced zero-shot reasoning, contextual translation, and cross-lingual comprehension. | Capable of interpreting nuanced colloquial Tenglish when explicitly prompted. |
| 20 | **2025–2026** | Multimodal Live AI & Indic Conformer | MeitY Bhashini, Meta, Google, Academic Labs | Sub-second latency live speech translation, bi-directional code-mixed interpretation. | Current frontier: Processing conversational spoken Tenglish in real-time meetings. |

---

### 2.8 The Theoretical "Why": Drivers of Generational Transitions

Technological transitions in NLP never occur by accident; each evolutionary leap was driven by mathematical and linguistic limitations in the preceding generation:

1. **Why RBMT ceded to SMT:** Natural language is fundamentally stochastic, ambiguous, and evolutionary. Rule-based systems required an unsustainable exponential increase in manual rules ($\mathcal{O}(2^n)$ interactions) to handle exceptions. SMT replaced deterministic rule trees with probabilistic estimation over real-world parallel corpora, drastically reducing human authoring costs.
2. **Why SMT ceded to NMT:** SMT represented words as discrete atomic identities ($w \in \{1, \dots, |V|\}$), ignoring semantic similarity between synonyms. It operated over local windows (3-grams or 5-grams) and was crippled by Telugu's agglutinative morphology and long-distance SOV verb displacement. NMT introduced continuous dense embeddings ($\mathbb{R}^d$) where semantically related tokens clustered geometrically, and recurrent memory cells captured arbitrary non-local dependencies.
3. **Why Recurrent NMT ceded to Transformers:** RNNs and LSTMs compute sequentially ($h_t = f(h_{t-1}, x_t)$), preventing horizontal scaling across GPU tensor cores. Furthermore, even with attention, gradients suffered degradation over sequences exceeding 100 steps. The Transformer eliminated recurrence, computing all pairwise token interactions simultaneously via self-attention ($\mathcal{O}(1)$ path length between any two tokens), allowing models to be scaled to billions of parameters.
4. **Why Modern LLMs still struggle with Tenglish:** LLMs rely on subword tokenizers (e.g., Tiktoken, SentencePiece) trained on massive English and formal web scrapes. Informal Romanized Telugu (`nenu repu clg ki velthunna bro`) is fragmented into nonsensical character shards, stripping away the semantic representations learned during pre-training.

---

# 3. Existing Technology and Ecosystem Study

To properly contextualize research in this domain, one must survey both the commercial tools utilized daily by millions of end-users and the advanced open-source systems developed by premier academic research consortia.

```
                              +---------------------------------------+
                              |      EXISTING NLP ECOSYSTEM           |
                              +---------------------------------------+
                                                  |
                +---------------------------------+---------------------------------+
                |                                                                   |
                v                                                                   v
+-------------------------------+                                   +-------------------------------+
|     Commercial Platforms      |                                   |  Open Research Consortia      |
+-------------------------------+                                   +-------------------------------+
| - Google Translate            |                                   | - AI4Bharat (IndicTrans2)     |
| - Microsoft Translator        |                                   | - AI4Bharat (IndicXlit)       |
| - DeepL                       |                                   | - Meta AI (SeamlessM4T)       |
| - Google Input Tools          |                                   | - Sarvam AI (Sarvam-1)        |
| - Gboard / SwiftKey           |                                   | - IIIT Hyderabad (LTRC)       |
+-------------------------------+                                   +-------------------------------+
```

### 3.1 Commercial Translation & Input Ecosystem

#### 1. Google Translate
*   **Organization:** Google LLC (Alphabet Inc.)
*   **Launch / Neural Transition:** Statistical MT launched for Telugu around 2011; fully transitioned to Google Neural Machine Translation (GNMT) in 2017; currently powered by Gemini-derived multilingual foundation models.
*   **Supported Modalities:** Formal Telugu script $\leftrightarrow$ English translation; speech-to-text; text-to-speech; optical character recognition (Google Lens).
*   **Transliteration Capabilities:** Includes a basic phonetic Latin-to-Telugu script conversion preview in web input boxes.
*   **Code-Mixing Capabilities:** **Partial / Fragile.** When given clean, simple Tenglish (e.g., `nenu college ki velthunnanu`), Google Translate often recognizes it and translates it correctly to English. However, when presented with heavy intra-sentential code-mixing, social media slang, or abbreviations (e.g., `ivala clg lo test undi bro submit chesava`), the engine misinterprets Romanized words as English vocabulary or phonetically hallucinates nonsensical translations.
*   **Availability:** Public Web UI, iOS/Android Apps, Google Cloud Translation REST API, Chrome Browser built-in page translation.
*   **Major Strengths:** Unmatched parallel data scale; exceptional fluency on formal, literary, journalistic, and administrative Telugu; sub-second global API latency.
*   **Major Limitations:** Closed proprietary API; black-box tokenization; frequent failure on non-standard Tenglish orthography.

#### 2. Microsoft Translator
*   **Organization:** Microsoft Corporation (Azure Cognitive Services)
*   **Launch / Neural Transition:** Telugu support introduced statistically in 2014–2015; neural models deployed in 2018; integrated into Microsoft 365 and Azure Translator API.
*   **Supported Modalities:** Formal Telugu script $\leftrightarrow$ English; document translation (PDF, DOCX); real-time multi-device conversation transcription.
*   **Transliteration Capabilities:** Exposes script conversion APIs via Azure Cognitive Services.
*   **Code-Mixing Capabilities:** **Minimal.** Microsoft Translator expects standard orthographic inputs in either Telugu Unicode script or English. When supplied with informal Tenglish, it attempts to parse the input through English dictionary decoders, yielding heavily corrupted translations.
*   **Availability:** Azure REST API, Microsoft Translator App, Edge browser built-in page translation, Office desktop suite.
*   **Major Strengths:** Enterprise SLA, strict enterprise data privacy compliance, high domain fidelity for technical, legal, and business documentation.
*   **Major Limitations:** Strictly optimized for formal corporate text; largely incapable of handling informal conversational code-mixing.

#### 3. DeepL Translator
*   **Organization:** DeepL SE (Germany)
*   **Status on Telugu Support:** **Unsupported as of late 2026.** DeepL focuses primarily on high-resource European and select Asian languages (Japanese, Chinese, Indonesian, Korean). Telugu is currently absent from their production offerings.
*   **Relevance to Study:** Demonstrates that despite rapid global NLP advances, major specialized commercial translation providers continue to overlook Dravidian languages due to parallel data acquisition hurdles.

#### 4. Google Input Tools, Gboard & Microsoft SwiftKey
*   **Organizations:** Google LLC / Microsoft Corporation
*   **What They Are:** Client-side virtual input method editors (IMEs) deployed across Android, iOS, and web browsers.
*   **Core Task:** Phonetic Transliteration (Latin QWERTY typing $\to$ Telugu script output).
*   **Mechanisms:**
    - Statistical dictionary trie lookup combined with character $n$-gram language models.
    - As the user types `meeru`, the keyboard proposes candidate Telugu script words: `1. మీరు  2. మీరూ  3. మెరు`.
*   **Code-Mixing Capabilities:** They allow users to toggle seamlessly between English words and Telugu transliterated words. However, they perform **word-level token transliteration only**; they do not perform sentence-level semantic translation or grammatical normalization.

---

### 3.2 Indian National & Academic Research Ecosystem

#### 1. AI4Bharat (IIT Madras)
*   **Organization:** AI4Bharat, a MeitY-supported research center at IIT Madras, partnered with EkStep Foundation.
*   **Key Systems Released:**
    - **IndicTrans2 (Gala et al., 2023):** State-of-the-art 1B-parameter sequence-to-sequence transformer model supporting 22 Indian languages and English across all 462 translation directions. Evaluated on the gold benchmark IN22, IndicTrans2 demonstrated superior BLEU, chrF, and human evaluation scores compared to Google Translate for formal Telugu $\leftrightarrow$ English.
    - **IndicXlit (Madhani et al., 2023):** A specialized transformer model trained on the 26-million-pair **Aksharantar** dataset. It provides high-precision character-level transliteration from Latin script into Telugu script.
    - **Samanantar (Ramesh et al., 2022):** The definitive public parallel text corpus containing 10.5 million Telugu–English sentence pairs.
*   **Licensing & Availability:** Fully open-source under permissive academic licenses (MIT / CC-BY-4.0); model checkpoints downloadable via Hugging Face; containerized inference APIs.
*   **Major Strengths:** Academically peer-reviewed; open weights; explicit architectural tuning for Indic phonology and morphological inflection.
*   **Major Limitations:** Primary models are optimized for canonical Unicode scripts; they do not natively accept unstructured, code-mixed Tenglish sentences as input for direct translation without a dedicated pre-transliteration and normalization pipeline.

#### 2. Sarvam AI
*   **Organization:** Sarvam AI (Bangalore, India)
*   **Launch Year:** Founded 2023; production model suite released 2024–2026.
*   **Key Innovations:**
    - **Sarvam-1:** A 2-billion parameter foundational language model pre-trained from scratch on a balanced corpus of 10 Indian languages. Features a specialized tokenizer that drastically reduces token fertility for Telugu text.
    - **Saaras & Bulbul:** Advanced speech recognition and synthesis models tuned to capture regional accents, prosody, and colloquial speech dynamics in Telugu and English.
*   **Relevance to Tenglish:** Sarvam AI's models demonstrate superior resilience to spoken vernacular syntax because their pre-training corpora include conversational spoken datasets.

#### 3. Meta AI: SeamlessM4T
*   **Organization:** Meta Fundamental AI Research (FAIR)
*   **Launch Year:** 2023 (v1), 2024 (v2)
*   **Architecture:** UnitY multi-stage multi-task framework with a shared speech/text encoder (w2v-BERT 2.0) and text-to-unit decoders.
*   **Capabilities:** Direct Speech-to-Speech (S2ST), Speech-to-Text (S2TT), Text-to-Speech (T2ST), and Text-to-Text (T2TT) for nearly 100 languages, including Telugu.
*   **Relevance to Tenglish:** SeamlessM4T excels when processing spoken Telugu containing English code-switching because acoustic speech signals bypass the arbitrary spelling inconsistencies of written Tenglish. However, for written chat text, its performance degrades if the input is heavily corrupted.

#### 4. IIIT Hyderabad (Language Technologies Research Centre - LTRC)
*   **Key Contributions:** The intellectual cradle of computational Dravidian linguistics. Pioneered Telugu Treebanking, morphological analyzers, Paninian dependency parsing, and the early consortium MT frameworks (Sampark). Current academic work focuses heavily on code-mixing shared tasks (e.g., ICON, FIRE workshops).

---

### 3.3 Comparative Ecosystem Matrix

The following multi-dimensional matrix evaluates the primary computational engines operating in the Telugu–English linguistic landscape:

| System / Platform | Primary Organization | Type | Telugu $\leftrightarrow$ Eng MT | Transliteration (Latin $\leftrightarrow$ Te) | Speech (STT / TTS) | Raw Tenglish Handling | Code-Mixing Resilience | Primary Deployment Interfaces |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Google Translate** | Google LLC | Commercial Proprietary | **High (SOTA)** | ⚠️ Basic Preview | ✅ Both | ⚠️ Partial (Heuristic) | ⚠️ Brittle on Slang | Web, App, REST API, Chrome |
| **Microsoft Translator** | Microsoft Corp | Commercial Proprietary | **High (SOTA)** | ⚠️ API Only | ✅ Both | ❌ Corrupts Output | ❌ Minimal | Azure API, Edge, Office |
| **DeepL Translator** | DeepL SE | Commercial Proprietary | ❌ Unsupported | ❌ None | ❌ None | ❌ Unsupported | ❌ None | Web, Desktop, API |
| **Google Input Tools** | Google LLC | Commercial Utility | ❌ None (IME) | ✅ Word-Level | ❌ None | ⚠️ Output Only | ⚠️ User-Driven | Web IME, Chrome Extension |
| **Gboard / SwiftKey** | Google / Microsoft | Commercial Virtual Keyboard | ❌ None (IME) | ✅ Predictive Trie | ✅ Voice Input | ⚠️ Keyboard Suggestions | ⚠️ User-Driven | Mobile OS Virtual Keyboards |
| **IndicTrans2** | AI4Bharat (IIT-M) | Open-Source Research | **High (SOTA)** | ❌ None (Requires Xlit) | ❌ None | ❌ Requires Native Script | ❌ Low (Pure Script Target) | Hugging Face, Python, Docker |
| **IndicXlit** | AI4Bharat (IIT-M) | Open-Source Research | ❌ None | **High (SOTA)** | ❌ None | ✅ Word-by-Word | ⚠️ Transliterates All Words | Hugging Face, Python Library |
| **SeamlessM4T v2** | Meta AI (FAIR) | Open-Weights Research | **High (Multimodal)** | ❌ None | ✅ S2ST, S2TT, T2ST | ⚠️ Speech Only | ⚠️ Audio-Only Robustness | PyTorch, Hugging Face Spaces |
| **Sarvam-1 / Suite** | Sarvam AI | Commercial / Open Weights | **Moderate / High** | ⚠️ Custom Pipeline | ✅ Bulbul / Saaras | ⚠️ Good Prompt Response | ⚠️ Moderate LLM Reasoning | REST API, Python SDK |

---

# 4. Six-Direction Telugu–English–Tenglish Analysis

To achieve academic completeness, language processing between Telugu, English, and Tenglish must not be treated as a monolithic task. Instead, it comprises **six distinct operational directions**, each possessing fundamentally different mathematical objectives, linguistic transformations, maturity levels, and failure modes.

```
                              THE SIX-DIRECTIONAL LANGUAGE SPACE
                              
            [Direction A: Translation]                  [Direction C: Romanization]
       +----------------------------------> [English] <-------------------------------+
       |                                       |                                      |
       |  [Direction B: Translation]           | [Direction F: Colloquialization]     |
       |                                       v                                      |
[Telugu Script] <===============================================================> [Tenglish]
       ^                     [Direction D: Back-Transliteration]                      ^
       |                                                                              |
       +------------------------------------------------------------------------------+
                                  [Direction E: Normalization & MT]
```

---

### 4.1 Direction A: Telugu (Script) $\to$ English (Translation)

*   **Task Definition:** Translating canonical Telugu Unicode script into grammatically fluent, semantically accurate English.
*   **Linguistic Nature:** **Standard Machine Translation.** Requires syntactic transformation from Dravidian SOV agglutinative structure to Indo-European SVO analytic structure.
*   **Existing Technologies:** Google Translate, Microsoft Translator, AI4Bharat IndicTrans2, Meta SeamlessM4T.
*   **Current Maturity Level:** **High (Production Mature).**
*   **Major Difficulties:** Resolving polysemy in literary or domain-specific contexts; handling long compound words (*సమాసాలు* / Samāsālu); preserving formal vs. informal honorific registers.
*   **Concrete Examples:**
    - *Input:* `నేను కాలేజీకి వెళ్తున్నాను.`
    - *Output:* `I am going to college.`
    - *Complex Input:* `వ్యవసాయ రంగంలో ఆధునిక సాంకేతిక పరిజ్ఞానాన్ని ఉపయోగించడం ద్వారా రైతుల ఆదాయం గణనీయంగా పెరిగే అవకాశం ఉంది.`
    - *Output:* `By utilizing modern technological knowledge in the agricultural sector, there is a possibility that farmers' income will increase significantly.`
*   **Commercial Support:** Universally supported across all major commercial APIs.
*   **Primary Academic References:** Gala et al. (2023) [IndicTrans2], Ramesh et al. (2022) [Samanantar].

---

### 4.2 Direction B: English $\to$ Telugu (Script) (Translation)

*   **Task Definition:** Translating English text into canonical Telugu script.
*   **Linguistic Nature:** **Standard Machine Translation.** Requires generating appropriate postpositional markers, verbal inflections matching the subject's gender, number, and person, and correct SOV syntax.
*   **Existing Technologies:** Google Translate, Microsoft Translator, AI4Bharat IndicTrans2.
*   **Current Maturity Level:** **High (Production Mature).**
*   **Major Difficulties:** English pronouns (e.g., "you") lack the explicit hierarchical social distinction present in Telugu (`నువ్వు` [informal] vs. `మీరు` [polite/respectful]); English passive voice constructions sound highly unnatural when literally translated into Telugu.
*   **Concrete Examples:**
    - *Input:* `I am going to college.`
    - *Output:* `నేను కాలేజీకి వెళ్తున్నాను.`
    - *Complex Input:* `Please submit the project report before tomorrow afternoon without fail.`
    - *Output:* `దయచేసి రేపు మధ్యాహ్నం లోగా ప్రాజెక్ట్ రిపోర్ట్‌ను తప్పకుండా సమర్పించండి.`
*   **Commercial Support:** Universally supported.
*   **Primary Academic References:** Gala et al. (2023), Ramesh et al. (2022).

---

### 4.3 Direction C: Telugu (Script) $\to$ Tenglish (Romanization / Transliteration)

*   **Task Definition:** Transcribing native Telugu script into Latin/Roman characters.
*   **Linguistic Nature:** **Phonetic Transliteration / Romanization.** No semantic translation occurs.
*   **Existing Technologies:** IndicNLP Library, AI4Bharat IndicXlit (reverse mode), custom rule-based phonetic mappers (e.g., ITRANS, Harvard-Kyoto).
*   **Current Maturity Level:** **High (Technologically Solved).**
*   **Major Difficulties:** Determining whether to generate formal academic transliteration with diacritics (ISO 15919: `nēnu kālējīki veḷtunnānu`) or informal colloquial Tenglish (`nenu college ki velthunnanu`). In colloquial Tenglish, English loanwords that were written in Telugu script (e.g., `కాలేజీ`) must ideally be recognized and restored to their standard English spelling (`college`) rather than phonetically transcribed as `kaaleejee`.
*   **Concrete Examples:**
    - *Input:* `నేను కాలేజీకి వెళ్తున్నాను.`
    - *Formal Romanization:* `nēnu kālējīki veḷtunnānu.`
    - *Colloquial Tenglish:* `nenu college ki velthunnanu.`
*   **Commercial Support:** Supported as an auxiliary pronunciation/reading aid in Google Translate.

---

### 4.4 Direction D: Tenglish $\to$ Telugu (Script) (Script Normalization / Back-Transliteration)

*   **Task Definition:** Converting informal Romanized Telugu into canonical Telugu Unicode script.
*   **Linguistic Nature:** **Phonetic Transliteration combined with Loanword Handling.**
*   **Existing Technologies:** Google Input Tools, AI4Bharat IndicXlit, Microsoft SwiftKey, Google Dakshina baselines.
*   **Current Maturity Level:** **Moderate to High for single words; Low to Moderate for code-mixed sentences.**
*   **Major Difficulties:** **Spelling Ambiguity and Homophonic Collisions.** Because users type phonetically without standard spelling rules, a single Romanized string can map to multiple distinct Telugu words:
    - `kadu` $\to$ `కాదు` (is not) or `కడు` (wash / belly)?
    - `undi` $\to$ `ఉంది` (is there) or `ఉండి` (having stayed)?
    - Furthermore, naive transliteration engines blindly convert English loanwords into Telugu gibberish:
      *Input:* `nenu college ki vellanu` $\to$ *Naive Translit:* `నేను కొల్లెగె కి వెళ్లాను` (corrupting `college` into `కొల్లెగె` instead of `కాలేజీ`).
*   **Concrete Examples:**
    - *Input:* `nenu college ki velthunnanu`
    - *Target Script:* `నేను కాలేజీకి వెళ్తున్నాను`
*   **Commercial Support:** Available in IMEs (Gboard) via user-assisted candidate selection; not fully automated for continuous unstructured text streams.
*   **Primary Academic References:** Roark et al. (2020) [Dakshina], Madhani et al. (2023) [Aksharantar].

---

### 4.5 Direction E: Tenglish $\to$ English (Code-Mixed Normalization + Translation)

*   **Task Definition:** Converting an informal, Romanized, code-mixed sentence directly into standard English.
*   **Linguistic Nature:** **Multi-Stage Compound Task: Language Identification (LID) + Orthographic Normalization + Machine Translation.**
*   **Existing Technologies:** Specialized academic research prototypes, advanced LLMs (GPT-4o, Gemini 2.0 via zero-shot prompt engineering). Mainstream commercial MT engines (Google Translate, Microsoft Translator) handle this heuristically and frequently fail.
*   **Current Maturity Level:** **Moderate (Active Research Frontier).**
*   **Major Difficulties:** The engine must parse a chaotic linguistic blend:
    1. Slang and abbreviations (`ivala clg lo test undi bro`).
    2. Telugu grammar framing English nouns (`project submit chesava?`).
    3. Severe spelling variance (`velthunna`, `velthuna`, `velthunaa`, `veltonna`).
*   **Concrete Examples:**
    - *Input:* `naku ivala clg lo important interview undi bro`
    - *Correct Output:* `I have an important interview in college today, bro.`
    - *Common Commercial Failure:* Translating `clg` as an unknown acronym or attempting to look up `undii` as a European word.
*   **Commercial Support:** **No dedicated commercial API exists for this explicit direction.** Users must rely on generalized LLMs or multi-hop workarounds (Tenglish $\to$ Telugu script $\to$ English).

---

### 4.6 Direction F: English $\to$ Tenglish (Translation + Romanization / Colloquialization)

*   **Task Definition:** Translating standard English into informal, conversational Romanized Telugu (Tenglish) as naturally used by youth on social media.
*   **Linguistic Nature:** **Translation followed by Colloquial Romanization and Stylistic Adaptation.**
*   **Existing Technologies:** Modern LLMs with explicit system instructions; absent from standard MT pipelines.
*   **Current Maturity Level:** **Low to Experimental (No Standard Benchmark).**
*   **Major Difficulties:** There is no single "correct" ground truth for Tenglish. Colloquial registers vary drastically across demographics and geographic regions:
    - Coastal Andhra colloquial: `Nenu college ki velthunnanu.`
    - Telangana / Hyderabad colloquial: `Nenu clg ki pothunna.`
    - Urban bilingual youth: `Nenu today college ki heading bro.`
*   **Concrete Examples:**
    - *Input:* `Are you coming to college today?`
    - *Formal Telugu Translation:* `మీరు ఈరోజు కళాశాలకు వస్తున్నారా?`
    - *Natural Tenglish Generation:* `Ivala college ki vasthunnava?` or `Today clg ki vasthava?`
*   **Commercial Support:** Completely unsupported by traditional commercial MT engines.

---

### 4.7 Synthesis: The Six-Direction Maturity & Difficulty Continuum

```
+---------------------------------------------------------------------------------------------------+
|                            DIRECTIONAL MATURITY SPECTRUM                                          |
+---------------------------------------------------------------------------------------------------+
|  [Direction A] Telugu (Script) -> English         | HIGH (SOTA Production Systems)                |
|  [Direction B] English -> Telugu (Script)         | HIGH (SOTA Production Systems)                |
|  [Direction C] Telugu (Script) -> Tenglish        | HIGH (Algorithmic / Deterministic)            |
|  [Direction D] Tenglish -> Telugu (Script)        | MODERATE (Ambiguity & Loanword Corruption)    |
|  [Direction E] Tenglish -> English                | MODERATE / RESEARCH (Code-Mixing Bottleneck)   |
|  [Direction F] English -> Tenglish                | LOW / EXPERIMENTAL (No Standardized Target)   |
+---------------------------------------------------------------------------------------------------+
```

---

# 5. Problem Statement and Research Gap

### 5.1 Deconstructing the Fallacy: What is Solved vs. What Remains Broken

An essential premise of this study is to enforce strict academic precision: **It is categorically false to assert that "Telugu–English translation is an unsolved problem."**

As established in Section 3 and Section 4, translating formal, standardized Telugu script to and from English is a highly solved engineering domain. Production neural systems trained on massive parallel corpora (Samanantar's 10.5 million pairs) routinely achieve BLEU scores exceeding 35–45 and chrF++ scores exceeding 60–70 on formal benchmarks.

**The Actual Research Gap:**
The unsolved frontier lies at the intersection of **informal Romanization, intra-sentential code-mixing, unstandardized orthography, and real-time interactive deployment**. The digital communication of 95+ million Telugu speakers has largely abandoned formal Telugu Unicode script on mobile devices in favor of Latin QWERTY input. Consequently, a massive linguistic chasm has opened between *the language data that MT models are trained on* (canonical, edited news and government publications) and *the language data that users actually produce* (noisy, unstandardized, code-mixed Tenglish).

```
   Canonical World (Where MT Works):
   [తెలుగు లిపి]  <======================================>  [English Text]
   "నేను కళాశాలకు వెళ్తున్నాను."                            "I am going to college."
   
                                    VS.
   
   Real-World Vernacular (The Unsolved Gap):
   [Noisy Tenglish] =====================================>  [Downstream AI / Humans]
   "nenu clg lo project submit chesi vasthe baguntundi bro"  (Fails on Tokenization,
                                                              Corrupts Loanwords,
                                                              Lacks Standardization)
```

### 5.2 Anatomy of Real-World Tenglish: Why Canonical MT Fails

To understand why a state-of-the-art model like IndicTrans2 or Google Translate degrades when fed Tenglish, consider the five linguistic pathologies intrinsic to digital code-mixing:

#### 1. Unstandardized Phonetic Orthography
Because Telugu is an Abugida with 56 distinct letters (including phonemically contrastive short/long vowels and aspirated/retroflex consonants), transcribing it into the 26-letter Latin alphabet forces severe loss of phonetic precision. Different speakers transcribe the exact same lexical item in radically different ways:
- Canonical Telugu: `చాలా` (very / much)
- Tenglish Variations: `chala`, `chaala`, `tsala`, `sala`, `chalaa`, `chla`
- Canonical Telugu: `వెళ్తున్నాను` (I am going)
- Tenglish Variations: `velthunnanu`, `velthunna`, `veltonna`, `velthuna`, `velthna`, `pothunna`

#### 2. English Loanwords Embedded in Telugu Inflectional Morphology
In Tenglish, English words are not merely interspersed; they are morphologically inflected as if they were native Telugu noun stems. Consider:
- `college-ki` $\to$ Base English noun `college` + Telugu dative suffix `-ki` (to).
- `interview-lo` $\to$ Base English noun `interview` + Telugu locative suffix `-lo` (in).
- `download avvaledu` $\to$ Base English verb `download` + Telugu negative auxiliary `avvaledu` (did not happen).
- `submit chesava` $\to$ Base English verb `submit` + Telugu past interrogative helper `chesava` (did you do?).

If a model attempts full transliteration, it converts the English word into Telugu script (e.g., `college` $\to$ `కొల్లెగె`). If it attempts English translation without morphological segmentation, it treats `collegeki` as an unknown Out-Of-Vocabulary (OOV) word.

#### 3. Social Media Slang, Shorthand, and Truncation
Digital text frequently omits vowels and truncates syllables:
- `clg` $\to$ `college`
- `intrvw` $\to$ `interview`
- `bro / bava / mama` $\to$ informal relational markers
- `plz / pls` $\to$ `please`
- `bday` $\to$ `birthday`
- `tq / ty` $\to$ `thank you`

#### 4. Intra-Sentential Code-Switching & Token-Level Language Identification
A single sentence may fluctuate between languages at every token:
$$\text{Sentence: } \underbrace{\text{naku}}_{\text{Telugu}} \quad \underbrace{\text{today}}_{\text{English}} \quad \underbrace{\text{college}}_{\text{English}} \quad \underbrace{\text{lo}}_{\text{Telugu}} \quad \underbrace{\text{important}}_{\text{English}} \quad \underbrace{\text{exam}}_{\text{English}} \quad \underbrace{\text{undi}}_{\text{Telugu}} \quad \underbrace{\text{bro}}_{\text{English Slang}}$$
A standard NLP system requires prior **Language Identification (LID)**. However, traditional LID tools (e.g., FastText LID, Compact Language Detector) operate at the *sentence level*, labeling the entire sequence as either English or Telugu. Applying sentence-level LID to intra-sententially code-mixed text results in catastrophic classification errors.

#### 5. Transliteration Ambiguity and Semantic Collisions
In Latin script, short and long vowels are rarely typed accurately. This creates identical Romanized forms for words with opposing meanings:
- `kadu`: Could represent `కాదు` (*kādu*, "is not") or `కడు` (*kaḍu*, "wash / very").
- `padi`: Could represent `పది` (*padi*, number "ten") or `పడి` (*paḍi*, "having fallen").
- `malli`: Could represent `మళ్ళీ` (*maḷḷī*, "again") or the name `మల్లి` (*malli*, "jasmine").
Only a sequence-to-sequence model capable of leveraging surrounding syntactic context can resolve which native word was intended.

---

### 5.3 The Tokenizer Failure Mode: Subword Fragmentation and Out-of-Vocabulary Explosions

The primary technical bottleneck preventing modern Transformers from mastering Tenglish lies in **Subword Tokenization** (Byte-Pair Encoding [BPE], WordPiece, SentencePiece).

In a Transformer, the input text is segmented into subword units from a pre-computed vocabulary $\mathcal{V}$. When an English-centric or formal Indic tokenizer encounters a colloquial Tenglish token, it lacks a dedicated vocabulary entry. The token is shattered into individual characters or arbitrary byte fragments:

```
Token: "velthunnanu"
Canonical Telugu BPE (IndicTrans2 Vocab):   [vel, ##th, ##un, ##nan, ##u] -> 5 fragments!
English GPT-4o Tokenizer:                    ["vel", "th", "un", "nan", "u"] -> 5 tokens!
Native Script Token: "వెళ్తున్నాను"
IndicTrans2 Vocab:                           [" వెళ్తున్నాను "] -> 1 single unified token!
```

This fragmentation causes three computational failures:
1. **Context Window Saturation:** A 15-word Tenglish sentence can explode into 50+ subword tokens, exhausting the attention budget.
2. **Positional Encoding Distortion:** Multi-head attention heads must learn complex, long-range dependencies across arbitrary character fragments rather than whole semantic units.
3. **Loss of Pre-trained Semantic Embeddings:** The embedding for `"vel"` bears zero geometric correlation to the embedding for `వెళ్లు` (go), rendering the model's pre-trained knowledge useless.

---

### 5.4 The English Loanword Corruption Dilemma

When processing code-mixed text through standard transliteration models (e.g., IndicXlit, Google Input Tools), a severe failure mode occurs: **Blind Phonetic Script Conversion**.

Consider the input sentence:
$$\text{Input: } \texttt{nenu college ki velthunna}$$

A naive transliteration model operates under the assumption that *all* input characters represent Telugu phonemes. Consequently, it attempts to phonetically transliterate the English loanword `college`:
$$\texttt{college} \xrightarrow{\quad\text{Naive Transliteration}\quad} \text{"కోల్లెగె" or "కాల్లేజ్"}$$

This outcome is linguistically unacceptable. In authentic Telugu digital text, English loanwords must either be:
1. **Preserved identically in their original Latin script** (`నేను college కి వెళ్తున్నాను`) to maintain visual and semantic clarity, or
2. **Normalized to their universally accepted loanword spelling** (`కాలేజీ`).

Standard Sequence-to-Sequence models lack an innate mechanism to distinguish between an English word that should be copied verbatim and a Romanized Telugu morpheme that must be transformed.

---

### 5.5 Capability vs. Maturity vs. Remaining Gap Matrix

The following table summarizes the technological maturity and remaining challenges across all core capabilities:

| Capability | Existing Technology Maturity | Dominant Existing Paradigm | Remaining Technical & Research Gap |
| :--- | :---: | :--- | :--- |
| **Telugu $\to$ English** | **High** | Dense Multilingual Transformers (IndicTrans2, GNMT) | Contextual nuance in rare cultural idioms; low-resource dialects (e.g., rural Telangana, tribal Gondi/Koya adoptions). |
| **English $\to$ Telugu** | **High** | Supervised NMT with Back-translation | Resolving social honorific ambiguity (informal `నువ్వు` vs. polite `మీరు`); generating natural phrasing instead of robotic literal translations. |
| **Telugu $\to$ Roman Telugu** | **High** | Rule-Based Mappers & Reverse IndicXlit | Preserving English loanwords in standard Latin spelling rather than phonetically spelling them out (e.g., `college` vs `kālējī`). |
| **Roman Telugu $\to$ Telugu** | **Moderate / High** | Predictive Character $n$-gram IMEs (Gboard) | Autonomous resolution of homophonic collisions and sentence-level context disambiguation without human interactive selection. |
| **Tenglish $\to$ English** | **Moderate / Research** | Zero-Shot Large Language Models (GPT-4o) | Handling extreme social media noise, abbreviations (`clg`, `intrvw`), and dialectal variations without massive inference cost and latency. |
| **Tenglish $\to$ Telugu** | **Moderate / Research** | Pipelined LID + Neural Transliteration | Preventing corruption of embedded English loanwords; segmenting hybrid morphological fusions (`college-ki`). |
| **English $\to$ Natural Tenglish** | **Low / No Standard** | Few-Shot Generative LLMs | Absolute absence of standardized training corpora; high variance in regional vernacular preferences; lack of objective evaluation metrics. |

---

# 6. Positioning of the Student's Implementation (NormMix AI)

Having established the 40-year evolution of machine translation, surveyed the current technological ecosystem, and formalized the research bottleneck surrounding Tenglish code-mixing, we now examine **NormMix AI**—an independent, student-developed local software system and neural framework.

It is vital to state the academic positioning clearly: **NormMix AI is not presented as an industrial replacement for hyperscale foundational translation platforms like Google Translate, Microsoft Translator, or IndicTrans2.** Rather, it is designed and positioned as an **academic case-study prototype and targeted experimental framework** engineered specifically to address the research gap identified in Section 5: **robust, real-time normalization, loanword-preserving transliteration, and multi-modal generation for informal Telugu–English code-mixed communication.**

```
                                  NORMMIX AI SYSTEM TOPOLOGY
                                  
+---------------------------------------------------------------------------------------------------+
| USER INTERACTION LAYER                                                                            |
|                                                                                                   |
|   +------------------------------------+          +-------------------------------------------+   |
|   | Enterprise Web Studio              |          | Manifest V3 Chrome Browser Extension      |   |
|   | (web/index.html)                   |          | (chrome-extension/)                       |   |
|   | - Phonetic Dropdown (1.నాకు 2.నాకూ) |          | - Browser Toolbar Popup Modal             |   |
|   | - Web Speech API (TTS & Voice)     |          | - Content Script Context-Menu On Webpages |   |
|   | - On-Screen Virtual Telugu Keyboard|          |   (WhatsApp Web, Twitter, Gmail, etc.)    |   |
|   +------------------------------------+          +-------------------------------------------+   |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  | HTTP REST API (/omni/process, /normalize)
                                                  v
+---------------------------------------------------------------------------------------------------+
| FASTAPI BACKEND MICROSERVICE (api/main.py)                                                        |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
| UNIVERSAL OMNI-PROCESSOR PIPELINE (src/pipeline/omni_processor.py)                                |
|                                                                                                   |
|  [Step 1: Unicode Text Cleaning & Sanitization]                                                   |
|       |                                                                                           |
|       v                                                                                           |
|  [Step 2: Token-Level Language Identification (LID)]                                              |
|       | -> Partitions tokens into: Pure English, Pure Telugu, Romanized Tanglish, or Punctuations |
|       v                                                                                           |
|  [Step 3: Multi-Modal Parallel Generation Engine]                                                 |
|       |                                                                                           |
|       +---> Mode 1: SOTA Normalized (BiGRU + Copy Gate Preserves English Loanwords)               |
|       |                                                                                           |
|       +---> Mode 2: All Telugu Script (Phonetic Transliteration of all words including loanwords) |
|       |                                                                                           |
|       +---> Mode 3: All Romanized Tenglish (Phonetic transliteration of all Telugu script to Latin)|
|       |                                                                                           |
|       +---> Mode 4: English Semantic Meaning & Gloss (Bilingual vocabulary mapping & translation) |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
| DEEP LEARNING INFERENCE ENGINE (src/models/ & src/attention/)                                     |
|                                                                                                   |
|   +-------------------------------------------------------------------------------------------+   |
|   | 2-Layer Bidirectional GRU Encoder (src/models/encoder.py)                                 |   |
|   | - Character-level embedding: captures fine-grained phonetic morphemes                     |   |
|   +-------------------------------------------------------------------------------------------+   |
|                                                 |                                                 |
|                                                 v                                                 |
|   +-------------------------------------------------------------------------------------------+   |
|   | Scaled Dot-Product Attention Mechanism (src/attention/scaled_dot_product.py)              |   |
|   | - Computes dynamic alignment weights alpha_{t,i} between decoder state and encoder memory |   |
|   +-------------------------------------------------------------------------------------------+   |
|                                                 |                                                 |
|                                                 v                                                 |
|   +-------------------------------------------------------------------------------------------+   |
|   | Pointer-Generator Copy Decoder (src/models/pointer_generator.py)                          |   |
|   | - Computes Generation Probability p_gen in [0, 1]                                         |   |
|   | - Copies English loanwords directly from source input sequence with 100% fidelity         |   |
|   | - Generates Telugu script characters from target vocabulary via softmax                   |   |
|   +-------------------------------------------------------------------------------------------+   |
+---------------------------------------------------------------------------------------------------+
```

### 6.1 System Conceptualization: An Experimental Case-Study Prototype

NormMix AI was developed to investigate how sequence-to-sequence neural architectures can be optimized for code-mixed text without requiring billion-parameter model footprints. Operating on consumer-grade hardware (specifically evaluated on an **NVIDIA GeForce RTX 4060 Laptop GPU with 8.0 GB GDDR6 VRAM**), the system explores:
1. Whether character-level recurrent encoders can outperform subword tokenizers on noisy Romanized orthography.
2. How copy mechanisms (Pointer-Generator Networks) solve the loanword corruption problem.
3. How to package these capabilities into accessible client-side tools (a Web Translation Studio and a Chrome Extension) for everyday digital communications.

### 6.2 Architectural Topology: 2-Layer BiGRU + Scaled Dot-Product Attention

Rather than adopting a massive Transformer that fragments Romanized text, NormMix AI deploys an optimized **Character-Level Sequence-to-Sequence Network** paired with a 2-Layer **Bidirectional Gated Recurrent Unit (BiGRU)** encoder:

#### 1. BiGRU Encoder Mathematical Formulation
Given an input sequence of characters $X = (x_1, x_2, \dots, x_{T_x})$ where each character is mapped to a dense embedding $e(x_i) \in \mathbb{R}^{d_e}$:
$$\vec{h}_i^{(1)} = \text{GRU}_{\text{fwd}}^{(1)}\left(e(x_i), \vec{h}_{i-1}^{(1)}\right), \quad \overleftarrow{h}_i^{(1)} = \text{GRU}_{\text{bwd}}^{(1)}\left(e(x_i), \overleftarrow{h}_{i+1}^{(1)}\right)$$
$$\vec{h}_i^{(2)} = \text{GRU}_{\text{fwd}}^{(2)}\left([\vec{h}_i^{(1)}; \overleftarrow{h}_i^{(1)}], \vec{h}_{i-1}^{(2)}\right), \quad \overleftarrow{h}_i^{(2)} = \text{GRU}_{\text{bwd}}^{(2)}\left([\vec{h}_i^{(1)}; \overleftarrow{h}_i^{(1)}], \overleftarrow{h}_{i+1}^{(2)}\right)$$
The final encoder representation for token $i$ is the concatenated hidden state:
$$h_i = \left[\vec{h}_i^{(2)}; \overleftarrow{h}_i^{(2)}\right] \in \mathbb{R}^{2 d_h}$$

#### 2. Scaled Dot-Product Attention
At decoding step $t$, with decoder hidden state $s_t \in \mathbb{R}^{d_s}$, the model computes attention energies against all encoder representations $h_i$:
$$e_{t,i} = \frac{s_t^T W_a h_i}{\sqrt{2 d_h}}$$
The scalar energy is normalized via softmax to produce an alignment distribution:
$$\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_{k=1}^{T_x} \exp(e_{t,k})}$$
The dynamic context vector $c_t$ is computed as the weighted expectation:
$$c_t = \sum_{i=1}^{T_x} \alpha_{t,i} h_i$$

### 6.3 The Pointer-Generator Copy Mechanism: Guaranteeing English Loanword Fidelity

The decisive architectural innovation in NormMix AI is the adaptation of the **Pointer-Generator Network** (originally formulated by See, Liu, and Manning, 2017 for text summarization) to code-mixed script normalization.

To prevent English loanwords (such as `college`, `interview`, `wifi`, `submit`) from being corrupted into phonetically mangled Telugu script, the decoder is equipped with a trainable **Generation Probability Gate** $p_{\text{gen}} \in [0, 1]$:

$$p_{\text{gen}} = \sigma\left(w_c^T c_t + w_s^T s_t + w_x^T e(y_{t-1}) + b_{\text{ptr}}\right)$$

Where $\sigma(\cdot)$ is the sigmoid activation function, $w_c, w_s, w_x$ are learnable weight vectors, and $b_{\text{ptr}}$ is a scalar bias.

The final probability of emitting any character $w$ is a mixture of two distinct probability distributions:
$$P(w) = p_{\text{gen}} P_{\text{vocab}}(w) + (1 - p_{\text{gen}}) \sum_{i: x_i = w} \alpha_{t,i}$$

```
                                  POINTER-GENERATOR COPY GATE
                                  
     Context Vector c_t ----+
     Decoder State s_t  ----+---> [Sigmoid Gate] ===> p_gen in [0, 1]
     Prev Output y_{t-1} ---+                               |
                                                            |
                     +--------------------------------------+--------------------------------------+
                     |                                                                             |
                     v (Weight: p_gen)                                                             v (Weight: 1 - p_gen)
     +-------------------------------+                                             +-------------------------------+
     | Vocabulary Distribution       |                                             | Attention Copy Distribution   |
     | P_vocab(w) = Softmax(V [s; c])|                                             | sum_{i: x_i = w} alpha_{t,i}  |
     | (Generates Telugu Characters) |                                             | (Copies Latin English Words)  |
     +-------------------------------+                                             +-------------------------------+
                     |                                                                             |
                     +-------------------------------------> (+) <---------------------------------+
                                                              |
                                                              v
                                              Final Output Character Distribution P(w)
```

**Computational Significance:**
When the model processes a native Romanized Telugu morpheme (e.g., `nenu`), $p_{\text{gen}}$ trends toward $1.0$, directing the decoder to generate Telugu Unicode characters (`నేను`) from the target vocabulary. Conversely, when the model encounters an embedded English loanword (e.g., `college`), $p_{\text{gen}}$ drops toward $0.0$, instructing the decoder to directly copy the source ASCII characters from the input buffer. **This completely eliminates loanword corruption, achieving 100% preservation of English technical vocabulary.**

---

### 6.4 Omni-Directional Processing Pipeline (`omni_processor.py`)

In production serving, user inputs do not arrive with neat modal labels. A user may enter pure English, pure Telugu script, Romanized Tenglish, or a bi-scriptal hybrid.

NormMix AI solves this through the `OmniProcessor` pipeline (`src/pipeline/omni_processor.py`), which executes a multi-stage flow:
1. **Sanitization & Edge Segmentation:** Cleans punctuation, normalizes Unicode diacritics, and strips surrounding whitespace.
2. **Token-by-Token Language Identification (LID):** Evaluates every whitespace-delimited token using character-range inspection, lexicon membership against a 64,421-entry verified vocabulary, and morphological affix matching. Tokens are classified into:
   - `pure_english`: Standard English vocabulary.
   - `telugu_script`: Unicode characters in range `0x0C00`–`0x0C7F`.
   - `tanglish`: Romanized Telugu morphemes.
   - `punctuation / numeric`: Preserved invariant tokens.
3. **Simultaneous Multi-Modal Target Synthesis:**
   Rather than producing a single output, the OmniProcessor simultaneously computes **four orthogonal representations** for every query:
   - **Target 1 (Standard Normalized):** Telugu morphemes normalized to Telugu Unicode script; English loanwords preserved in Latin script via the Pointer-Generator copy gate.
   - **Target 2 (All Telugu Script):** Every word—including English loanwords—is transliterated into Telugu Unicode script (e.g., `college` $\to$ `కాలేజీ`).
   - **Target 3 (All Romanized / Tenglish):** Any native Telugu script is phonetically Romanized into Latin characters (`కాలేజీ` $\to$ `college / kaaleeji`).
   - **Target 4 (English Semantic Meaning & Gloss):** High-frequency Telugu roots and postpositions are translated into their English gloss equivalents, providing immediate bilingual comprehension.

---

### 6.5 The User Interface Layer: Web Translation Studio and Manifest V3 Chrome Extension

To validate real-world utility, the framework is packaged with two front-end implementations:

#### 1. Enterprise Web Translation Studio (`web/index.html`)
*   **Google Input Tools Style Suggestion Dropdown:** Intercepts keystrokes in real time; queries the local FastAPI `/dictionary/lookup` endpoint; displays a dynamic floating candidate menu (e.g., typing `naku` renders: `1. నాకు  2. నాకూ  3. నకు`).
*   **Web Speech API Integration:** Enables voice dictation via microphone input (`🎙️ Speak`) and speech synthesis audio playback (`🔊 Listen`).
*   **Virtual On-Screen Telugu Keyboard:** An interactive on-screen drawer allowing users without physical Telugu keycaps to click Telugu consonants, independent vowels, and Guṇintālu diacritics.
*   **Active Learning Feedback Loop:** Users can submit thumbs-up (`👍`) or thumbs-down (`👎`) evaluations, which append real-world corrections directly to `data/feedback.jsonl` for continuous model retraining.

#### 2. Manifest V3 Chrome Browser Extension (`chrome-extension/`)
*   **Architecture:** Complies strictly with Google Chrome Manifest V3 standards, utilizing an asynchronous background service worker (`background.js`), a popup modal interface (`popup.html` / `popup.js`), and injected content scripts (`content.js`).
*   **Context-Menu Webpage Normalization:** Users highlighting Romanized Tenglish on any web surface (e.g., reading a tweet on X, an email in Gmail, or a message on WhatsApp Web) can right-click and select **"Normalize Tenglish with NormMix"**. The content script queries the local FastAPI daemon (`http://127.0.0.1:8000/normalize`) and replaces or overlays the normalized Telugu text in situ.

---

### 6.6 Empirical Benchmarks and Ablation Analysis

NormMix AI was evaluated against rigorous zero-leakage test partitions sourced from the **Google Dakshina Dataset** (Roark et al., 2020), **AI4Bharat Aksharantar** (Madhani et al., 2023), and synthetic colloquial chat corpora.

The benchmark suite tracks five standard NLP metrics:
1. **Character Error Rate (CER):** Levenshtein distance at the character level normalized by reference length ($\downarrow$).
2. **BLEU (Bilingual Evaluation Understudy):** Modified $n$-gram precision score ($\uparrow$).
3. **chrF++:** Character $n$-gram F-score with word bi-grams ($\uparrow$).
4. **English Token Preservation Accuracy:** Percentage of English loanwords retained without script corruption ($\uparrow$).
5. **Unseen Vocabulary BLEU:** Generalization performance on words absent from the training set ($\uparrow$).

```
+-------------------------------------------------------------------------------------------------------------------------+
|                                    NORMMIX EMPIRICAL BENCHMARK ABLATION TABLE                                           |
+------------------------------------+------------+-------------+------------+-------------+--------------+---------------+
| Model Architecture                 | Parameters | Gold CER    | Gold BLEU  | Gold chrF   | English Acc  | Unseen BLEU   |
+------------------------------------+------------+-------------+------------+-------------+--------------+---------------+
| 1. Rule Baseline (64k Lexicon)     | --         | 0.1720      | 58.40      | 78.10       | 100.0%       | 14.20         |
| 2. Vanilla Seq2Seq (No Attention)  | 2.10M      | 0.4820      | 22.10      | 41.30       | 12.5%        | 8.40          |
| 3. Seq2Seq + Bahdanau Attention    | 2.78M      | 0.3120      | 44.50      | 62.80       | 38.2%        | 18.90         |
| 4. Seq2Seq + Luong Attention       | 2.78M      | 0.2940      | 48.20      | 66.40       | 42.0%        | 21.50         |
| 5. Standard Transformer (Small)    | 1.85M      | 0.3850      | 35.10      | 54.20       | 29.4%        | 15.10         |
| 6. NormMix SOTA (BiGRU + Copy)     | 3.05M      | 0.0537      | 79.84      | 91.29       | 100.0%       | 39.67         |
+------------------------------------+------------+-------------+------------+-------------+--------------+---------------+
```

**Key Empirical Insights:**
*   **The Power of the Copy Gate:** Vanilla Seq2Seq models suffered an abysmal 12.5% English loanword accuracy, as they attempted to phonetically decode Latin words into Telugu script. The Pointer-Generator mechanism achieved **100% preservation**.
*   **Character Error Reduction:** Character Error Rate on the gold benchmark dropped by an order of magnitude: from 0.4820 (Vanilla) down to **0.0537 (5.37%)** in the proposed BiGRU + Copy architecture.
*   **Inference Latency:** Because the BiGRU model contains only 3.05 million parameters, inference latency on an RTX 4060 GPU is **8.2 milliseconds per sequence**, enabling true real-time, sub-keystroke interactive suggestions.

---

# 7. Comprehensive Comparative Evaluation

### 7.1 Cross-System Feature Matrix

The following comprehensive matrix evaluates the student's system (NormMix AI) alongside leading commercial and research platforms.

**Legend:**
- ✅ **Supported:** Officially supported as a core production capability.
- ⚠️ **Partial / Limited:** Constrained support, heuristic workaround, or fragile quality.
- ❌ **Not Primarily Supported:** Unsupported or actively corrupts output.
- 🔬 **Research / Experimental:** Demonstrated in research prototypes or academic checkpoints.

| Evaluation Dimension | Google Translate | Microsoft Translator | DeepL | IndicXlit (AI4Bharat) | IndicTrans2 (AI4Bharat) | SeamlessM4T (Meta) | Sarvam AI | NormMix AI (Student System) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Telugu $\to$ English (Script)** | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ⚠️ (Gloss Only) |
| **English $\to$ Telugu (Script)** | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ⚠️ (Dictionary Only) |
| **Telugu $\to$ Roman Telugu** | ⚠️ | ⚠️ | ❌ | ✅ | ❌ | ❌ | ⚠️ | ✅ |
| **Roman Telugu $\to$ Telugu** | ⚠️ | ❌ | ❌ | ✅ | ❌ | ❌ | ⚠️ | ✅ |
| **Tenglish $\to$ English** | ⚠️ | ❌ | ❌ | ❌ | ❌ | ⚠️ | ⚠️ | ✅ (Target 4 Gloss) |
| **Tenglish $\to$ Telugu Script** | ⚠️ | ❌ | ❌ | ⚠️ | ❌ | ❌ | ⚠️ | ✅ (Target 1 & 2) |
| **English $\to$ Tenglish** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | 🔬 | ✅ (Target 3) |
| **Code-Mixed Input Resilience** | ⚠️ | ❌ | ❌ | ❌ | ❌ | ⚠️ (Speech) | ⚠️ (Prompt) | ✅ (Design Focus) |
| **English Loanword Copy Gate**| ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ⚠️ | ✅ (Pointer-Gen) |
| **Real-time Phonetic Suggestions**| ⚠️ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (Web UI) |
| **Speech STT / TTS** | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ (Web Speech API) |
| **Chrome Extension Integration**| ✅ (Full Page)| ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (Manifest V3) |
| **Offline / Local Inference** | ⚠️ (Mobile) | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ (Local CUDA/CPU)|
| **Custom Architecture Tuning** | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ (Modular PyTorch)|

---

### 7.2 What Existing Systems Already Do Exceptionally Well

Academic integrity demands acknowledging the extraordinary achievements of existing production platforms:
1. **High-Resource Script Translation:** Google Translate, Microsoft Translator, and IndicTrans2 possess vast parallel data foundations (exceeding 10 million sentence pairs). On formal, grammatically pristine Telugu script text (news editorials, legal statutes, literature), these systems generate fluent, high-fidelity English translations that cannot be matched by small prototypes.
2. **Acoustic Speech Modeling:** Meta's SeamlessM4T and Sarvam AI's Saaras/Bulbul have solved major components of the acoustic challenge. By training end-to-end models directly on spoken audio waveforms, they bypass the chaotic spelling inconsistencies of written Tenglish altogether.
3. **Massive Transliteration Scale:** AI4Bharat’s IndicXlit, trained on the 26-million-pair Aksharantar corpus, possesses unmatched coverage of Indian named entities, surnames, and geographical locations.

### 7.3 Where Existing Commercial and Research Systems Fall Short

1. **The Code-Mixed Orthography Gap:** Commercial engines are trained under the assumption that an input sentence is either entirely English or entirely Telugu script. When exposed to an informal, unstandardized sentence like `nenu ivala clg lo test submit chesi vasthe baguntundi bro`, commercial tokenizers fragment the Romanized text, producing gibberish translations or silent omissions.
2. **Loanword Transliteration Corruption:** Specialized transliteration tools like IndicXlit lack an innate loanword copy mechanism; they assume every Latin character string must be mapped into Telugu script, corrupting standard English technical vocabulary (`college` $\to$ `కోల్లెగె`).
3. **Lack of In Situ Browser Normalization:** Users browsing social media or web-based messaging clients (WhatsApp Web, Reddit, LinkedIn) have no lightweight mechanism to normalize selected Tenglish phrases directly within their browser workflow without copy-pasting text into an external translation window.

### 7.4 What the Student's Implementation Specifically Solves

NormMix AI occupies a precise, well-defined operational niche:
1. **Loanword-Preserving Normalization:** By integrating a **Pointer-Generator Copy Mechanism**, NormMix AI dynamically determines whether to generate a Telugu script character or copy an English Latin character. It is explicitly engineered not to corrupt English technical loanwords.
2. **Omni-Directional Multi-Modal Generation:** Instead of forcing the user into a rigid single-direction pipeline, the system simultaneously synthesizes four distinct output perspectives: Standard Normalized, All Telugu Script, All Romanized Tenglish, and English Semantic Gloss.
3. **End-to-End Client Ecosystem:** The implementation demonstrates full-stack integration: an RTX 4060 GPU-accelerated PyTorch backend, a high-throughput FastAPI REST microservice, an enterprise-styled Web Translation Studio with real-time phonetic suggestions and voice audio, and a fully functional Manifest V3 Chrome Extension.

### 7.5 What Remains Unresolved: Academic Limitations of the Current Prototype

In accordance with strict academic standards, the student system's current limitations must be openly stated:
1. **Lexical Semantic Depth:** While NormMix AI performs normalization and bilingual semantic glossing with high accuracy, it does not possess a deep generative decoder capable of synthesizing long, complex, multi-clause English prose with full discourse coherence. It relies on bilingual lexicon mapping rather than a 10-billion-parameter language generation engine.
2. **Dialectal Diversity:** The training corpus is predominantly grounded in standard Coastal Andhra and formal Google Dakshina Romanizations. It exhibits reduced accuracy when encountering non-standard regional dialects (e.g., colloquial Telangana slang or Rayalaseema idioms).
3. **Dependency on Clean Input Tokenization:** While resilient to character-level spelling noise, severe whitespace truncation (e.g., run-on strings like `nenucollegekivelthunna`) degrades the token-level language identification pipeline.

---

# 8. Conclusion and Future Scope

### 8.1 Summary of Findings Across Ten Architectural Dimensions

1. **Evolutionary Trajectory:** Telugu language computation has successfully transitioned across six technical epochs—from rule-based expert systems (TDIL) and Paninian grammar frameworks (Anusaaraka), through statistical modeling (Moses) and recurrent networks (GNMT), to dense multilingual Transformers (IndicTrans2) and multimodal foundation models (SeamlessM4T, Sarvam AI).
2. **Maturity of Canonical Translation:** Bidirectional translation between formal, standardized Telugu Unicode script and English is a technologically mature discipline with high production accuracy.
3. **The Digital Vernacular Divide:** The true unsolved bottleneck in Telugu NLP is the computational processing of informal, Romanized, code-mixed **Tenglish**, which dominates real-world digital communication on mobile and web platforms.
4. **Failure of Subword Tokenizers:** Standard BPE and SentencePiece tokenizers, pre-trained primarily on English or formal native scripts, fragment Romanized Telugu words into arbitrary byte shards, destroying semantic representations.
5. **The Loanword Preservation Dilemma:** Naive neural transliteration engines blindly convert English loanwords into Telugu script, degrading text comprehension.
6. **The Pointer-Generator Solution:** Introducing a trainable copy mechanism ($p_{\text{gen}}$) enables neural networks to selectively emit target-script characters while copying source loanword characters with 100% fidelity.
7. **Omni-Directional Value:** Real-world code-mixing requires flexible, multi-modal outputs (Standard Normalized, Full Script, Full Romanized, and Semantic Gloss) rather than rigid single-target translation.
8. **Client-Side Accessibility:** The combination of an interactive Web Translation Studio and an in situ Manifest V3 Chrome Extension bridges the gap between academic neural models and daily user communication workflows.
9. **Empirical Validation:** Rigorous benchmarking confirms that a compact 3.05M-parameter BiGRU model with attention and copy mechanisms can achieve a low 5.37% Character Error Rate and 8.2 ms latency on consumer GPU hardware.
10. **Academic Honest Assessment:** Compact task-specific normalization models excel at low-latency script restitution and loanword preservation, but must ultimately be paired with large foundational MT models to achieve full literary prose translation.

---

### 8.2 Future Research Roadmap (Seventeen Concrete Vectors)

The findings of this case study chart seventeen explicit pathways for future research in Dravidian computational linguistics:

```
+---------------------------------------------------------------------------------------------------+
|                                  SEVENTEEN FUTURE RESEARCH VECTORS                                |
+---------------------------------------------------------------------------------------------------+
|  1. Context-Aware Tenglish Normalization: Scaling to multi-sentence discourse contexts.           |
|  2. Personalized Phonetic Spelling Adaptation: User-specific typing profiling and adaptation.     |
|  3. Slang and Colloquial Neologism Lexicons: Dynamic updates for viral social media slang.       |
|  4. Deep Generative Tenglish-to-English MT: End-to-end LLM fine-tuning (LoRA / QLoRA).             |
|  5. Sub-Token Intra-Word Language Identification: Fine-grained morpheme segmentation.             |
|  6. End-to-End Speech-to-Tenglish Transcription: Direct acoustic decoding into Romanized text.   |
|  7. Tenglish-Aware Automatic Speech Recognition (ASR): Joint acoustic-phonetic modeling.         |
|  8. Low-Latency Real-Time Meeting Interpretation: Sub-500ms streaming translation.                |
|  9. System-Wide Browser Accessibility: Full DOM tree live translation and screen-reader support.  |
| 10. Multimodal Optical Character Recognition (OCR): Recognizing Tenglish in memes and video text.|
| 11. Foundation Model Quantization (INT4/GGUF): On-device edge execution on mobile NPU hardware.   |
| 12. Telugu Dialectal Robustness: Dedicated corpora for Telangana, Rayalaseema, and North Andhra.  |
| 13. Domain-Specific Normalization: Specialized adaptations for medical, legal, and banking chat.  |
| 14. Standardized Public Tenglish Benchmarks: Curating open, human-annotated evaluation test sets.  |
| 15. Cross-Dravidian Code-Mixed Transfer: Zero-shot transfer learning to Tanglish and Kanglish.    |
| 16. Active Learning and Continuous Human Feedback: Incorporating production telemetry safely.     |
| 17. Privacy-Preserving On-Device Inference: Fully local processing to safeguard personal chats.   |
+---------------------------------------------------------------------------------------------------+
```

1. **Context-Aware Disambiguation:** Extending sequence context beyond isolated sentences to full conversational threads to resolve homophonic collisions (`kadu` $\to$ `కాదు` vs. `కడు`).
2. **Personalized Typing Profiles:** Implementing adaptive decoders that learn individual user orthographic idiosyncrasies over time.
3. **Colloquial Neologism Harvesting:** Developing unsupervised scraping pipelines to continuously ingest emerging Telugu slang from YouTube, Instagram, and Reddit.
4. **End-to-End LLM Fine-Tuning:** Applying Low-Rank Adaptation (LoRA / QLoRA) to open models like Sarvam-1 or LLaMA-3 using synthetic code-mixed parallel corpora.
5. **Sub-Token Morpheme Segmentation:** Building specialized neural segmenters capable of dissecting hybrid words (`college-ki` $\to$ `[college]` + `[-ki]`) prior to decoding.
6. **Speech-to-Tenglish Transcription:** Fine-tuning Whisper and IndicConformer to output Romanized Tenglish directly for users who prefer reading Latin script.
7. **Conversational ASR for Code-Mixing:** Resolving acoustic code-switching boundaries in live conversational speech.
8. **Streaming Low-Latency Translation:** Implementing chunk-based attention for simultaneous interpretation in live video calls.
9. **Deep Browser DOM Translation:** Enhancing the Chrome Extension to seamlessly translate dynamic, asynchronous single-page applications without altering webpage layout.
10. **Multimodal Meme and Image OCR:** Building vision-language models capable of parsing Tenglish captions rendered on image graphics and memes.
11. **Edge Deployment and Mobile NPU Quantization:** Quantizing models to INT4 and INT8 via ONNX Runtime and ExecuTorch for execution on smartphone Neural Processing Units without internet connectivity.
12. **Dialectal Diversity Benchmarks:** Curating diverse linguistic corpora capturing Telangana, Rayalaseema, and Coastal regional expressions.
13. **Domain-Specific Adaptation:** Training specialized code-mixing models for banking chatbots, telemedicine, and legal citizen services.
14. **Standardized Community Benchmarks:** Establishing an open, public evaluation benchmark (akin to GLUE or FLORES) specifically for Romanized Indic code-mixing.
15. **Cross-Dravidian Transfer Learning:** Leveraging shared structural typologies between Telugu, Tamil (Tanglish), and Kannada (Kanglish) to train joint multi-code-mixed models.
16. **Production Active Learning Pipelines:** Safely integrating user feedback (`feedback.jsonl`) into continuous automated retraining pipelines with regression testing.
17. **Privacy-Preserving Local Inference:** Ensuring that sensitive instant messaging text never leaves the user's local device, upholding strict data privacy standards.

---

### 8.3 Central Academic Conclusion

The evolution of natural language processing for the Telugu language mirrors the broader history of artificial intelligence: a continuous progression from rigid, handcrafted symbolic formalisms toward flexible, data-driven representation learning. 

Today, standard machine translation between formal Telugu script and English can no longer be characterized as an unsolved scientific mystery; it is an established engineering discipline driven by mature neural foundation systems. **The emerging grand challenge of South Asian computational linguistics is understanding and transforming the informal, Romanized, and code-mixed language actually spoken and written by millions of people in daily digital life.** 

Bridging this gap requires systems that respect the delicate interplay between native Dravidian morphology and English loanword vocabulary. Experimental frameworks like NormMix AI demonstrate that targeted, linguistically motivated neural architectures—combining attention-based representation with copy-mechanism guarantees—offer an effective, low-latency, and accessible path toward solving the real-world communication challenges of the modern digital era.

---

# 9. References

1. **Akshar Bharati, V. Chaitanya, and R. Sangal**, *Natural Language Processing: A Paninian Perspective*, Prentice-Hall of India, New Delhi, 1995.
2. **P. F. Brown, V. J. Della Pietra, S. A. Della Pietra, and R. L. Mercer**, "The mathematics of statistical machine translation: Parameter estimation," *Computational Linguistics*, vol. 19, no. 2, pp. 263–311, 1993.
3. **P. Koehn, H. Hoang, A. Birch, C. Callison-Burch, M. Federico, N. Bertoldi, B. Cowan, W. Shen, C. Moran, R. Zens, C. Dyer, O. Bojar, A. Constantin, and E. Herbst**, "Moses: Open source toolkit for statistical machine translation," in *Proc. 45th Annual Meeting of the Association for Computational Linguistics (ACL)*, Companion Volume, Prague, Czech Republic, 2007, pp. 177–180.
4. **F. J. Och and H. Ney**, "A systematic comparison of various statistical alignment models," *Computational Linguistics*, vol. 29, no. 1, pp. 19–51, 2003.
5. **I. Sutskever, O. Vinyals, and Q. V. Le**, "Sequence to sequence learning with neural networks," in *Advances in Neural Information Processing Systems (NeurIPS 2014)*, Montreal, Canada, vol. 27, 2014, pp. 3104–3112.
6. **K. Cho, B. van Merriënboer, Ç. Gülçehre, D. Bahdanau, F. Bougares, H. Schwenk, and Y. Bengio**, "Learning phrase representations using RNN encoder-decoder for statistical machine translation," in *Proc. Conference on Empirical Methods in Natural Language Processing (EMNLP)*, Doha, Qatar, 2014, pp. 1724–1734.
7. **D. Bahdanau, K. Cho, and Y. Bengio**, "Neural machine translation by jointly learning to align and translate," in *Proc. 3rd International Conference on Learning Representations (ICLR)*, San Diego, CA, 2015.
8. **M.-T. Luong, H. Pham, and C. D. Manning**, "Effective approaches to attention-based neural machine translation," in *Proc. Conference on Empirical Methods in Natural Language Processing (EMNLP)*, Lisbon, Portugal, 2015, pp. 1412–1421.
9. **Y. Wu, M. Schuster, Z. Chen, Q. V. Le, M. Norouzi, W. Macherey, M. Krikun, Y. Cao, Q. Gao, K. Macherey, et al.**, "Google's neural machine translation system: Bridging the gap between human and machine translation," *arXiv preprint arXiv:1609.08144*, 2016.
10. **A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin**, "Attention is all you need," in *Advances in Neural Information Processing Systems (NeurIPS 2017)*, Long Beach, CA, vol. 30, 2017, pp. 5998–6008.
11. **A. See, P. J. Liu, and C. D. Manning**, "Get to the point: Summarization with pointer-generator networks," in *Proc. 55th Annual Meeting of the Association for Computational Linguistics (ACL)*, Vancouver, Canada, 2017, pp. 1073–1083.
12. **B. Roark, L. Wolf-Sonkin, C. Gorman, I. Kadenko, R. Gheewala, C. Tillmann, N. Erdmann, and A. Khullar**, "Processing South Asian languages written in the Latin script: The Dakshina dataset," in *Proc. 12th Language Resources and Evaluation Conference (LREC)*, Marseille, France, 2020, pp. 2413–2423.
13. **G. Ramesh, S. Doddapaneni, A. Bheemambika, R. Akkapeddi, H. Aralikatte, K. Jha, J. Raman, M. A. K, R. Puduppully, P. Kumar, and M. M. Khapra**, "Samanantar: The largest publicly available parallel corpora collection for Indic languages," in *Proc. 60th Annual Meeting of the Association for Computational Linguistics (ACL)*, Dublin, Ireland, 2022, pp. 8543–8560.
14. **Y. Madhani, S. Parida, R. S. M, A. Doddapaneni, V. Seshadri, K. Jha, A. K, J. Raman, A. Bheemambika, P. Kumar, and M. M. Khapra**, "Aksharantar: Towards building open transliteration systems for Indian languages," in *Proc. 61st Annual Meeting of the Association for Computational Linguistics (ACL)*, Toronto, Canada, 2023, pp. 11234–11255.
15. **J. Gala, P. Baswani, K. S. M, C. B. A, B. D. N, A. Doddapaneni, R. S. M, K. Sreedhar, K. Jha, A. Bheemambika, P. Kumar, and M. M. Khapra**, "IndicTrans2: Towards high-quality and accessible machine translation models for all 22 scheduled Indian languages," *arXiv preprint arXiv:2305.16307*, 2023.
16. **D. Kakwani, A. K, S. Parida, C. B. A, J. Raman, A. Bhattacharyya, M. M. Khapra, and P. Kumar**, "IndicNLPSuite: Monolingual corpora, language models and evaluation benchmarks for Indian languages," in *Proc. Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 2020, pp. 3501–3514.
17. **L. Barrault, Y.-A. Chung, M. C. Meglioli, D. Dale, N. Dong, M. Duquenne, H. Elsahar, H. Gao, J. Fu, P. A. Algom, et al.**, "SeamlessM4T: Massively multilingual and multimodal machine translation," *arXiv preprint arXiv:2308.11596*, Meta AI, 2023.
18. **Pratyush Kumar, Vivek Raghavan, et al.**, "Sarvam-1: An efficient foundation language model for Indian languages," *Sarvam AI Technical Report*, Bangalore, India, 2024.
19. **R. M. K. Sinha and A. Jain**, "Anglabharti-II: A machine translation system with modern architecture," in *Proc. International Conference on Machine Translation (MT Summit IX)*, New Orleans, LA, 2003.
20. **R. Sangal and V. Chaitanya**, "Anusaaraka: Overcoming the language barrier," *Information Technology in Developing Countries*, vol. 6, no. 1, pp. 2–5, 1996.
21. **D. Sharma and R. Sangal**, "Computational linguistics in India: An overview," in *Language and Technology in India*, Central Institute of Indian Languages (CIIL), Mysore, 2012.
22. **K. Bali, J. Sharma, M. Choudhury, and Y. Vyas**, "“I am borrowing ya mixing?” An analysis of English-Hindi code-mixing in Facebook," in *Proc. 1st Workshop on Computational Approaches to Code Switching (EMNLP 2014)*, Doha, Qatar, 2014, pp. 116–126.
23. **Y. Vyas, S. Gella, J. Sharma, K. Bali, and M. Choudhury**, "POS tagging of English-Hindi code-mixed social media content," in *Proc. Conference on Empirical Methods in Natural Language Processing (EMNLP)*, Doha, Qatar, 2014, pp. 974–979.
24. **A. Joshi**, "Processing of sentences with intra-sentential code-switching," in *Natural Language Parsing: Psychological, Computational, and Theoretical Perspectives*, Cambridge University Press, 1985, pp. 191–205.
25. **S. Poplack**, "Sometimes I’ll start a sentence in Spanish y termino en español: Toward a typology of code-switching," *Linguistics*, vol. 18, no. 7–8, pp. 581–618, 1980.
26. **Microsoft Corporation**, "Azure AI Translator Documentation: Customization and Neural Machine Translation for Indian Languages," *Microsoft Learn Technical Documentation*, 2024. [Online]. Available: https://learn.microsoft.com/azure/ai-services/translator/
27. **Google Cloud**, "Cloud Translation API v3: Advanced Features and Supported Indic Languages," *Google Cloud Documentation*, 2024. [Online]. Available: https://cloud.google.com/translate/docs
28. **Ministry of Electronics and Information Technology (MeitY)**, "Digital India Bhashini: Democratizing Language Technologies for Inclusive Digital Governance," *Government of India Official Portal*, 2024. [Online]. Available: https://bhashini.gov.in/
29. **M. Post**, "A call for clarity in reporting BLEU scores," in *Proc. 3rd Conference on Machine Translation (WMT 2018)*, Brussels, Belgium, 2018, pp. 186–191.
30. **M. Popović**, "chrF: character n-gram F-score for machine translation evaluation," in *Proc. 10th Workshop on Statistical Machine Translation (WMT 2015)*, Lisbon, Portugal, 2015, pp. 392–395.
31. **T. Kudo and J. Richardson**, "SentencePiece: A simple and language independent subword tokenizer and detokenizer for neural text processing," in *Proc. Conference on Empirical Methods in Natural Language Processing (EMNLP)*: System Demonstrations, Brussels, Belgium, 2018, pp. 66–71.
32. **R. Sennrich, B. Haddow, and A. Birch**, "Neural machine translation of rare words with subword units," in *Proc. 54th Annual Meeting of the Association for Computational Linguistics (ACL)*, Berlin, Germany, 2016, pp. 1715–1725.
33. **S. Hochreiter and J. Schmidhuber**, "Long short-term memory," *Neural Computation*, vol. 9, no. 8, pp. 1735–1780, 1997.
34. **P. Bhattacharyya**, *Machine Translation*, CRC Press, Taylor & Francis Group, Boca Raton, FL, 2015.
35. **N. Gala et al.**, "Evaluating linguistic diversity and cultural preservation in Indic foundation models," *Transactions of the Association for Computational Linguistics (TACL)*, vol. 12, pp. 412–430, 2024.
