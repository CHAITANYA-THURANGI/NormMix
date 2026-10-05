# Required Datasets

## Essential Datasets (Tier 1)
1. **Dakshina Dataset** (Google Research)
   - URL: https://github.com/google-research-datasets/dakshina
   - License: CC BY 4.0
   - Download: `git clone https://github.com/google-research-datasets/dakshina.git`
   - Copy Telugu files to: `data/external/dakshina/te/`
   - Expected files: `te.translit.sampled.{train,dev,test}.tsv`, `te.lexicon.{train,dev,test}.tsv`

2. **Aksharantar** (AI4Bharat)
   - URL: https://huggingface.co/datasets/ai4bharat/Aksharantar
   - License: CC0
   - Download Telugu subset and place in: `data/external/aksharantar/te/`

## Recommended Datasets (Tier 2)
3. **IndicCorp v2** (Telugu subset)
   - URL: AI4Bharat HuggingFace
   - License: CC-0
   - Purpose: Clean Telugu sentences for synthetic data generation

4. **Samanantar** (Telugu-English)
   - URL: https://huggingface.co/datasets/ai4bharat/samanantar
   - License: CC-BY-NC-4.0
   - Purpose: Parallel translation corpus

## Experimental Datasets (Tier 3)
5. **DravidianLangTech** shared task data
   - Available on Kaggle
   - Purpose: Real noisy social media Telugu-English text
