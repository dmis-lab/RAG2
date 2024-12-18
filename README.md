# Rationale-Guided Retrieval Augmented Generation for Medical Question Answering

**Paper** | [Rationale-Guided Retrieval Augmented Generation for Medical Question Answering](https://arxiv.org/abs/2411.00300)

**Authors**: Jiwoong Sohn, Yein Park, Chanwoong Yoon, Sihyeon Park, Hyeon Hwang, Mujeen Sung, Hyunjae Kim, Jaewoo Kang

**Abstract**: Large language models (LLM) hold significant potential for applications in biomedicine, but they struggle with hallucinations and outdated knowledge. While retrieval-augmented generation (RAG) is generally employed to address these issues, it also has its own set of challenges: (1) LLMs are vulnerable to irrelevant or incorrect context, (2) medical queries are often not well-targeted for helpful information, and (3) retrievers are prone to bias toward the specific source corpus they were trained on. In this study, we present RAG² (RAtionale-Guided RAG), a new framework for enhancing the reliability of RAG in biomedical contexts. RAG² incorporates three key innovations: a small filtering model trained on perplexity-based labels of rationales, which selectively augments informative snippets of documents while filtering out distractors; LLM-generated rationales as queries to improve the utility of retrieved snippets; a structure designed to retrieve snippets evenly from a comprehensive set of four biomedical corpora, effectively mitigating retriever bias. Our experiments demonstrate that RAG² improves the state-of-the-art LLMs of varying sizes, with improvements of up to 6.1%, and it outperforms the previous best medical RAG model by up to 5.6% across three medical question-answering benchmarks.

**Repository Overview**

This repository contains the implementation of **Rationale-Guided Retrieval-Augmented Generation (RAG²)**. It includes code for training the filtering model, setting up the retriever, and running inference. The repository is organized as follows:

**Retriever Setup**

The retriever is based on [Self-BioRAG](https://github.com/dmis-lab/self-biorag/tree/main/retriever). To set it up, Clone the Self-BioRAG repository and follow the setup instructions provided in the [Self-BioRAG retriever documentation](https://github.com/dmis-lab/self-biorag/tree/main/retriever).

**Filtering Model Training**
The filtering model training code is based on [Adaptive-RAG](https://github.com/starsuzi/Adaptive-RAG).

**Inference**

### Citation
If you use this work, please cite our paper:

```
@article{sohn2024rag,
  title={Rationale-Guided Retrieval Augmented Generation for Medical Question Answering},
  author={Jiwoong Sohn and Yein Park and Chanwoong Yoon and Sihyeon Park and Hyeon Hwang and Mujeen Sung and Hyunjae Kim and Jaewoo Kang},
  journal={arXiv preprint arXiv:2411.00300},
  year={2024}
}
```

Stay tuned for updates on data and code!
