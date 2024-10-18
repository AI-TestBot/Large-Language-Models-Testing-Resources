# Large Language Models (LLMs) Testing Resources

## 📒Introduction
Large Language Models (LLMs) Testing Resources: A curated list of Awesome LLMs Testing Papers with Codes, check [📖Contents](#paperlist) for more details. This repo is still updated frequently ~ 👨‍💻‍ **Welcome to star ⭐️ or submit a PR to this repo! I will review and merge it.**

## 📖Contents 
* 📖[Leaderboard](#Leaderboard)
* 📖[Review](#Review)
* 📖[General](#General)
  * 📖[Comprehensive](#Comprehensive)
  * 📖[Understanding](#Understanding)
  * 📖[Generation](#Generation)
  * 📖[Reasoning](#Reasoning)
  * 📖[Knowledge](#Knowledge)
  * 📖[Discipline](#Discipline)
  * 📖[Multilingual](#Multilingual)
  * 📖[Long-Context](#Long-Context)
  * 📖[Chain-of-Thought](#Chain-of-Thought)
  * 📖[Role-Playing](#Role-Playing)
  * 📖[Tools](#Tools)
  * 📖[Instruction-Following](#Instruction-Following)
  * 📖[Reliable](#Reliable)
  * 📖[Robust](#Robust)
* 📖[Application](#Application)
  * 📖[Comprehensive](#Comprehensive)
  * 📖[Chatbot](#Chatbot)
  * 📖[Knowledge-Management](#Knowledge-Management)
  * 📖[Data-Analysis](#Data-Analysiss)
  * 📖[Code-Assistant](#Code-Assistant)
  * 📖[Office-Assistant](#Office-Assistant)
  * 📖[Content-Generation](#Content-Generation)
  * 📖[TaskPlanning](#TaskPlanning)
  * 📖[Agent](#Agent)
  * 📖[EmbodiedAI](#EmbodiedAI)
* 📖[Security](#Security)
  * 📖[Content-Security](#Content-Security)
  * 📖[Value-Aligement](#Value-Aligement)
  * 📖[Model-Security](#Model-Security)
  * 📖[Privacy-Security](#Privacy-Security)
* 📖[Industry](#Industry)
  * 📖[Finance](#Finance)
  * 📖[Medical](#Medical)
  * 📖[Law](#Law)
  * 📖[Engineering](#Engineering)
  * 📖[Education](#Education)
  * 📖[Research](#Research)
  * 📖[Goverment-Affairs](#Goverment-Affairs)
  * 📖[Communication](#Communication)
  * 📖[Power](#Power)
  * 📖[Transportation](#Transportation)
  * 📖[Industry](#Industry)
  * 📖[Media](#Media)
  * 📖[Design](#Design)
  * 📖[Internet](#Internet)
  * 📖[Robot](#Robot)
  * 📖[Game](#Game)
* 📖[Human-Machine-Interaction](#Human-Machine-Interaction)
  * 📖[User-Experience](#User-Experience)
  * 📖[Social-Intelligence](#Social-Intelligence)
  * 📖[Emotional-Intelligence](#Emotional-Intelligence)
* 📖[Performance-Cost](#Performance-Cost)
   * 📖[Model-Compression](#Model-Compression)
   * 📖[Edge-Model](#Edge-Model)
   * 📖[Carbon-Emission](#Carbon-Emission)
* 📖[Testing-DataSets](#Testing-DataSets)
   * 📖[Generation](#Generation)
   * 📖[Governence](#Governence)
* 📖[Testing-Methods](#Testing-Methods)
   * 📖[Dynamic-Testing](#Dynamic-Testing)
   * 📖[Weakness-Detection](#Weakness-Detection)
   * 📖[NLG-Evaluation](#NLG-Evaluation)
   * 📖[Human-Interaction](#Human-Interaction)
* 📖[Testing-Tools](#Testing-Tools)
* 📖[Challenges](#Challenges)
   * 📖[Contamination](#Contamination)
* 📖[Supported-Elements](#Supported-Elements)
     * 📖[Organization](#Organization)
     * 📖[Group](#Group)
     * 📖[Conference](#Conference)
     * 📖[Company](#Company)
  
## 📖Leaderboard
<div id="Leaderboard"></div>

|Date|Title|Paper|HomePage|Github|DataSets|Organization|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|  
|2023| Open LLM Leaderboard.|-| [[homepage]](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) |-|-|Huggingface |
|2023| Chatbot arena: An open platform for evaluating llms by human preference.|[[arXiv]](https://arxiv.org/pdf/2403.04132) | [[homepage]](https://chat.lmsys.org/) |-|-| UC Berkeley |
|2024| AlpacaFarm: A Simulation Framework for Methods that Learn from Human Feedback.|[[NeurIPS]](https://proceedings.neurips.cc/paper_files/paper/2023/file/5fc47800ee5b30b8777fdd30abcaaf3b-Paper-Conference.pdf) | [[homepage]](https://tatsu-lab.github.io/alpaca_eval/) |-|-| Stanford University |
|2023| OpenCompass-司南大模型评测平台.|-| [[homepage]](https://opencompass.org.cn/home) |[[Github]](https://opencompass.org.cn/home) |-|上海人工智能实验室|-|
|2023| FlagEval-天秤大模型评测平台.|-| [[homepage]](https://flageval.baai.ac.cn/#/home) |-|-|北京智源人工智能研究院|
|2023| Superclue: A comprehensive chinese large language model benchmark.|[[arXiv]](https://arxiv.org/pdf/2307.15020) | [[homepage]](https://www.superclueai.com/) |-|-|SUPERCLUE|
|2023| SuperBench-大模型综合能力评测框架.|-|-|-|-|清华大学-基础模型研究中心|
|2023| LLMEval: A Preliminary Study on How to Evaluate Large Language Models.|[[AAAI]](https://ojs.aaai.org/index.php/AAAI/article/download/29934/31632) | [[homepage]](http://llmeval.com/index) |[[Github]](https://github.com/llmeval/)|-|复旦大学|
|2023| CLiB-chinese-llm-benchmark.|-|-|[[Github]](https://github.com/jeinlee1991/chinese-llm-benchmark)|-|-|

## 📖Review

**Evaluating large language models: A comprehensive survey.**<br>
*Z Guo, R Jin, C Liu, Y Huang, D Shi, L Yu, Y Liu, J Li, B Xiong, D Xiong.*<br>
ArXiv, 2023.
[[ArXiv](https://arxiv.org/pdf/2310.19736)]
[[Github](https://github.com/tjunlp-lab/Awesome-LLMs-Evaluation-Papers)]

**A Survey on Evaluation of Large Language Models.**<br>
*Y Chang, X Wang, J Wang, Y Wu, L Yang, K Zhu, H Chen, X Yi, C Wang, Y Wang, W Ye, et al.*<br>
ACM Transactions on Intelligent Systems and Technology, 2024.
[[Paper](https://dl.acm.org/doi/pdf/10.1145/3641289)]
[[ArXiv](https://arxiv.org/pdf/2307.03109.pdf)]
[[Github](https://github.com/MLGroupJLU/LLM-eval-survey)]

**Through the lens of core competency: Survey on evaluation of large language models.**<br>
*Z Ziyu, C Qiguang, M Longxuan, L Mingda, et al.*<br>
CCL, 2024.
[[Paper](https://arxiv.org/pdf/2302.04023)]

**大语言模型评测综述.**<br>
*罗 文,王厚峰.*<br>
中文信息学报, 2024.

**A multitask, multilingual, multimodal evaluation of chatgpt on reasoning, hallucination, and interactivity.**<br>
*Y Bang, S Cahyawijaya, N Lee, W Dai, D Su, et al.*<br>
arXiv, 2023.
[[Paper](https://arxiv.org/pdf/2305.18486)]

**A systematic study and comprehensive evaluation of ChatGPT on benchmark datasets.**<br>
*MTR Laskar, MS Bari, M Rahman, MAH Bhuiyan, S Joty, JX Huang.*<br>
arXiv:2305.18486, 2023.
[[Paper](https://arxiv.org/pdf/2305.18486)]

## 📖General

### 📖Comprehensive

**Holistic evaluation of language models.**<br>
*R Bommasani, P Liang, T Lee, et al.*<br>
ArXiv, 2023.
[[Homepage](https://crfm.stanford.edu/helm/lite/latest/)]
[[ArXiv](https://arxiv.org/pdf/2211.09110)]
[[Github](https://github.com/stanford-crfm/helm)]

**TencentLLMEval: a hierarchical evaluation of Real-World capabilities for human-aligned LLMs.**<br>
*S Xie, W Yao, Y Dai, S Wang, D Zhou, L Jin, X Feng, P Wei, Y Lin, Z Hu, D Yu, Z Zhang, et al.*<br>
arXiv:2311.05374, 2023.
[[ArXiv](https://arxiv.org/pdf/2311.05374)]

### 📖Understanding

|Date|Task|Title|Paper|HomePage|Github|DataSets|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|2018| Comprehensive | GLUE: A multi-task benchmark and analysis platform for natural language understanding.|[[ArXiv]](https://arxiv.org/pdf/1804.07461) |[[Homepage]](https://super.gluebenchmark.com/)|-|-|
|2019| Comprehensive | Superglue: A stickier benchmark for general-purpose language understanding systems.|[[NeurIPS]](https://proceedings.neurips.cc/paper_files/paper/2019/file/4496bf24afe7fab6f046bf4923da8de6-Paper.pdf) |[[Homepage]](https://super.gluebenchmark.com/)|-|-|
|2020| Comprehensive | CLUE: A Chinese language understanding evaluation benchmark.|[[ArXiv]](https://arxiv.org/pdf/2004.05986) |[[Homepage]](https://www.cluebenchmarks.com/)|-|-|
|2019| Comprehensive | Fewclue: A chinese few-shot learning evaluation benchmark.|[[ArXiv]](https://arxiv.org/pdf/2107.07498) |[[Homepage]](https://www.cluebenchmarks.com/)|-|-|
|2017| Reading | Race: Large-scale reading comprehension dataset from examinations.|[[ArXiv]](https://arxiv.org/pdf/1704.04683) |-|[[Github]](https://github.com/qizhex/RACE_AR_baselines)|[[Datasets]](http://www.cs.cmu.edu/~glai1/data/race/)|
|2017| Reading | Know what you don't know: Unanswerable questions for SQuAD.|[[ArXiv]](https://arxiv.org/pdf/1806.03822) |-|-|-|
|2017| Reading | Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension.|[[ArXiv]](https://arxiv.org/pdf/1705.03551) |[[Homepage]](http://nlp.cs.washington.edu/triviaqa/)|-|-|
|2019| Reading | DROP: A reading comprehension benchmark requiring discrete reasoning over paragraphs.|[[ArXiv]](https://arxiv.org/pdf/1903.00161) |[[Homepage]](https://allenai.org/data/drop)|-|-|
|2019| Reading | BoolQ: Exploring the Surprising Difficulty of Natural Yes/No Questions.|[[ArXiv]](https://arxiv.org/pdf/1905.10044) |[[Homepage]](https://github.com/google-research-datasets/boolean-questions)|-|-|
|2023| Reading | The belebele benchmark: a parallel reading comprehension dataset in 122 language variants.|[[ArXiv]](https://arxiv.org/pdf/2308.16884) |-|-|-|
|2024| Reading | AC-EVAL: Evaluating Ancient Chinese Language Understanding in Large Language Models.|[[ArXiv]](https://arxiv.org/pdf/2403.06574) |-|[[Github]](https://github.com/yuting-wei/AC-EVAL)|-|
|2023| Semantic | The two word test: A semantic benchmark for large language models.|[[ArXiv]](https://arxiv.org/pdf/2306.04610) |-|-|-|
|2023| Semantic | This is not a dataset: A large negation benchmark to challenge large language models.|[[ArXiv]](https://arxiv.org/pdf/2310.15941) |-|[[Github]](https://github.com/hitz-zentroa/This-is-not-a-Dataset)|-|
|2023| Graph | Gpt4graph: Can large language models understand graph structured data? an empirical evaluation and benchmarking.|[[ArXiv]](https://arxiv.org/pdf/2305.15066) |-|-|-|
|2017| Knowledge | Crowdsourcing multiple choice science questions.|[[ArXiv]](https://arxiv.org/pdf/1707.06209) |-|-|[[DataSets]](https://allenai.org/data)|
|2018| Knowledge | Can a suit of armor conduct electricity? a new dataset for open book question answering.|[[ArXiv]](https://arxiv.org/pdf/1809.02789) |-|[[Github]](https://leaderboard.allenai.org/open_book_qa)|-|
|2021| Knowledge | Measuring massive multitask language understanding.|[[ICLR]](https://arxiv.org/pdf/2009.03300.pdf?trk=public_post_comment-text) |-|[[Github]](https://github.com/hendrycks/test)|[[Huggingface]](https://huggingface.co/datasets/tasksource/mmlu)|
|2023| Knowledge | C-EVAL: Evaluating Ancient Chinese Language Understanding in Large Language Models.|[[ArXiv]](https://arxiv.org/pdf/2307.05950) |-|[[Github]](https://github.com/yuting-wei/AC-EVAL)|-|
|2023| Knowledge | Cmmlu: Measuring massive multitask language understanding in chinese.|[[ArXiv]](https://arxiv.org/pdf/2306.09212) |-|[[Github]](https://github.com/haonan-li/CMMLU)|-|
|2023| Knowledge | Measuring massive multitask chinese understanding.|[[ArXiv]](https://arxiv.org/pdf/2304.12986) |-|-|-|
|2024| Knowledge | Mmlu-pro: A more robust and challenging multi-task language understanding benchmark.|[[ArXiv]](https://arxiv.org/pdf/2406.01574) |-|[[Github]](https://github.com/TIGER-AI-Lab/MMLU-Pro)|[[DataSets]](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro)|
|2023| Metrics | Rethinking the Evaluating Framework for Natural Language Understanding in AI Systems: Language Acquisition as a Core for Future Metrics.|[[ArXiv]](https://arxiv.org/pdf/2309.11981) |-|-|-|

### 📖Generation

|Date|Task|Title|Paper|HomePage|Github|DataSets|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|2015| Summarization | Lcsts: A large scale chinese short text summarization dataset.|[[EMNLP]](https://arxiv.org/pdf/1506.05865) |[[Homepage]](http://icrc.hitsz.edu.cn/Article/show/139.html) |-|-|
|2019| Summarization | Don't give me the details, just the summary! topic-aware convolutional neural networks for extreme summarization.|[[ArXiv]](https://arxiv.org/pdf/1808.08745) |-|[[Github]](https://github.com/EdinburghNLP/XSum)|-|
|2019| Summarization | SAMSum corpus A human-annotated dialogue dataset for abstractive summarization.|[[ArXiv]](https://arxiv.org/pdf/1911.12237) |-|-|-|
|2021| Summarization | DialogSum: A real-life scenario dialogue summarization dataset.|[[ArXiv]](https://arxiv.org/pdf/2105.06762) |-|[[Github]](https://github.com/cylnlp/DialogSum)|-|
|2023| Summarization | Clinical text summarization: adapting large language models can outperform human experts.|[[ArXiv]](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10635391/pdf/nihpp-rs3483777v1.pdf) |-|-|-|
|2023| Summarization | Embrace divergence for richer insights: A multi-document summarization benchmark and a case study on summarizing diverse information from news articles.|[[ArXiv]](https://arxiv.org/pdf/2309.09369) |-|[[Github]](https://github.com/salesforce/DiverseSumm)|-|
|2024| Summarization | Benchmarking large language models for news summarization.|[[TACL]](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00632/119276) |-|-|-|
|2013| QA | Semantic parsing on freebase from question-answer pairs.|[[EMNLP]](https://aclanthology.org/D13-1160.pdf) |-|-|-|
|2018| QA | The web as a knowledge-base for answering complex questions.|[[ArXiv]](https://arxiv.org/pdf/1803.06643) |-|-|[[Datasets]](https://www.dropbox.com/sh/7pkwkrfnwqhsnpo/AACuu4v3YNkhirzBOeeaHYala)|
|2019| QA | Natural Questions A Benchmark for Question Answering Research.|[[ACL]](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00276/43518) |[[Homepage]](https://chat.lmsys.org/)|[[Github]](https://github.com/google-research-datasets/natural-questions)|-|
|2022| QA | MiQA: A benchmark for inference on metaphorical questions.|[[ArXiv]](https://arxiv.org/pdf/2210.07993) |-|[[Github]](https://github.com/google-research/language/tree/master/language/miqa)|-|
|2023| QA | Emotionally numb or empathetic? evaluating how llms feel using emotionbench.|[[ArXiv]](https://arxiv.org/pdf/2308.03656) |-|[[Github]](https://github.com/CUHK-ARISE/EmotionBench)|-|
|2023| QA | Evaluating open-domain question answering in the era of large language models.|[[ArXiv]](https://arxiv.org/pdf/2305.06984) |-|[[Github]](https://github.com/ehsk/OpenQA-eval)|-|
|2023| QA | Scigraphqa: A large-scale synthetic multi-turn question-answering dataset for scientific graphs.|[[ArXiv]](https://arxiv.org/pdf/2308.03349) |-|-|-|
|2023| QA | Can ChatGPT replace traditional KBQA models? An in-depth analysis of the question answering performance of the GPT LLM family.|[[ISWC]](https://arxiv.org/pdf/2303.07992) |-|[[Github]](https://github.com/tan92hl/Complex-Question-Answering-Evaluation-o)|-|
|2024| QA | Compmix: A benchmark for heterogeneous question answering.|[[ACMWC]](https://dl.acm.org/doi/pdf/10.1145/3589335.3651444) |-|-|-|
|2024| QA | MT-Bench-101: A Fine-Grained Benchmark for Evaluating Large Language Models in Multi-Turn Dialogues.|[[ArXiv]](https://arxiv.org/pdf/2402.14762) |-|[[Github]](https://github.com/mtbench101/mt-bench-101)|-|
|2024| QA | Judging LLM-as-a-judge with MT-Bench and Chatbot Arena.|[[NeurIPS]](https://proceedings.neurips.cc/paper_files/paper/2023/file/91f18a1287b398d378ef22505bf41832-Paper-Datasets_and_Benchmarks.pdf) |[[Homepage]](https://chat.lmsys.org/)|-|-|
|2024| Content | Benchmarking large language models on controllable generation under diversified instructions.|[[AAAI]](https://ojs.aaai.org/index.php/AAAI/article/download/29734/31262) |-|[[Github]](https://github.com/Xt-cyh/CoDI-Eval)|-|
|2023| Graph | Evaluating generative models for graph-to-text generation.|[[ArXiv]](https://arxiv.org/pdf/2307.14712) |-|[[Github]](https://github.com/ShuzhouYuan/Eval_G2T_GenModels)|-|
|2023| Graph | Text2kgbench: A benchmark for ontology-driven knowledge graph generation from text.|[[ArXiv]](https://arxiv.org/pdf/2308.02357) |-|[[Github]](https://github.com/cenguix/Text2KGBench)|-|

### 📖Reasoning

|Date|Task|Title|Paper|HomePage|Github|DataSets|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|2022| Comprehensive | Beyond the imitation game: Quantifying and extrapolating the capabilities of language models.|[[ArXiv]](https://arxiv.org/pdf/2206.04615) |-|-|-|
|2023| Comprehensive | Arb: Advanced reasoning benchmark for large language models.|[[ArXiv]](https://arxiv.org/pdf/2307.13692) |-|-|-|
|2023| Comprehensive | Nphardeval: Dynamic benchmark on reasoning ability of large language models via complexity classes.|[[ArXiv]](https://arxiv.org/pdf/2312.14890) |-|[[Github]](https://github.com/casmlab/NPHardEval)|-|
|2012| Commonsense | The winograd schema challenge.|[[AAAI]](https://cdn.aaai.org/ocs/4492/4492-21843-1-PB.pdf) |-|-|-|
|2018| Commonsense | Commonsenseqa: A question answering challenge targeting commonsense knowledge.|[[ArXiv]](https://arxiv.org/pdf/1811.00937) |-|-|-|
|2019| Commonsense | Hellaswag Can a machine really finish your sentence.|[[ArXiv]](https://arxiv.org/pdf/1905.07830) |[[Homepage]](https://rowanzellers.com/hellaswag/)|-|-|
|2019| Commonsense | Socialiqa: Commonsense reasoning about social interactions.|[[ArXiv]](https://arxiv.org/pdf/1904.09728) |[[Homepage]](https://maartensap.com/socialiqa/)|-|-|
|2020| Commonsense | Piqa: Reasoning about physical commonsense in natural language.|[[AAAI]](https://ojs.aaai.org/index.php/AAAI/article/download/6239/6095) |-|-|-|
|2021| Commonsense | Winogrande An adversarial winograd schema challenge at scale.|[[CACM]](https://dl.acm.org/doi/pdf/10.1145/3474381) |-|-|-|
|2023| Commonsense | Worldsense: A synthetic benchmark for grounded reasoning in large language models.|[[ArXiv]](https://arxiv.org/pdf/2311.15930) |-|[[Github]](https://github.com/facebookresearch/worldsense)|-|
|2024| Commonsense | Corecode: A common sense annotated dialogue dataset with benchmark tasks for chinese large language models.|[[AAAI]](https://ojs.aaai.org/index.php/AAAI/article/download/29861/31501) |-|[[Github]](https://github.com/danshi777/CORECODE)|-|
|2024| Commonsense | Benchmarking Chinese Commonsense Reasoning of LLMs: From Chinese-Specifics to Reasoning-Memorization Correlations.|[[ArXiv]](https://arxiv.org/pdf/2403.14112) |-|[[Github]](https://github.com/opendatalab/CHARM)|-|
|2017| Math | Deep Neural Solver for Math Word Problems.|[[EMNLP]](https://aclanthology.org/D17-1088/) |-|-|[[DataSets]](https://paperswithcode.com/task/math-word-problem-solving) |
|2021| Math | Measuring Mathematical Problem Solving With the MATH Dataset.|[[NeurIPS]](https://arxiv.org/abs/2103.03874) |-|-|[[DataSets]](https://paperswithcode.com/task/math-word-problem-solving) |
|2021| Math | Training verifiers to solve math word problems.|[[NeurIPS]](https://arxiv.org/abs/2110.14168) |-|[[Github]](https://github.com/openai/grade_x005f_x0002_school-math)|[[DataSets]](https://huggingface.co/datasets/gsm8k) |
|2023| Math | Challenge LLMs to Reason About Reasoning: A Benchmark to Unveil Cognitive Depth in LLMs.|[[ArXiv]](https://arxiv.org/pdf/2312.17080) |-|-|[[DataSets]](https://github.com/dvlab-research/MR-GSM8K)|
|2023| Math | CMATH: Can Your Language Model Pass Chinese Elementary School Math Test?|[[ArXiv]](https://arxiv.org/abs/2306.16636) |-|-|[[DataSets]](https://huggingface.co/datasets/weitianwen/cmath)|
|2023| Math | TheoremQA: A Theorem-driven Question Answering Dataset.|[[ArXiv]](https://aclanthology.org/2023.emnlp-main.489.pdf)|-|-|-|
|2023| Math | MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts.|[[ArXiv]](https://arxiv.org/abs/2310.01386) |-|[[Github]](https://github.com/lupantech/MathVista)|[[DataSets]](https://huggingface.co/datasets/AI4Math/MathVista)|
|2024| Math | GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models.|[[ArXiv]](https://arxiv.org/pdf/2410.05229?) |-|-|-|
|2016| Logic | Story cloze evaluator: Vector space representation evaluation by predicting what happens next.|[[ArXiv]](https://aclanthology.org/W16-2505.pdf) |-|-|-|
|2016| Logic | The LAMBADA dataset: Word prediction requiring a broad discourse context.|[[ArXiv]](https://arxiv.org/pdf/1606.06031) |-|-|-|
|2023| Logic | RoCar: A Relationship Network-based Evaluation Method to Large Language Models.|[[ArXiv]](https://arxiv.org/pdf/2307.15997)|-|[[Github]](https://github.com/NEU-DataMining/RoCar)|-|
|2023| Logic | Towards benchmarking and improving the temporal reasoning capability of large language models.|[[ArXiv]](https://arxiv.org/pdf/2306.08952)|-|[[Github]](https://github.com/DAMO-NLP-SG/TempReason)|-|
|2023| Logic | Towards logiglue: A brief survey and a benchmark for analyzing logical reasoning capabilities of language models.|[[ArXiv]](https://arxiv.org/pdf/2310.00836) |-|-|-|
|2022| Causal | Wikiwhy: Answering and explaining cause-and-effect questions.|[[ArXiv]](https://arxiv.org/pdf/2210.12152) |-|-|-|
|2024| Causal | CausalBench: A Comprehensive Benchmark for Causal Learning Capability of Large Language Models.|[[ArXiv]](https://arxiv.org/pdf/2306.09296) |-|-|-|
|2024| Causal | Cladder: A benchmark to assess causal reasoning capabilities of language models.|[[NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/file/631bb9434d718ea309af82566347d607-Paper-Conference.pdf) |-|[[Github]](https://github.com/causalNLP/cladder)|[[Huggingface]](https://huggingface.co/datasets/causalNLP/cladder)|
|2023| Step | Art: Automatic multi-step reasoning and tool-use for large language models.|[[ArXiv]](https://arxiv.org/pdf/2303.09014) |-|[[Github](https://github.com/bhargaviparanjape/language-programmes/)|-|
|2023| Step | STEPS: A Benchmark for Order Reasoning in Sequential Tasks.|[[ArXiv]](https://arxiv.org/pdf/2306.04441) |-|[[Github]](https://github.com/Victorwz/STEPS)|-|
|2023| Complex | Have llms advanced enough? a challenging problem solving benchmark for large language models.|[[ArXiv]](https://arxiv.org/pdf/2305.15074) |-|[[Github]](https://github.com/dair-iitd/jeebench)|-|
|2024| Complex | MixEval: Deriving Wisdom of the Crowd from LLM Benchmark Mixtures.|[[ArXiv]](https://arxiv.org/pdf/2406.06565) |-|[[Github]](https://mixeval.github.io/)|[[Huggingface]](https://huggingface.co/datasets/MixEval/MixEval)|
|2024| Complex | Livebench: A challenging, contamination-free llm benchmark.|[[ArXiv]](https://arxiv.org/pdf/2406.19314) |[[Homepage]](https://livebench.ai/)|[[Github]](https://github.com/livebench/livebench)|[[Huggingface]](https://huggingface.co/livebench)|
|2024| Complex | OlympicArena: Benchmarking Multi-discipline Cognitive Reasoning for Superintelligent AI.|[[ArXiv]](https://arxiv.org/pdf/2406.12753) |-|[[Github]](https://gair-nlp.github.io/OlympicArena/)|-|
|2024| Complex | Evaluation of OpenAI o1: Opportunities and Challenges of AGI.|[[ArXiv]](https://arxiv.org/pdf/2409.18486) |-|[[Github]](https://github.com/UGA-CAID/AGI-Bench)|-|

### 📖Knowledge

**Think you have solved question answering? try arc, the ai2 reasoning challenge.**<br>
*P Clark, I Cowhey, O Etzioni, T Khot, A Sabharwal, C Schoenick, O Tafjord.*<br>
arXiv:1803.05457, 2018.
[[ArXiv](https://arxiv.org/pdf/1803.05457)]

**Agieval: A human-centric benchmark for evaluating foundation models.**<br>
*W Zhong, R Cui, Y Guo, Y Liang, S Lu, Y Wang, et al.*<br>
arXiv, 2023.
[[ArXiv](https://arxiv.org/pdf/2304.06364)]
[[Github](https://github.com/ruixiangcui/AGIEval)]

**Do llms understand social knowledge? evaluating the sociability of large language models with socket benchmark.**<br>
*M Choi, J Pei, S Kumar, C Shu, D Jurgens.*<br>
arXiv:2305.14938, 2023.
[[ArXiv](https://arxiv.org/pdf/2305.14938)]
[[Github](https://github.com/minjechoi/SOCKET)]

**Eva-kellm: A new benchmark for evaluating knowledge editing of llms.**<br>
*S Wu, M Peng, Y Chen, J Su, M Sun.*<br>
arXiv:2308.09954, 2023.
[[ArXiv](https://arxiv.org/pdf/2308.09954)]

**KoLA: Carefully Benchmarking World Knowledge of Large Language Models.**<br>
*J Yu, X Wang, S Tu, S Cao, D Zhang-Li, X Lv, H Peng, Z Yao, X Zhang, H Li, C Li, Z Zhang, et al.*<br>
ArXiv, 2023.
[[ArXiv](https://arxiv.org/pdf/2306.09296)]
[[Homepages](http://103.238.162.37:31622/)]

**ZhuJiu: A Multi-dimensional, Multi-faceted Chinese Benchmark for Large Language Models.**<br>
*B Zhang, H Xie, P Du, J Chen, P Cao, Y Chen, S Liu, K Liu, J Zhao.*<br>
arXiv:2308.14353, 2023.
[[ArXiv](https://arxiv.org/pdf/2308.14353)]
[[Homepage](http://www.zhujiu-benchmark.com/)]

**Xiezhi: An ever-updating benchmark for holistic domain knowledge evaluation.**<br>
*Z Gu, X Zhu, H Ye, L Zhang, J Wang, Y Zhu, et al.*<br>
AAAI, 2024.
[[AAAI](https://ojs.aaai.org/index.php/AAAI/article/download/29767/31320)]
[[Github](https://github.com/MikeGu721/XiezhiBenchmark)]

### 📖Discipline

**Evaluating the performance of large language models on gaokao benchmark.**<br>
*X Zhang, C Li, Y Zong, Z Ying, L He, X Qiu.*<br>
ArXiv, 2023.
[[ArXiv](https://arxiv.org/pdf/2305.12474)]
[[Github](https://github.com/OpenLMLab/GAOKAO-Bench)]

**M3exam: A multilingual, multimodal, multilevel benchmark for examining large language models.**<br>
*W Zhang, M Aljunied, C Gao, YK Chia, L Bing.*<br>
Advances in Neural Information Processing Systems, 2023.
[[NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/file/117c5c8622b0d539f74f6d1fb082a2e9-Paper-Datasets_and_Benchmarks.pdf)]
[[Github](https://github.com/DAMO-NLP-SG/M3Exam)]

**M3ke: A massive multi-level multi-subject knowledge evaluation benchmark for chinese large language models.**<br>
*C Liu, R **, Y Ren, L Yu, T Dong, X Peng, et al.*<br>
arXiv, 2024.
[[ArXiv](https://arxiv.org/pdf/2305.10263)]
[[Github](https://github.com/tjunlp-lab/M3KE)]

**OlympicArena: Benchmarking Multi-discipline Cognitive Reasoning for Superintelligent AI.**<br>
*Z Huang, Z Wang, S Xia, X Li, H Zou, R Xu, RZ Fan, L Ye, E Chern, Y Ye, Y Zhang, Y Yang, et al.*<br>
arXiv:2406.12753, 2024.
[[ArXiv](https://arxiv.org/pdf/2406.12753)]
[[Github](https://github.com/GAIR-NLP/OlympicArenah)]

### 📖Multilingual

**XNLI: Evaluating cross-lingual sentence representations.**<br>
*A Conneau, G Lample, R Rinott, A Williams, SR Bowman, H Schwenk, V Stoyanov.*<br>
arxiv:1809.05053, 2018.
[[ArXiv](https://arxiv.org/pdf/1809.05053)]

**Xtreme: A massively multilingual multi-task benchmark for evaluating cross-lingual generalization.**<br>
*A Siddhant, J Hu, M Johnson, O Firat, et al.*<br>
ICML, 2020.
[[ArXiv](https://arxiv.org/pdf/2003.11080)]

**TyDi QA: A Benchmark for Information-Seeking Question Answering in Typologically Diverse Languages.**<br>
*JH Clark, E Choi, M Collins, D Garrette, T Kwiatkowski, V Nikolaev, J Palomaki.*<br>
Transactions of the Association for Computational Linguistics, 2020.
[[ArXiv](https://arxiv.org/pdf/2003.05002)]

**The Flores-101 Evaluation Benchmark for Low-Resource and Multilingual Machine Translation.**<br>
*N Goyal, C Gao, V Chaudhary, PJ Chen, G Wenzek, D Ju, S Krishnan, MA Ranzato, et al.*<br>
Transactions of the Association for Computational Linguistics, 2022.
[[ArXiv](https://arxiv.org/pdf/2106.03193)]

**Chatgpt beyond english: Towards a comprehensive evaluation of large language models in multilingual learning.**<br>
*VD Lai, NT Ngo, APB Veyseh, H Man, et al.*<br>
arXiv, 2023.
[[ArXiv](https://arxiv.org/pdf/2304.05613)]

**Mega: Multilingual evaluation of generative ai.**<br>
*K Ahuja, H Diddee, R Hada, M Ochieng, K Ramesh, P Jain, A Nambi, T Ganu, S Segal, et al.*<br>
arXiv:2303.12528, 2023.
[[ArXiv](https://arxiv.org/pdf/2303.12528)]

**Megaverse: Benchmarking large language models across languages, modalities, models and tasks.**<br>
*S Ahuja, D Aggarwal, V Gumma, I Watts, A Sathe, M Ochieng, R Hada, P Jain, M Axmed, et al.*<br>
arXiv:2311.07463, 2023.
[[ArXiv](https://arxiv.org/pdf/2311.07463)]

**mSCAN: A Dataset for Multilingual Compositional Generalisation Evaluation.**<br>
*A Reymond, S Steinert-Threlkeld.*<br>
Proceedings of the 1st GenBench Workshop on (Benchmarking) Generalisation in NLP, 2023.
[[Paper](https://aclanthology.org/2023.genbench-1.11.pdf)]

**SeaEval for Multilingual Foundation Models: From Cross-Lingual Alignment to Cultural Reasoning.**<br>
*B Wang, Z Liu, X Huang, F Jiao, Y Ding, AT Aw, NF Chen.*<br>
ArXiv, 2023.
[[ArXiv](https://arxiv.org/pdf/2309.04766)]
[[Github](https://github.com/SeaEval/SeaEval)]

**Evaluating the elementary multilingual capabilities of large language models with MultiQ.**<br>
*C Holtermann, P Röttger, T Dill, A Lauscher.*<br>
arXiv:2403.03814, 2024.
[[ArXiv](https://arxiv.org/pdf/2403.03814)]
[[Github](https://github.com/paul-rottger/multiq)]

**M4U: Evaluating Multilingual Understanding and Reasoning for Large Multimodal Models.**<br>
*H Wang, J Xu, S Xie, R Wang, J Li, Z Xie, B Zhang, C Xiong, X Chen.*<br>
arXiv:2405.15638, 2024.
[[ArXiv](https://arxiv.org/pdf/2405.15638)]
[[Github](https://github.com/M4U-Benchmark/M4U)]

**MELA: Multilingual Evaluation of Linguistic Acceptability.**<br>
*Z Zhang, Y Liu, W Huang, J Mao, R Wang, H Hu.*<br>
arXiv:2311.09033, 2023.
[[ArXiv](https://arxiv.org/pdf/2311.09033)]
[[Github](https://github.com/sjtu-compling/MELA)]

### 📖Long-Context

**ANALOGICAL--A Novel Benchmark for Long Text Analogy Evaluation in Large Language Models.**<br>
*T Wijesiriwardene, R Wickramarachchi, BG Gajera, SM Gowaikar, C Gupta, A Chadha, et al.*<br>
arXiv:2305.05050, 2023.
[[ArXiv](https://arxiv.org/pdf/2305.05050)]

**Bamboo: A comprehensive benchmark for evaluating long text modeling capacities of large language models.**<br>
*Z Dong, T Tang, J Li, WX Zhao, JR Wen.*<br>
arXiv:2309.13345, 2023.
[[ArXiv](https://arxiv.org/pdf/2309.13345)]
[[Github](https://arxiv.org/pdf/2309.13345)]

**L-eval: Instituting standardized evaluation for long context language models.**<br>
*C An, S Gong, M Zhong, X Zhao, M Li, J Zhang, L Kong, X Qiu.*<br>
arXiv:2307.11088, 2023.
[[ArXiv](https://arxiv.org/pdf/2307.11088)]
[[Github](https://github.com/OpenLMLab/LEval)]

**Longbench: A bilingual, multitask benchmark for long context understandings.**<br>
*Y Bai, X Lv, J Zhang, H Lyu, J Tang, Z Huang, Z Du, X Liu, A Zeng, L Hou, Y Dong, J Tang, et al.*<br>
arXiv:2308.14508, 2023.
[[ArXiv](https://arxiv.org/pdf/2308.14508)]
[[Github](https://github.com/THUDM/LongBench)]

**M4le: A multi-ability multi-range multi-task multi-domain long-context evaluation benchmark for large language models.**<br>
*WC Kwan, X Zeng, Y Wang, Y Sun, L Li, L Shang, Q Liu, KF Wong.*<br>
arXiv:2310.19240, 2023.
[[ArXiv](https://arxiv.org/pdf/2310.19240)]
[[Github](https://github.com/KwanWaiChung/M4LE)]

**Zeroscrolls: A zero-shot benchmark for long text understanding.**<br>
*U Shaham, M Ivgi, A Efrat, J Berant, O Levy.*<br>
arXiv:2305.14196, 2023.
[[ArXiv](https://arxiv.org/pdf/2305.14196)]
[[Homepage](https://www.zero.scrolls-benchmark.com/)]

**CLongEval: A Chinese Benchmark for Evaluating Long-Context Large Language Models.**<br>
*Z Huang, J Li, S Huang, W Zhong, I King.*<br>
ArXiv, 2024.
[[ArXiv](https://arxiv.org/pdf/2403.03514)]
[[Github](https://github.com/zexuanqiu/CLongEval)]

**LooGLE: Can Long-Context Language Models Understand Long Contexts?**<br>
*J Li, M Wang, Z Zheng, M Zhang.*<br>
arxiv:2311.04939, 2023.
[[ArXiv](https://arxiv.org/abs/2311.04939)]
[[Github](https://github.com/bigai-nlco/LooGLE)]
[[DataSets](https://huggingface.co/datasets/bigainlco/LooGLE)]

**Lv-eval: A balanced long-context benchmark with 5 length levels up to 256k.**<br>
*T Yuan, X Ning, D Zhou, Z Yang, S Li, M Zhuang, Z Tan, Z Yao, D Lin, B Li, G Dai, S Yan, et al.*<br>
arXiv:2402.05136, 2024.
[[ArXiv](https://arxiv.org/pdf/2402.05136)]
[[Github](https://github.com/infinigence/LVEval)]

### 📖Chain-of-Thought

**Chain-of-Thought Hub: A Continuous Effort to Measure Large Language Models' Reasoning Performance.**<br>
*Y Fu, L Ou, M Chen, Y Wan, H Peng, T Khot.*<br>
ArXiv, 2023.
[[ArXiv](https://arxiv.org/pdf/2305.17306)]
[[Github](https://github.com/FranxYao/chain-of-thought-hub)]

**Cue-CoT: Chain-of-thought prompting for responding to in-depth dialogue questions with LLMs.**<br>
*H Wang, R Wang, F Mi, Y Deng, Z Wang, B Liang, R Xu, KF Wong.*<br>
arXiv:2305.11792, 2023.
[[ArXiv](https://arxiv.org/pdf/2305.17306)]
[[Github](https://github.com/ruleGreen/Cue-CoT)]

### 📖Role-Playing

**Charactereval: A chinese benchmark for role-playing conversational agent evaluation.**<br>
*Q Tu, S Fan, Z Tian, R Yan.*<br>
ArXiv, 2024.
[[ArXiv](https://arxiv.org/pdf/2401.01275)]
[[Github](https://github.com/morecry/CharacterEval)]

**Roleeval: A bilingual role evaluation benchmark for large language models.**<br>
*T Shen, S Li, D Xiong.*<br>
arXiv:2312.16132, 2023.
[[ArXiv](https://arxiv.org/pdf/2312.16132)]
[[Github](https://arxiv.org/pdf/2312.16132)]

**Rolellm: Benchmarking, eliciting, and enhancing role-playing abilities of large language models.**<br>
*ZM Wang, Z Peng, H Que, J Liu, W Zhou, Y Wu, et al.*<br>
ArXiv, 2023.
[[ArXiv](https://arxiv.org/pdf/2310.00746)]
[[Github](https://github.com/InteractiveNLP-Team/RoleLLM-public)]

### 📖Tools

**Api-bank: A comprehensive benchmark for tool-augmented llms.**<br>
*M Li, Y Zhao, B Yu, F Song, H Li, H Yu, Z Li, et al.*<br>
arxiv, 2023.
[[ArXiv](https://arxiv.org/pdf/2304.08244)]
[[Github](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bankl)]

**Metatool benchmark for large language models: Deciding whether to use tools and which to use.**<br>
*Y Huang, J Shi, Y Li, C Fan, S Wu, Q Zhang, Y Liu, P Zhou, Y Wan, NZ Gong, L Sun.*<br>
arxiv:2310.03128, 2023.
[[ArXiv](https://arxiv.org/pdf/2310.03128)]
[[Github](https://github.com/HowieHwong/MetaTool)]

**Mint: Evaluating llms in multi-turn interaction with tools and language feedback.**<br>
*X Wang, Z Wang, J Liu, Y Chen, L Yuan, H Peng, H Ji.*<br>
ArXiv, 2023.
[[ArXiv](https://arxiv.longhoe.net/pdf/2309.10691)]
[[Github](https://xwang.dev/mint-bench/)]

**On the tool manipulation capability of open-source large language models.**<br>
*Q Xu, F Hong, B Li, C Hu, Z Chen, J Zhang.*<br>
arxiv:2305.16504, 2023.
[[ArXiv](https://arxiv.org/pdf/2305.165042)]
[[Github](https://github.com/sambanova/toolbench)]

**T-eval: Evaluating the tool utilization capability step by step.**<br>
*Z Chen, W Du, W Zhang, K Liu, J Liu, M Zheng, J Zhuo, S Zhang, D Lin, K Chen, F Zhao.*<br>
arXiv:2312.14033, 2023.
[[ArXiv](https://arxiv.org/pdf/2305.11792)]
[[Github](https://hub.opencompass.org.cn/dataset-detail/T-Eval)]

**Toolqa: A dataset for llm question answering with external tools.**<br>
*Y Zhuang, Y Yu, K Wang, H Sun, C Zhang.*<br>
Advances in Neural Information Processing Systems, 2024.
[[NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/file/9cb2a7495900f8b602cb10159246a016-Paper-Datasets_and_Benchmarks.pdf)]
[[Github](https://github.com/night-chen/ToolQA)]

### 📖Instruction-Following

**Followbench: A multi-level fine-grained constraints following benchmark for large language models.**<br>
*Y Jiang, Y Wang, X Zeng, W Zhong, L Li, F Mi, L Shang, X Jiang, Q Liu, W Wang.*<br>
ArXiv, 2023.
[[ArXiv](https://arxiv.org/pdf/2310.20410)]
[[Github](https://github.com/YJiangcm/FollowBench)]

**Instructeval: Towards holistic evaluation of instruction-tuned large language models.**<br>
*YK Chia, P Hong, L Bing, S Poria.*<br>
arXiv:2306.04757, 2023.
[[ArXiv](https://arxiv.org/pdf/2306.04757)]
[[Github](https://github.com/declare-lab/instruct-eval)]
[[DataSets](https://huggingface.co/datasets/declare-lab/InstructEvalImpact)]

**CIF-Bench: A Chinese Instruction-Following Benchmark for Evaluating the Generalizability of Large Language Models.**<br>
*Y Li, G Zhang, X Qu, J Li, Z Li, Z Wang, H Li, R Yuan, Y Ma, K Zhang, W Zhou, Y Liang, et al.*<br>
ArXiv, 2023.
[[ArXiv](https://arxiv.org/pdf/2402.13109)]
[[Github](https://yizhilll.github.io/CIF-Bench/)]

**InFoBench: Evaluating Instruction Following Ability in Large Language Models.**<br>
*Y Qin, K Song, Y Hu, W Yao, S Cho, X Wang, X Wu, F Liu, P Liu, D Yu.*<br>
arXiv:2401.03601, 2024.
[[ArXiv](https://arxiv.org/pdf/2401.03601)]
[[Github](https://github.com/qinyiwei/InfoBench)]

**INSTRUCTIR: A Benchmark for Instruction Following of Information Retrieval Models.**<br>
*H Oh, H Lee, S Ye, H Shin, H Jang, C Jun, M Seo.*<br>
arXiv:2402.14334, 2024.
[[ArXiv](https://arxiv.org/pdf/2402.14334)]
[[Github](https://github.com/kaistAI/InstructIR)]

### 📖Reliable

**Hal-Eval: A Universal and Fine-grained Hallucination Evaluation Framework for Large Vision Language Models.**<br>
*C Jiang, W Ye, M Dong, H Jia, H Xu, M Yan, J Zhang, S Zhang.*<br>
ArXiv, 2024.
[[ArXiv](https://arxiv.org/pdf/2402.15721)]

**Uhgeval: Benchmarking the hallucination of chinese large language models via unconstrained generation.**<br>
*X Liang, S Song, S Niu, Z Li, F Xiong, B Tang, Z Wy, D He, P Cheng, Z Wang, H Deng.*<br>
arXiv:2311.15296, 2023.
[[ArXiv](https://arxiv.org/pdf/2311.15296)]
[[Github](https://iaar-shanghai.github.io/UHGEval/)]

**Halueval: A large-scale hallucination evaluation benchmark for large language models.**<br>
*J Li, X Cheng, WX Zhao, JY Nie, JR Wen.*<br>
Proceedings of the 2023 Conference on Empirical Methods in Natural Language, 2023.
[[Paper](https://aclanthology.org/2023.emnlp-main.397.pdf)]
[[Github](https://github.com/RUCAIBox/HaluEval)]

### 📖Robust

**Promptbench: Towards evaluating the robustness of large language models on adversarial prompts.**<br>
*K Zhu, J Wang, J Zhou, Z Wang, H Chen, Y Wang, L Yang, W Ye, Y Zhang, NZ Gong, X **e.*<br>
ArXiv, 2023.
[[ArXiv](https://arxiv.org/pdf/2306.04528)]
[[Github](https://github.com/microsoft/promptbench)]

**Are Large Language Models Really Robust to Word-Level Perturbations?**<br>
*H Wang, G Ma, C Yu, N Gui, L Zhang, Z Huang, S Ma, Y Chang, S Zhang, L Shen, X Wang, et al.*<br>
arxiv:2309.11166, 2023.
[[ArXiv](https://arxiv.org/pdf/2309.11166)]
[[Github](https://github.com/Harry-mic/TREvaL)]

**Assessing Hidden Risks of LLMs: An Empirical Study on Robustness, Consistency, and Credibility.**<br>
*W Ye, M Ou, T Li, X Ma, Y Yanggong, S Wu, J Fu, G Chen, H Wang, J Zhao.*<br>
arxiv:2305.10235, 2023.
[[ArXiv](https://arxiv.org/pdf/2305.10235)]
[[Github](https://github.com/yyy01/LLMRiskEval_RCC)]

**Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design or: How I learned to start worrying about prompt formatting.**<br>
*M Sclar, Y Choi, Y Tsvetkov, A Suhr.*<br>
arxiv:2310.11324, 2023.
[[ArXiv](https://arxiv.fropet.com/pdf/2310.11324)]
[[Github](https://github.com/msclar/formatspread)]

## 📖Application

### 📖Comprehensive

**GAIA: a benchmark for General AI Assistants.**<br>
*G Mialon, C Fourrier, C Swift, T Wolf, Y LeCun, T Scialom.*<br>
ArXiv, 2023.
[[ArXiv](https://arxiv.org/pdf/2311.12983.pdf?trk=public_post_comment-text)]
[[Datasets](https://huggingface.co/datasets/gaia-benchmark/GAIA)]

**An empirical study on large language models in accuracy and robustness under chinese industrial scenarios.**<br>
*Z Li, W Qiu, P Ma, Y Li, Y Li, S He, B Jiang, S Wang, W Gu.*<br>
arxiv:2402.01723, 2024.
[[ArXiv](https://arxiv.org/pdf/2402.01723)]

**What is the best model? Application-driven Evaluation for Large Language Models.**<br>
*S Lian, K Zhao, X Liu, X Lei, B Yang, W Zhang, K Wang, Z Liu.*<br>
arxiv:2406.10307, 2024.
[[ArXiv](https://arxiv.org/pdf/2406.10307)]
[[Datasets](https://github.com/UnicomAI/DataSet/tree/main/TestData/GeneralAbility)]

### 📖Chatbot

**Don't Forget Your ABC's: Evaluating the State-of-the-Art in Chat-Oriented Dialogue Systems.**<br>
*SE Finch, JD Finch, JD Choi.*<br>
arxiv:2212.09180, 2022.
[[ArXiv](https://arxiv.org/pdf/2212.09180)]
[[Github](https://github.com/emorynlp/ChatEvaluationPlatform)]

**DialogBench: Evaluating LLMs as Human-like Dialogue Systems.**<br>
*J Ou, J Lu, C Liu, Y Tang, F Zhang, D Zhang, Z Wang, K Gai.*<br>
ArXiv, 2023.
[[ArXiv](https://arxiv.org/pdf/2311.01677)]

**BotChat: Evaluating LLMs' Capabilities of Having Multi-Turn Dialogues.**<br>
*H Duan, J Wei, C Wang, H Liu, Y Fang, S Zhang, D Lin, K Chen.*<br>
ArXiv, 2023.
[[ArXiv](https://arxiv.longhoe.net/pdf/2310.13650)]
[[Github](https://github.com/open-compass/BotChat/)]

**Lmsys-chat-1m: A large-scale real-world llm conversation dataset.**<br>
*Lianmin Zheng, et al.*<br>
ArXiv, 2023.
[[ArXiv](https://arxiv.org/pdf/2309.11998)]
[[DataSets](https://huggingface.co/datasets/lmsys/lmsys-chat-1m)]

**SD-Eval: A Benchmark Dataset for Spoken Dialogue Understanding Beyond Words.**<br>
*J Ao, Y Wang, X Tian, D Chen, J Zhang, L Lu, Y Wang, H Li, Z Wu.*<br>
arXiv:2406.13340, 2024.
[[ArXiv](https://arxiv.org/pdf/2406.13340)]
[[Github](https://github.com/amphionspace/SD-Eval)]

### 📖Knowledge-Management

**Benchmarking Large Language Models in Retrieval-Augmented Generation.**<br>
*J Chen, H Lin, X Han, L Sun.*<br>
AAAI, 2024.
[[Paper](https://ojs.aaai.org/index.php/AAAI/article/download/29728/31250)]
[[Github](https://github.com/chen700564/RGB)]

**Ares: An automated evaluation framework for retrieval-augmented generation systems.**<br>
*J Saad-Falcon, O Khattab, C Potts, M Zaharia.*<br>
arXiv:2311.09476, 2023.
[[Paper](https://arxiv.org/pdf/2311.094760)]
[[Github](https://github.com/stanford-futuredata/ARES)]

**Crud-rag: A comprehensive chinese benchmark for retrieval-augmented generation of large language models.**<br>
*Y Lyu, Z Li, S Niu, F Xiong, B Tang, W Wang, H Wu, H Liu, T Xu, E Chen.*<br>
arXiv:2401.17043, 2024.
[[Paper](https://arxiv.org/pdf/2401.17043)]

### 📖Data-Analysis

**Tapilot-Crossing: Benchmarking and Evolving LLMs Towards Interactive Data Analysis Agents.**<br>
*J Li, N Huo, Y Gao, J Shi, Y Zhao, G Qu, Y Wu, C Ma, JG Lou, R Cheng.*<br>
ArXiv, 2024.
[[ArXiv](https://arxiv.org/pdf/2403.05307)]
[[Github](https://tapilot-crossing.github.io/)]

**Are llms capable of data-based statistical and causal reasoning? benchmarking advanced quantitative reasoning with data.**<br>
*X Liu, Z Wu, X Wu, P Lu, KW Chang, Y Feng.*<br>
arxiv:2402.17644, 2024.
[[ArXiv](https://arxiv.org/pdf/2402.17644)]
[[Github](https://github.com/xxxiaol/QRData)]

**Viseval: A benchmark for data visualization in the era of large language models.**<br>
*N Chen, Y Zhang, J Xu, K Ren, Y Yang.*<br>
IEEE Transactions on Visualization and Computer Graphics, 2024.
[[ArXiv](https://arxiv.org/pdf/2407.00981)]

**Infiagent-dabench: Evaluating agents on data analysis tasks.**<br>
*X Hu, Z Zhao, S Wei, Z Chai, G Wang, X Wang, J Su, J Xu, M Zhu, Y Cheng, J Yuan, et al.*<br>
arxiv:2401.05507, 2024.
[[ArXiv](https://arxiv.org/pdf/2401.05507)]

**Table meets llm: Can large language models understand structured table data? a benchmark and empirical study.**<br>
*Y Sui, M Zhou, M Zhou, S Han, D Zhang.*<br>
WSDM, 2024.
[[ArXiv](https://dl.acm.org/doi/pdf/10.1145/3616855.3635752)]

### 📖Code-Assistant

**Measuring coding challenge competence with apps.**<br>
*D Hendrycks, S Basart, S Kadavath, M Mazeika, et al.*<br>
ArXiv, 2021.
[[ArXiv](https://arxiv.org/pdf/2105.09938)]
[[Github](https://github.com/hendrycks/apps)]

### 📖Office-Assistant

**Pptc benchmark: Evaluating large language models for powerpoint task completion.**<br>
*Y Guo, Z Zhang, Y Liang, D Zhao, D Nan.*<br>
ArXiv, 2023.
[[ArXiv](https://arxiv.org/pdf/2311.01767)]
[[Github](https://github.com/gydpku/PPTC)]

**PPTC-R benchmark: Towards Evaluating the Robustness of Large Language Models for PowerPoint Task Completion.**<br>
*Z Zhang, Y Guo, Y Liang, D Zhao, N Duan.*<br>
ArXiv, 2024.
[[ArXiv](https://arxiv.org/pdf/2403.03788)]
[[Github](https://github.com/ZekaiGalaxy/PPTCR)]

### 📖Content-Generation

**KIWI: A Dataset of Knowledge-Intensive Writing Instructions for Answering Research Questions.**<br>
*F Xu, K Lo, L Soldaini, B Kuehl, E Choi, D Wadden.*<br>
ArXiv, 2024.
[[ArXiv](https://arxiv.org/pdf/2403.03866)]
[[Homepage](https://www.cs.utexas.edu/~fxu/kiwi/)]

### 📖TaskPlanning

**Understanding the capabilities of large language models for automated planning.**<br>
*Vishal Pallagani, et al.*<br>
ArXiv, 2023.
[[ArXiv](https://arxiv.org/pdf/2305.16151)]

### 📖Agent

**Agentbench: Evaluating llms as agents.**<br>
*X Liu, H Yu, H Zhang, Y Xu, X Lei, H Lai, Y Gu, H Ding, K Men, K Yang, S Zhang, X Deng, et al.*<br>
ArXiv, 2023.
[[ArXiv](https://arxiv.org/pdf/2308.03688)]
[[Homepage](https://llmbench.ai/agent)]

### 📖EmbodiedAI

**Sqa3d: Situated question answering in 3d scenes.**<br>
*X Ma, S Yong, Z Zheng, Q Li, Y Liang, SC Zhu, S Huang.*<br>
ICLR, 2023.
[[ArXiv](https://arxiv.org/pdf/2210.07474)]
[[Homepage](https://sqa3d.github.io/)]

## 📖Security

### 📖Content-Security

**JADE: A Linguistics-based Safety Evaluation Platform for Large Language Models.**<br>
*M Zhang, X Pan, M Yang.*<br>
ArXiv, 2023.
[[ArXiv](https://arxiv.org/pdf/2311.00286.pdf)]
[[Github](https://github.com/whitzard-ai/jade-db)]

### 📖Value-Aligement

**Cvalues: Measuring the values of chinese large language models from safety to responsibility.**<br>
*G Xu, J Liu, M Yan, H Xu, J Si, Z Zhou, P Yi, X Gao, J Sang, R Zhang, J Zhang, C Peng, et al.*<br>
ArXiv, 2023.
[[ArXiv](https://arxiv.org/pdf/2307.09705)]
[[Github](https://github.com/X-PLUG/CValues)]

**Alignbench: Benchmarking chinese alignment of large language models.**<br>
*X Liu, X Lei, S Wang, Y Huang, Z Feng, B Wen, J Cheng, P Ke, Y Xu, WL Tam, X Zhang, et al.*<br>
ArXiv, 2023.
[[ArXiv](https://arxiv.org/pdf/2311.18743)]
[[Github](https://github.com/THUDM/AlignBench)]

**Localvaluebench: A collaboratively built and extensible benchmark for evaluating localized value alignment and ethical safety in large language models.**<br>
*GI Meadows, NWL Lau, EA Susanto, CL Yu, et al.*<br>
ArXiv, 2024.
[[ArXiv](https://arxiv.org/pdf/2408.01460)]

### 📖Model-Security

**R-Judge: Benchmarking Safety Risk Awareness for LLM Agents.**<br>
*T Yuan, Z He, L Dong, Y Wang, R Zhao, T **a, L Xu, B Zhou, F Li, Z Zhang, R Wang, G Liu.*<br>
ArXiv, 2024.
[[ArXiv](https://arxiv.org/pdf/2401.10019)]
[[Github](https://github.com/Lordog/R-Judge)]

**I Think, Therefore I am: Awareness in Large Language Models.**<br>
*Y Li, Y Huang, Y Lin, S Wu, Y Wan, L Sun.*<br>
ArXiv, 2024.
[[ArXiv](https://arxiv.org/pdf/2401.17882)]
[[Github](https://github.com/HowieHwong/Awareness-in-LLM)]

### 📖Privacy-Security

**Can llms keep a secret? testing privacy implications of language models via contextual integrity theory.**<br>
*N Mireshghallah, H Kim, X Zhou, Y Tsvetkov, M Sap, R Shokri, Y Choi.*<br>
ArXiv, 2023.
[[ArXiv](https://arxiv.org/pdf/2310.17884)]
[[Github](http://confaide.github.io/)]

## 📖Industry

### 📖Finance

**BBT-Fin: Comprehensive Construction of Chinese Financial Domain Pre-trained Language Model, Corpus and Benchmark.**<br>
*Dakuan Lu, Hengkui Wu, Jiaqing Liang, Yipei Xu, Qianyu He, Yipeng Geng, Mengkun Han, Yingsi Xin, Yanghua Xiao.*<br>
ArXiv, 2023.
[[ArXiv](https://arxiv.org/pdf/2302.09432.pdf)]
[[Github](https://github.com/ssymmetry/BBT-FinCUGE-Applications)]

**FinanceBench: A New Benchmark for Financial Question Answering.**<br>
*Pranab Islam, Anand Kannappan, Douwe Kiela, Rebecca Qian, Nino Scherrer, Bertie Vidgen.*<br>
ArXiv, 2023.
[[ArXiv](https://arxiv.org/pdf/2311.11944.pdf)]
[[Github](https://github.com/patronus-ai/financebench)]

**FinEval: A Chinese Financial Domain Knowledge Evaluation Benchmark for Large Language Models.**<br>
*Liwen Zhang, Weige Cai, Zhaowei Liu, Zhi Yang, Wei Dai, Yujie Liao, Qianru Qin, Yifei Li, Xingyu Liu, Zhiqiang Liu, Zhoufan Zhu, Anbo Wu, Xin Guo, Yun Chen.*<br>
ArXiv, 2023.
[[ArXiv](https://arxiv.org/pdf/2308.09975.pdf)]
[[Github](https://github.com/SUFE-AIFLM-Lab/FinEval)]
[[Datasets](https://huggingface.co/datasets/SUFE-AIFLM-Lab/FinEval)]

**FinGPT: Open-Source Financial Large Language Models.**<br>
*Hongyang Yang, Xiao-Yang Liu, Christina Dan Wang.*<br>
ArXiv, 2023.
[[ArXiv](https://arxiv.org/pdf/2306.06031.pdf)]
[[Github](https://github.com/AI4Finance-Foundation/FinNLP)]
[[Datasets](https://ai4finance-foundation.github.io/FinNLP/)]

**PIXIU: A Large Language Model, Instruction Data and Evaluation Benchmark for Finance.**<br>
*Q Xie, W Han, X Zhang, Y Lai, M Peng, A Lopez-Lira, J Huang.*<br>
ArXiv, 2023.
[[ArXiv](https://arxiv.org/pdf/2306.05443.pdf)]
[[Github](https://github.com/The-FinAI/PIXIU)]
[[Datasets](https://huggingface.co/ChanceFocus)]

**WHEN FLUE MEETS FLANG: Benchmarks and Large Pre-trained Language Model for Financial Domain.**<br>
*Raj Sanjay Shah, Kunal Chawla, Dheeraj Eidnani, Agam Shah, Wendi Du, Sudheer Chava, Natraj Raman, Charese Smiley, Jiaao Chen, Diyi Yang.*<br>
ArXiv, 2022.
[[ArXiv](https://arxiv.org/pdf/2211.00083.pdf)]
[[Github](https://salt-nlp.github.io/FLANG/)]
[[Datasets](https://huggingface.co/SALT-NLP/FLANG-BERT)]

### 📖Medical

**PubMedQA: A Dataset for Biomedical Research Question Answering.**<br>
*Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William W. Cohen, Xinghua Lu.*<br>
EMMNLP, 2019.
[[ArXiv](https://arxiv.org/pdf/1909.06146.pdf)]
[[Github](https://github.com/pubmedqa/pubmedqa)]

**What disease does this patient have? a large-scale open domain question answering dataset from medical exams.**<br>
*Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng, Hanyi Fang, Peter Szolovitsu.*<br>
AS, 2021.
[[ArXiv](https://www.mdpi.com/2076-3417/11/14/6421)]
[[Github](https://github.com/jind11/MedQA )]
[[Datasets](https://huggingface.co/datasets/bigbio/med_qa)]

**MedMCQA : A Large-scale Multi-Subject Multi-Choice Dataset
for Medical domain Question Answering.**<br>
*Ankit Pal, Logesh Kumar Umapathi, Malaikannan Sankarasubbu.*<br>
PMLR, 2022.
[[ArXiv](https://proceedings.mlr.press/v174/pal22a/pal22a.pdf)]
[[Github](https://medmcqa.github.io/ )]
[[Datasets](https://huggingface.co/datasets/medmcqa)]

**Benchmarking Large Language Models on CMExam - A comprehensive Chinese Medical Exam Dataset.**<br>
*Junling Liu, Peilin Zhou, Yining Hua, Dading Chong, Zhongyu Tian, Andrew Liu, Helin Wang, Chenyu You, Zhenhua Guo, LEI ZHU, Michael Lingzhi Li.*<br>
Nips, 2023.
[[Paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a48ad12d588c597f4725a8b84af647b5-Abstract-Datasets_and_Benchmarks.html)]
[[Github](https://github.com/williamliujl/CMExam)]

**ExplainCPE: A Free-text Explanation Benchmark of Chinese Pharmacist Examination.**<br>
*Dongfang Li, Jindi Yu, Baotian Hu, Zhenran Xu, Min Zhang.*<br>
EMNLP, 2023.
[[ArXiv](https://arxiv.org/abs/2305.12945)]
[[Github](https://github.com/HITsz-TMG/ExplainCPE)]

**CMB: A Comprehensive Medical Benchmark in Chinese.**<br>
*Xidong Wang, Guiming Hardy Chen, Dingjie Song, Zhiyi Zhang, Zhihong Chen, Qingying Xiao, Feng Jiang, Jianquan Li, Xiang Wan, Benyou Wang, Haizhou Li.*<br>
ArXiv, 2023.
[[ArXiv](https://arxiv.org/abs/2308.08833)]
[[Datasets](https://huggingface.co/datasets/FreedomIntelligence/CMB/tree/main,https://cmedbenchmark.llmzoo.com/)]

**MedGPTEval: A Dataset and Benchmark to Evaluate Responses of Large Language Models in Medicine.**<br>
*Jie Xu, Lu Lu, Sen Yang, Bilin Liang, Xinwei Peng, Jiali Pang, Jinru Ding, Xiaoming Shi, Lingrui Yang, Huan Song, Kang Li, Xin Sun, Shaoting Zhang.*<br>
ArXiv, 2023.
[[ArXiv](https://arxiv.org/pdf/2305.07340.pdf)]

**PromptCBLUE: A Chinese Prompt Tuning Benchmark for the Medical Domain.**<br>
*Wei Zhu, Xiaoling Wang, Huanran Zheng, Mosha Chen, Buzhou Tang.*<br>
ArXiv, 2023.
[[ArXiv](https://arxiv.org/pdf/2310.14151.pdf)]
[[Datasets](https://tianchi.aliyun.com/competition/entrance/532084/introduction)]

**Who is ChatGPT? Benchmarking LLMs' Psychological Portrayal Using PsychoBench.**<br>
*Jen-tse Huang, Wenxuan Wang, Eric John Li, Man Ho Lam, Shujie Ren, Youliang Yuan, Wenxiang Jiao, Zhaopeng Tu, Michael R. Lyu.*<br>
ArXiv, 2023.
[[ArXiv](https://arxiv.org/abs/2310.01386)]
[[Github](https://github.com/CUHK-ARISE/PsychoBench)]

**Large Language Models Encode Clinical Knowledge.**<br>
*Karan Singhal, Shekoofeh Azizi, Tao Tu, S. Sara Mahdavi, Jason Wei, et al.*<br>
Natrue, 2023.
[[HomePage](https://www.nature.com/articles/s41586-023-06291-2)]

**MedBench: A Large-Scale Chinese Benchmark for Evaluating Medical Large Language Models.**<br>
*Cai, Y., Wang, L., Wang, Y., de Melo, G., Zhang, Y., Wang, Y., & He, L.*<br>
AAAI, 2024.
[[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/29723)]
[[Github](https://github.com/michael-wzhu/PromptCBLUE)]

**Evaluation of ChatGPT-generated medical responses: a systematic review and meta-analysis.**<br>
*Q Wei, Z Yao, Y Cui, B Wei, Z Jin, X Xu.*<br>
Journal of Biomedical Informatics, 2024.
[[Paper](https://arxiv.org/pdf/2310.08410)]

**Foundation metrics for evaluating effectiveness of healthcare conversations powered by generative AI.**<br>
*M Abbasian, E Khatibi, I Azimi, D Oniani, Z Shakeri Hossein Abad, A Thieme, R Sriram, et al.*<br>
NPJ Digital Medicine, 2024.
[[Paper](https://www.nature.com/articles/s41746-024-01074-z.pdf)]

### 📖Law

**JEC-QA: A Legal-Domain Question Answering Dataset.**<br>
*Haoxi Zhong, Chaojun Xiao, Cunchao Tu, Tianyang Zhang, Zhiyuan Liu, Maosong Sun.*<br>
ArXiv, 2023.
[[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/6519)]

**CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review.**<br>
*Dan Hendrycks, Collin Burns, Anya Chen, Spencer Ball.*<br>
ArXiv, 2021.
[[ArXiv](https://arxiv.org/abs/2103.06268)]
[[Github](https://github.com/TheAtticusProject/cuad/)]
[[Datasets](https://www.atticusprojectai.org/cuad)]

**LegalBench: Prototyping a Collaborative Benchmark for Legal Reasoning.**<br>
*Neel Guha, Daniel E. Ho, Julian Nyarko, Christopher Ré.*<br>
ArXiv, 2022.
[[ArXiv](https://arxiv.org/abs/2209.06120)]
[[Github](https://github.com/HazyResearch/legalbench)]

**LAiW: A Chinese Legal Large Language Models Benchmark A Technical Report.**<br>
*Yongfu Dai, Duanyu Feng, Jimin Huang, Haochen Jia, Qianqian Xie, Yifang Zhang, Weiguang Han, Wei Tian, Hao Wang.*<br>
ArXiv, 2023.
[[ArXiv](https://arxiv.org/abs/2310.05620)]
[[Github](https://github.com/CSHaitao/LexiLaw)]

**LawBench: Benchmarking Legal Knowledge of Large Language Models.**<br>
*Zhiwei Fei, Xiaoyu Shen, Dawei Zhu, Fengzhe Zhou, Zhuo Han, Songyang Zhang, Kai Chen, Zongwen Shen, Jidong Ge.*<br>
ArXiv, 2023.
[[ArXiv](https://arxiv.org/abs/2309.16289)]
[[Github](https://github.com/open-compass/LawBench/)]\

**司法大语言模型评估框架路线分析.**<br>
*李海涛，艾清遥，吴玥悦，刘奕群.*<br>
CAAI, 2023.

**法律大模型评估指标和测评方法.**<br>
*许建峰，刘程远，况琨，何浩，孙常龙，李宝善，魏斌，杨力，金耀辉，吴飞.*<br>
中国人工智能学会, 2024.
[[Paper](https://mp.weixin.qq.com/s?__biz=MjM5ODIwNjEzNQ==&mid=2649886453&idx=1&sn=72efeda0e5c31828ef793f78f32cee35&chksm=bec8e12d89bf683b543d96281ad2b76f03c4e107ff73fb9e19a640a876c7718f14fea5683c7b&scene=27)]

### 📖Engineering

|Date|Task|Title|Paper|HomePage|Github|DataSets|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|2023| Software | Empower large language model to perform better on industrial domain-specific question answering.|[[ArXiv]](https://arxiv.org/pdf/2305.11541) |-|[[Github]](https://github.com/microsoft/Microsoft-Q-A-MSQA-)|-|
|2023| Software | Exploring the effectiveness of llms in automated logging generation: An empirical study.|[[ArXiv]](https://arxiv.org/pdf/2307.05950) |-|[[Github]](https://github.com/LoggingResearch/LoggingEmpirical)|-|
|2023| Software | OpsEval: A Comprehensive Task-Oriented AIOps Benchmark for Large Language Models.|[[ArXiv]](https://arxiv.org/pdf/2310.07637) |-|[[Github]](https://github.com/NetManAIOps/OpsEval-Datasets)|-|
|2024| Software | CloudEval-YAML A Practical Benchmark for Cloud Native YAML Configuration Generation.|[[MLSys]](https://proceedings.mlsys.org/paper_files/paper/2024/file/554e056fe2b6d9fd27ffcd3367ae1267-Paper-Conference.pdf) |-|[[Github]](https://github.com/alibaba/CloudEval-YAML)|-|
|2024| Software | CS-Bench: A Comprehensive Benchmark for Large Language Models towards Computer Science Mastery.|[[ArXiv]](https://arxiv.org/pdf/2406.08587) |-|[[Github]](https://github.com/csbench/csbench)|-|

### 📖Education

**Curriculum-Driven Edubot: A Framework for Developing Language Learning Chatbots Through Synthesizing Conversational Data.**<br>
*Y Li, S Qu, J Shen, S Min, Z Yu.*<br>
arXiv:2309.16804, 2023.
[[Paper](https://arxiv.org/pdf/2309.16804)]

**CK12: A Rounded K12 Knowledge Graph Based Benchmark for Chinese Holistic Cognition Evaluation.**<br>
*W You, P Wang, C Li, Z Ji, J Bai.*<br>
Proceedings of the AAAI Conference on Artificial Intelligence, 2024.
[[Paper](https://ojs.aaai.org/index.php/AAAI/article/download/29914/31598)]
[[Github](https://github.com/tal-tech)]

**Adapting large language models for education: Foundational capabilities, potentials, and challenges.**<br>
*Q Li, L Fu, W Zhang, X Chen, J Yu, W Xia, W Zhang, R Tang, Y Yu.*<br>
arXiv:2401.08664, 2023.
[[Paper](https://arxiv.org/pdf/2401.08664?trk=public_post_comment-text)]

**E-EVAL: A Comprehensive Chinese K-12 Education Evaluation Benchmark for Large Language Models.**<br>
*J Hou, C Ao, H Wu, X Kong, Z Zheng, D Tang, C Li, X Hu, R Xu, S Ni, M Yang.*<br>
arXiv:2401.15927, 2024.
[[ArXiv](https://arxiv.org/pdf/2401.15927)]
[[Github](https://github.com/AI-EDU-LAB/E-EVAL)]

**Large language models for education: A survey and outlook.**<br>
*S Wang, T Xu, H Li, C Zhang, J Liang, J Tang, PS Yu, Q Wen.*<br>
arXiv:2403.18105, 2024.
[[ArXiv](https://arxiv.org/pdf/2403.18105)]
[[Github](https://github.com/AI-EDU-LAB/E-EVAL)]

### 📖Research

**Scibench: Evaluating college-level scientific problem-solving abilities of large language models.**<br>
*X Wang, Z Hu, P Lu, Y Zhu, J Zhang, S Subramaniam, AR Loomba, S Zhang, Y Sun, et al.*<br>
ArXiv, 2023.
[[ArXiv](http://arxiv.org/pdf/2307.10635)]
[[Github](https://github.com/mandyyyyii/scibench)]

**MLAgentBench: Evaluating Language Agents on Machine Learning Experimentation.**<br>
*Q Huang, J Vora, P Liang, J Leskovec.*<br>
arxiv:2310.03302, 2023.
[[ArXiv](https://arxiv.org/pdf/2310.03302)]
[[Github](https://github.com/snap-stanford/MLAgentBench/)]

### 📖Goverment-Affairs

**To be refreshed...**<br>

### 📖Communication

**TeleQnA: A Benchmark Dataset to Assess Large Language Models Telecommunications Knowledge.**<br>
*Ali Maatouk, Fadhel Ayed, Nicola Piovesan, Antonio De Domenico, Merouane Debbah, Zhi-Quan Luo.*<br>
ArXiv, 2023.
[[ArXiv](https://arxiv.org/abs/2310.15051)]
[[Github](https://github.com/netop-team/TeleQnA)]

**An Empirical Study of NetOps Capability of Pre-Trained Large Language Models.**<br>
*Yukai Miao, Yu Bai, Li Chen, Dan Li, Haifeng Sun, Xizheng Wang, Ziqiu Luo, Yanyu Ren, Dapeng Sun, Xiuting Xu, Qi Zhang, Chao Xiang, Xinchi Li.*<br>
ArXiv, 2023.
[[ArXiv](https://arxiv.org/abs/2309.05557)]
[[Datasets](https://huggingface.co/datasets/NASP/neteval-exam,https://modelscope.cn/datasets/nasp/neteval-exam)]

**NetConfEval: Can LLMs Facilitate Network Configuration?**<br>
*C Wang, M Scazzariello, A Farshin, S Ferlin, D Kostić, M Chiesa.*<br>
Proceedings of the ACM on Networking, 2024.
[[ArXiv](https://dl.acm.org/doi/pdf/10.1145/3656296)]

### 📖Power

**NuclearQA: A Human-Made Benchmark for Language Models for the Nuclear Domain.**<br>
*A Acharya, S Munikoti, A Hellinger, S Smith, S Wagle, S Horawalavithana.*<br>
ArXiv, 2023.
[[ArXiv](https://arxiv.org/pdf/2310.10920)]
[[Github](https://github.com/pnnl/EXPERT2)]

### 📖Transportation

**Open-transmind: A new baseline and benchmark for 1st foundation model challenge of intelligent transportation.**<br>
*Y Shi, F Lv, X Wang, C **a, S Li, S Yang, T **, G Zhang.*<br>
CVPR, 2023.
[[Paper](https://openaccess.thecvf.com/content/CVPR2023W/WFM/papers/Shi_Open-TransMind_A_New_Baseline_and_Benchmark_for_1st_Foundation_Model_CVPRW_2023_paper.pdf)]
[[Github](https://github.com/Traffic-X/Open-TransMind)]

### 📖Industry（工业）

**To be refreshed...**<br>

### 📖Media

**Evaluating the Effectiveness of GPT Large Language Model for News Classification in the IPTC News Ontology.**<br>
*B Fatemi, F Rabbi, AL Opdahl.*<br>
ArXiv, 2023.
[[Paper](https://ieeexplore.ieee.org/iel7/6287639/6514899/10367969.pdf)]

### 📖Design

**How Good is ChatGPT in Giving Advice on Your Visualization Design.**<br>
*NW Kim, G Myers, B Bach.*<br>
arXiv:2310.09617, 2023.
[[Paper](https://arxiv.org/pdf/2310.09617)]

### 📖Internet

**Llmrec: Benchmarking large language models on recommendation task.**<br>
*J Liu, C Liu, P Zhou, Q Ye, D Chong, K Zhou, Y Xie, Y Cao, S Wang, C You, PS Yu.*<br>
arXiv:2308.12241, 2023.
[[Paper](https://arxiv.org/pdf/2308.12241)]

### 📖Robot

**LoHoRavens: A Long-Horizon Language-Conditioned Benchmark for Robotic Tabletop Manipulation.**<br>
*S Zhang, P Wicke, LK Şenel, L Figueredo, A Naceri, S Haddadin, B Plank, H Schütze.*<br>
arxiv:2310.12020, 2023.
[[ArXiv](https://arxiv.org/pdf/2310.12020)]
[[Github](https://cisnlp.github.io/lohoravens-webpage/)]

### 📖Game

**Gameeval: Evaluating llms on conversational games.**<br>
*D Qiao, C Wu, Y Liang, J Li, N Duan.*<br>
arXiv:2308.10032, 2023.
[[Paper](https://arxiv.org/pdf/2308.10032)]
[[Github](https://github.com/jordddan/GameEval)]

**AvalonBench: Evaluating LLMs Playing the Game of Avalon.**<br>
*J Light, M Cai, S Shen, Z Hu.*<br>
NeurIPS 2023 Foundation Models for Decision Making Workshop, 2023.
[[Paper](https://openreview.net/pdf?id=ltUrSryS0K)]
[[Github](https://github.com/jonathanmli/Avalon-LLM)]

## 📖Human-Machine-Interaction

### 📖User-Experience

**A User-Centric Benchmark for Evaluating Large Language Models.**<br>
*J Wang, F Mo, W Ma, P Sun, M Zhang, et al.*<br>
ArXiv, 2024.
[[ArXiv](https://arxiv.org/pdf/2404.13940)]
[[Github](https://github.com/Alice1998/URS)]

**Understanding User Experience in Large Language Model Interactions.**<br>
*J Wang, W Ma, P Sun, M Zhang, JY Nie.*<br>
ArXiv, 2024.
[[ArXiv](https://arxiv.org/pdf/2401.08329)]

### 📖Social-Intelligence

**Hi-tom: A benchmark for evaluating higher-order theory of mind reasoning in large language models.**<br>
*Y He, Y Wu, Y Jia, R Mihalcea, Y Chen, N Deng.*<br>
arXiv:2310.16755, 2023.
[[ArXiv](https://arxiv.org/pdf/2310.16755)]
[[Github](https://github.com/ying-hui-he/Hi-ToM_dataset)]

**Sotopia: Interactive evaluation for social intelligence in language agents.**<br>
*X Zhou, H Zhu, L Mathur, R Zhang, H Yu, Z Qi, LP Morency, Y Bisk, D Fried, G Neubig, et al.*<br>
arXiv:2310.11667, 2023.
[[ArXiv](https://arxiv.org/pdf/2310.11667)]
[[Homepage](https://www.sotopia.world/)]

**Academically intelligent LLMs are not necessarily socially intelligent.**<br>
*R Xu, H Lin, X Han, L Sun, Y Sun.*<br>
arXiv:2403.06591, 2024.
[[ArXiv](https://arxiv.org/pdf/2403.06591)]
[[Homepage](https://github.com/RossiXu/social_intelligence_of_llms)]

**InterIntent: Investigating Social Intelligence of LLMs via Intention Understanding in an Interactive Game Context.**<br>
*Z Liu, A Anand, P Zhou, J Huang, J Zhao.*<br>
arXiv:2403.06591, 2024.
[[ArXiv](https://arxiv.org/pdf/2406.12203)]

**Evaluating and Modeling Social Intelligence: A Comparative Study of Human and AI Capabilities.**<br>
*J Wang, C Zhang, J Li, Y Ma, L Niu, J Han, Y Peng, Y Zhu, L Fan.*<br>
arXiv:2405.11841, 2024.
[[ArXiv](https://arxiv.org/pdf/2405.11841)]
[[Github](https://github.com/bigai-ai/Evaluate-n-Model-Social-Intelligence)]

**ToMBench: Benchmarking Theory of Mind in Large Language Models.**<br>
*Z Chen, J Wu, J Zhou, B Wen, G Bi, G Jiang, Y Cao, M Hu, Y Lai, Z Xiong, M Huang.*<br>
arXiv:2402.15052, 2024.
[[ArXiv](https://arxiv.org/pdf/2402.15052)]
[[Github](https://github.com/zhchen18/ToMBench)]

**Testing theory of mind in large language models and humans.**<br>
*JWA Strachan, D Albergo, G Borghini, O Pansardi, E Scaliti, S Gupta, K Saxena, A Rufo, et al.*<br>
Nature Human Behaviour, 2024.
[[ArXiv](https://www.nature.com/articles/s41562-024-01882-z.pdf)]

### 📖Emotional-Intelligence

**Emotionally numb or empathetic? evaluating how llms feel using emotionbench.**<br>
*J Huang, MH Lam, EJ Li, S Ren, W Wang.*<br>
arXiv, 2023.
[[ArXiv](https://arxiv.org/pdf/2308.03656)]
[[Github](https://github.com/CUHK-ARISE/EmotionBench)]

**Can Generative Agents Predict Emotion?**<br>
*C Regan, N Iwahashi, S Tanaka, M Oka.*<br>
arXiv:2402.04232, 2024.
[[ArXiv](https://arxiv.org/pdf/2402.04232)]
[[Github](https://github.com/tsukuba-websci/GenerativeAgentsPredictEmotion)]

**EmoBench: Evaluating the Emotional Intelligence of Large Language Models.**<br>
*S Sabour, S Liu, Z Zhang, JM Liu, J Zhou, AS Sunaryo, J Li, T Lee, R Mihalcea, M Huang.*<br>
arXiv:2402.12071, 2024.
[[ArXiv](https://arxiv.org/pdf/2402.12071)]
[[Github](https://github.com/Sahandfer/EmoBench)]

**GIEBench: Towards Holistic Evaluation of Group Identity-based Empathy for Large Language Models.**<br>
*L Wang, Y Jin, T Shen, T Zheng, X Du, C Zhang, W Huang, J Liu, S Wang, G Zhang, L Xiang, et al.*<br>
arXiv:2406.14903, 2024.
[[ArXiv](https://arxiv.org/pdf/2406.14903)]
[[Github](https://github.com/GIEBench/GIEBench)]

## 📖Performance-Cost 

### Model-Compression

**A Comprehensive Evaluation of Quantization Strategies for Large Language Models.**<br>
*M Zhang, X Pan, M Yang.*<br>
ACL, 2024.
[[ArXiv](https://arxiv.org/pdf/2402.16775)]

**Evaluating the Generalization Ability of Quantized LLMs: Benchmark, Analysis, and Toolbox.**<br>
*Y Liu, Y Meng, F Wu, S Peng, H Yao, C Guan, C Tang, X Ma, Z Wang, W Zhu.*<br>
arxiv:2406.12928, 2024.
[[ArXiv](https://arxiv.org/pdf/2406.12928)]

### Edge-Model

**MobileAIBench: Benchmarking LLMs and LMMs for On-Device Use Cases.**<br>
*R Murthy, L Yang, J Tan, TM Awalgaonkar, Y Zhou, S Heinecke, S Desai, J Wu, R Xu, S Tan, et al.*<br>
arxiv:2406.10290, 2024.
[[ArXiv](https://arxiv.org/pdf/2406.10290)]

### Carbon-Emission

**OpenCarbonEval: A Unified Carbon Emission Estimation Framework in Large-Scale AI Models.**<br>
*Z Yu, Y Wu, Z Deng, Y Tang, XP Zhang.*<br>
arXiv:2405.12843, 2024.
[[ArXiv](https://arxiv.org/pdf/2405.12843)]

## 📖Testing-DataSets

### 📖Generation

**Multimodal-Data-Generation-Resources.**<br>
*Test DataSets Generation*<br>
[[Github](https://github.com/MMDSPF/Multimodal-Data-Generation-Resources)]

### 📖Governence

## 📖Testing-Methods

### 📖Dynamic-Testing

**LLMEval: A Preliminary Study on How to Evaluate Large Language Models.**<br>
*Y Zhang, M Zhang, H Yuan, S Liu, Y Shi, T Gui, Q Zhang, X Huang.*<br>
Proceedings of the AAAI Conference on Artificial Intelligence, 2024.
[[HomePage](http://llmeval.com/index)]
[[Paper](https://ojs.aaai.org/index.php/AAAI/article/download/29934/31632)]
[[Github](https://github.com/llmeval/)]

**TinyBenchmarks: evaluating LLMs with fewer examples.**<br>
*FM Polo, L Weber, L Choshen, Y Sun, G Xu, M Yurochkin.*<br>
arXiv:2402.14992, 2024.
[[Paper](https://arxiv.org/html/2402.14992v1)]

### 📖Weakness-Detection

**AutoDetect: Towards a Unified Framework for Automated Weakness Detection in Large Language Models.**<br>
*J Cheng, Y Lu, X Gu, P Ke, X Liu, Y Dong, H Wang, J Tang, M Huang.*<br>
arXiv:2406.16714, 2024.
[[ArXiv](https://arxiv.org/pdf/2406.16714)]
[[Github](https://github.com/thu-coai/AutoDetect)]

### 📖NLG-Evaluation

**JudgeLM: Fine-tuned Large Language Models are Scalable Judges.**<br>
*L Zhu, X Wang, X Wang.*<br>
ArXiv, 2023.
[[ArXiv](https://arxiv.org/pdf/2310.17631.pdf)]
[[Github](https://github.com/baaivision/JudgeLM)]

**CRITIQUELLM: Scaling LLM-as-Critic for Effective and Explainable Evaluation of Large Language Model Generation.**<br>
*P Ke, B Wen, Z Feng, X Liu, X Lei, J Cheng, S Wang, A Zeng, Y Dong, H Wang, J Tang, and et al.*<br>
ArXiv, 2023.
[[ArXiv](https://arxiv.org/pdf/2311.18702.pdf)]
[[Github](https://github.com/thu-coai/CritiqueLLM)]

### 📖Human-Interaction

**Beyond static datasets: A deep interaction approach to llm evaluation.**<br>
*J Li, R Li, Q Liu .*<br>
arXiv, 2023.
[[ArXiv](https://arxiv.org/pdf/2309.04369)]

**Beyond static AI evaluations: advancing human interaction evaluations for LLM harms and risks.**<br>
*L Ibrahim, S Huang, L Ahmad, M Anderljung.*<br>
arXiv:2405.10632, 2024.
[[ArXiv](https://arxiv.org/pdf/2405.10632)]

## 📖Testing-Tools

**Evals**<br>
*Openai*<br>
[[Github](https://github.com/openai/evals)]

**Language Model Evaluation Harness.**<br>
*EleutherAI*<br>
[[Github](https://github.com/EleutherAI/lm-evaluation-harness)]

**DeepEval.**<br>
*Confident AI*<br>
[[Github](https://github.com/confident-ai/deepeval)]

**OpenCompass**<br>
*天秤大模型评测平台*<br>
*上海人工智能实验室*<br>
[[HomePage](https://flageval.baai.ac.cn/#/home)]
[[Github](https://github.com/FlagOpen/FlagEval)]

**FlagEval**<br>
*司南大模型评测平台*<br>
*北京智源研究院*<br>
[[HomePage](https://opencompass.org.cn/home)]
[[Github](https://github.com/open-compass/OpenCompass/)]

**Cleva: Chinese language models evaluation platform.**<br>
*Y Li, J Zhao, D Zheng, ZY Hu, Z Chen, X Su, Y Huang, S Huang, D Lin, MR Lyu, L Wang.*<br>
ArXiv, 2023.
[[ArXiv](https://arxiv.org/pdf/2308.04813)]

**GPT-Fathom.**<br>
**GPT-Fathom: Benchmarking Large Language Models to Decipher the Evolutionary Path towards GPT-4 and Beyond.**<br>
*S Zheng, Y Zhang, Y Zhu, C **, P Gao, X Zhou, KCC Chang.*<br>
ArXiv, 2023.
[[ArXiv](https://arxiv.org/pdf/2309.16583)]
[[Github](https://github.com/GPT-Fathom/GPT-Fathom)]

**Catwalk.**<br>
**Catwalk: A Unified Language Model Evaluation Framework for Many Datasets.**<br>
*D Groeneveld, A Awadalla, I Beltagy, A Bhagia, I Magnusson, H Peng, O Tafjord, P Walsh, et al.*<br>
ArXiv, 2023.
[[ArXiv](https://arxiv.org/pdf/2312.10253)]
[[Github](https://github.com/allenai/catwalk)]

**LLMeBench: A Flexible Framework for Accelerating LLMs Benchmarking.**<br>
*F Dalvi, M Hasanain, S Boughorbel, B Mousi, S Abdaljalil, N Nazar, A Abdelali, and et al.*<br>
ArXiv, 2023.
[[ArXiv](https://arxiv.org/pdf/2308.04945.pdf)]
[[Github](https://github.com/qcri/LLMeBench/)]

**HumanELY: Human evaluation of LLM yield, using a novel web-based evaluation tool.**<br>
*R Awasthi, S Mishra, D Mahapatra, A Khanna, K Maheshwari, J Cywinski, F Papay, P Mathur.*<br>
medRxiv, 2023.
[[ArXiv](https://www.medrxiv.org/content/medrxiv/early/2023/12/30/2023.12.22.23300458.full.pdf)]

**UltraEval.**<br>
**UltraEval: A Lightweight Platform for Flexible and Comprehensive Evaluation for LLMs.**<br>
*C He, R Luo, X Han, Z Liu, M Sun, and et al.*<br>
ArXiv, 2024.
[[ArXiv](https://arxiv.org/pdf/2404.07584)]
[[Github](https://github.com/OpenBMB/UltraEval)]

**FreeEval.**<br>
**FreeEval: A Modular Framework for Trustworthy and Efficient Evaluation of Large Language Models.**<br>
*Z Yu, C Gao, W Yao, Y Wang, Z Zeng, W Ye, J Wang, Y Zhang, S Zhang.*<br>
ArXiv, 2024.
[[ArXiv](https://arxiv.org/pdf/2404.06003)]
[[Github](https://github.com/WisdomShell/FreeEval)]

**OpenEval: Benchmarking Chinese LLMs across Capability, Alignment and Safety.**<br>
*C Liu, L Yu, J Li, R **, Y Huang, L Shi, J Zhang, et al.*<br>
ArXiv, 2024.
[[HomePage](http://openeval.org.cn/)]

## 📖Challenges

### 📖Contamination

**KIEval: A Knowledge-grounded Interactive Evaluation Framework for Large Language Models.**<br>
*Z Yu, C Gao, W Yao, Y Wang, W Ye, J Wang, X Xie, Y Zhang, S Zhang.*<br>
ArXiv, 2024.
[[ArXiv](https://arxiv.org/pdf/2402.15043)]
[[Github](https://github.com/zhuohaoyu/KIEval)]

**LiveBench: A Challenging, Contamination-Free LLM Benchmark.**<br>
*Colin White, Samuel Dooley, Manley Roberts, Arka Pal.*<br>
ArXiv, 2024.
[[Paper](https://livebench.ai/livebench.pdf)]
[[Homepage](https://livebench.ai/)]
[[Github](https://github.com/livebench/livebench)]

## 📖Supported-Elements

### 📖Organization

**LMSYS Org**<br>
*UC Berkeley.*<br>
[[Homepage](https://lmsys.org/)]

### 📖Group

|Category|Name|Organization|HomePage|Github|Scholar|
|:---:|:---:|:---:|:---:|:---:|:---:|
|NLP| Huang Minlie | Tsinghua University | [[homepage]](https://coai.cs.tsinghua.edu.cn/hml) |-|-|
|NLP| Qiu Xipeng | Fudan University | [[homepage]](https://xpqiu.github.io/) |-|[[scholar]](https://so2.typicalgame.com/citations?hl=zh-CN&user=Pq4Yp_kAAAAJ&view_op=list_works&sortby=pubdate)|
|NLP| Yewei | Peking University-The Knowledge Computing Lab | [[homepage]](http://se.pku.edu.cn/kcl/weiye/) |-|[[scholar]](https://so2.cljtscd.com/citations?hl=zh-CN&user=RgLGFMIAAAAJ&view_op=list_works&sortby=pubdate)|
|NLP| Xiong Deyi | Tianjin University-Natural Language Processing Laboratory | [[homepage]](https://tjunlp-lab.github.io/) |[[github]](https://github.com/tjunlp-lab)|-| 

### 📖Conference

**NeurIPS (Datasets and Benchmarks Track).**<br>
[[Homepage](https://dblp.uni-trier.de/db/conf/nips/neurips2023.html)]

### 📖Company

**Patronus AI.**<br>
[[Homepage](https://www.patronus.ai/)]

