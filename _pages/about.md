---
permalink: /
title: "About Me"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---


I'm Nianlong Gu, a Machine Learning Scientist at [Universität Zürich](https://www.liri.uzh.ch/). I earned my PhD in Computer Science from [ETH Zürich](https://ethz.ch/), where I developed advanced NLP models for scientific literature retrieval, extractive summarization, and LLM-based citation text generation. My current research focuses on AI-based vocal segmentation and speech recognition, crucial for bioacoustic studies and language evolution research. I also develop web platform to assist researchers in annotating audio files using AI.

### Research Interests

- Vocal Segmentation and Speech Recognition
  - AI-based tools for human and animal voice activity detection
  - Enhancing bioacoustic research and cross-species vocal comparisons
  - Utterance extraction and transcription pipelines for minor languages

- Information Retrieval and Document Summarization
  - Efficient retrieval systems for large-scale scientific databases
  - Summarization models for scientific articles
  - Large language models for scientific writing and discovery
  - Deep learning and reinforcement learning applications

### News
* 07/2024: Our paper "Controllable Citation Sentence Generation with Language Models" has been accepted at The 4th Workshop on Scholarly Document Processing @ ACL 2024.
* 06/2024: Our paper "Sentiment- and Keyword-Controllable Text Generation in German with
Pre-trained Language Models" has been accepted at [SwissText 2024](https://www.swisstext.org/).
* 04/2024: We presented our work "[WhisperSeg: Positive Transfer of the Whisper Speech Transformer to Human and Animal Voice Activity Detection](https://ieeexplore.ieee.org/document/10447620)" at [ICASSP 2024](https://2024.ieeeicassp.org/).
* 03/2024: Our paper "[SciPara: A New Dataset for Investigating Paragraph Discourse Structure in Scientific Papers](https://aclanthology.org/2024.codi-1.2/)" has been accepted at [CODI 2024](https://sites.google.com/view/codi2024/home). 


### Publications


<html lang="en">
<head>
<style>
  .grid-container {
    display: grid;
    grid-template-columns: 50px 300px minmax(300px, auto);
    gap: 20px;
    align-items: start;
    margin-bottom: 20px;
  }
  .grid-container-no-figure {
    display: grid;
    grid-template-columns: 50px minmax(300px, auto);
    gap: 20px;
    align-items: start;
    margin-bottom: 20px;
  }
  .grid-item img {
    max-width: 100%;
    height: auto;
  }
  .github-logo {
    width: 20px;
    height: 20px;
    vertical-align: middle;
    margin-right: 5px;
  }
  .publication-title {
    font-weight: bold;
  }
  .authors {
    font-size: smaller;
  }
  .conference-details {
    font-size: smaller;
    font-weight: bold;
  }
  .grid-item p {
    margin: 0;
  }
  .align-top {
    align-self: start;
  }
  /* Media Queries for responsive design */
  @media (max-width: 768px) {
    .grid-container {
      grid-template-columns: 1fr;
      gap: 10px;
    }
    .grid-container .grid-item:nth-child(2) {
      order: 2; /* Image goes after text */
    }
    .grid-container .grid-item:nth-child(3) {
      order: 1; /* Text goes first */
    }
    .grid-item img {
      max-width: 350px; /* Limit max width */
      height: auto;
      margin: 0 auto; /* Center image */
    }
  }
</style>
</head>
<body>


<div class="grid-container">
  <div class="grid-item align-top">
    <p><span class="conference-details">ICASSP 2024</span></p>
  </div>
  <div class="grid-item align-top">
    <img src="/images/publications/2024_whisperseg/teaser.png" alt="2024 whisperseg">
  </div>
  <div class="grid-item align-top">
    <p>
      <span class="publication-title"> <a href="https://doi.org/10.1109/ICASSP48485.2024.10447620" target="_blank">Positive Transfer of the Whisper Speech Transformer to Human and Animal Voice Activity Detection</a></span><br>
      <span class="authors">Nianlong Gu, Kanghwi Lee, Maris Basha, Sumit Kumar Ram, Guanghao You, and Richard H. R. Hahnloser</span><br>
      <span class="conference-details">ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (2024).</span>
    </p>
    <p>
     <img src="/images/github-mark.svg" alt="GitHub Logo" class="github-logo">
      <a href="https://github.com/nianlonggu/WhisperSeg" target="_blank">Code</a>
    </p>
  </div>
</div>

<div class="grid-container">
  <div class="grid-item align-top">
    <p>Under Review</p>
  </div>
  <div class="grid-item align-top">
    <img src="/images/publications/2024_LLM_surpass/teaser.png" alt="2024_LLM_surpass">
  </div>
  <div class="grid-item align-top">
    <p>
      <span class="publication-title"> <a href="https://doi.org/10.48550/arXiv.2403.03230" target="_blank">Large language models surpass human experts in predicting neuroscience results</a></span><br>
      <span class="authors">Xiaoliang Luo, Akilles Rechardt, Guangzhi Sun, ..., Nianlong Gu, ... et al.</span><br>
      <span class="conference-details">arXiv preprint (2024).</span>
    </p>
  </div>
</div>

<div class="grid-container">
  <div class="grid-item align-top">
    <p><span class="conference-details">EACL 2024</span></p>
  </div>
  <div class="grid-item align-top">
    <img src="/images/publications/2024_argument_aligner/teaser.png" alt="2024_argument_aligner">
  </div>
  <div class="grid-item align-top">
    <p>
      <span class="publication-title"> <a href="https://aclanthology.org/2024.eacl-short.14" target="_blank">Evaluating Unsupervised Argument Aligners via Generation of Conclusions of Structured Scientific Abstracts</a></span><br>
      <span class="authors">Yingqiang Gao, Nianlong Gu, Jessica Lam, James Henderson, and Richard Hahnloser</span><br>
      <span class="conference-details">Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics. (2024).</span>
    </p>
  </div>
</div>

<div class="grid-container">
  <div class="grid-item align-top">
    <p><span class="conference-details">SDProc 2024</span></p>
  </div>
  <div class="grid-item align-top">
    <img src="/images/publications/2024_cit_gen/teaser.png" alt="2024_cit_gen">
  </div>
  <div class="grid-item align-top">
    <p>
      <span class="publication-title"> <a href="https://doi.org/10.48550/arXiv.2211.07066" target="_blank">Controllable Citation Sentence Generation with Language Models</a></span><br>
      <span class="authors">Nianlong Gu, and Richard HR Hahnloser</span><br>
      <span class="conference-details">Proceedings of the 4th Workshop on Scholarly Document Processing at ACL 2024. (2024).</span>
    </p>
    <p>
     <img src="/images/github-mark.svg" alt="GitHub Logo" class="github-logo">
      <a href="https://github.com/nianlonggu/LMCiteGen" target="_blank">Code</a>
    </p>
  </div>
</div>

<div class="grid-container">
  <div class="grid-item align-top">
    <p><span class="conference-details">SwissText 2024</span></p>
  </div>
  <div class="grid-item align-top">
    <img src="/images/publications/2024_sent_keyword_control/teaser.png" alt="2024_sent_keyword_control">
  </div>
  <div class="grid-item align-top">
    <p>
      <span class="publication-title"> Sentiment- and Keyword-Controllable Text Generation in German with Pre-trained Language Models</span><br>
      <span class="authors">Paulina Aleksandra Zal, Nianlong Gu, and Guang Lu</span><br>
      <span class="conference-details">SwissText 2024. (2024).</span>
    </p>
    <p>
     <img src="/images/github-mark.svg" alt="GitHub Logo" class="github-logo">
      <a href="https://github.com/polie94/SwissText2024" target="_blank">Code</a>
    </p>
  </div>
</div>

<!-- <div class="grid-container">
  <div class="grid-item align-top">
    <p>Under Review</p>
  </div>
  <div class="grid-item align-top">

  </div>
  <div class="grid-item align-top">
    <p>
      <span class="publication-title"> Lexical, Syntactical, and Superficial Semantic Biases Evaluation in Datasets</span><br>
      <span class="authors">Denis Sutter, Nianlong Gu, and Elliott Ash</span><br>
      <span class="conference-details">(Under review) (2024).</span>
    </p>
  </div>
</div> -->

<div class="grid-container">
  <div class="grid-item align-top">
    <p><span class="conference-details">CODI 2024</span></p>
  </div>
  <div class="grid-item align-top">
    <img src="/images/publications/2024_scipara/teaser.png" alt="2024_scipara">
  </div>
  <div class="grid-item align-top">
    <p>
      <span class="publication-title"> <a href="https://aclanthology.org/2024.codi-1.2" target="_blank">SciPara: A New Dataset for Investigating Paragraph Discourse Structure in Scientific Papers</a></span><br>
      <span class="authors">Anna Kiepura, Yingqiang Gao, Jessica Lam, Nianlong Gu, and Richard H.R. Hahnloser</span><br>
      <span class="conference-details">Proceedings of the 5th Workshop on Computational Approaches to Discourse (CODI 2024). (2024).</span>
    </p>
    <p>
     <img src="/images/github-mark.svg" alt="GitHub Logo" class="github-logo">
      <a href="https://github.com/annamkiepura/SciPara" target="_blank">Code</a>
    </p>
  </div>
</div>

<div class="grid-container">
  <div class="grid-item align-top">
    <p><span class="conference-details">ACL Demo 2024</span></p>
  </div>
  <div class="grid-item align-top">
    <img src="/images/publications/2023_scilit/teaser.png" alt="2023_scilit">
  </div>
  <div class="grid-item align-top">
    <p>
      <span class="publication-title"> <a href="https://doi.org/10.18653/v1/2023.acl-demo.22" target="_blank">SciLit: A Platform for Joint Scientific Literature Discovery, Summarization and Citation Generation</a></span><br>
      <span class="authors">Nianlong Gu and Richard H.R. Hahnloser</span><br>
      <span class="conference-details">Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations) (2023).</span>
    </p>
    <p>
     <img src="/images/github-mark.svg" alt="GitHub Logo" class="github-logo">
      <a href="https://github.com/nianlonggu/SciLit" target="_blank">Code</a>
      &nbsp;&nbsp;
     <img src="/images/app-icon.svg" alt="App" class="github-logo">
      <a href="https://scilit.vercel.app/" target="_blank">Online Demo</a>
    </p>
  </div>
</div>

<div class="grid-container">
  <div class="grid-item align-top">
    <p><span class="conference-details">DocIU 2023</span></p>
  </div>
  <div class="grid-item align-top">
    <img src="/images/publications/2023_memsum_dqa/teaser.png" alt="2023_memsum_dqa">
  </div>
  <div class="grid-item align-top">
    <p>
      <span class="publication-title"> <a href="https://doi.org/10.48550/arXiv.2310.06436" target="_blank">MemSum-DQA: Adapting An Efficient Long Document Extractive Summarizer for Document Question Answering</a></span><br>
      <span class="authors">Nianlong Gu, Yingqiang Gao, and Richard HR Hahnloser</span><br>
      <span class="conference-details">CIKM DocIU Workshop. (2023).</span>
    </p>
    <p>
     <img src="/images/github-mark.svg" alt="GitHub Logo" class="github-logo">
      <a href="https://github.com/nianlonggu/MemSum-DQA" target="_blank">Code</a>
      &nbsp;&nbsp;
     <img src="/images/cup-17.svg" alt="Competition" class="github-logo">
      <a href="https://www.kaggle.com/competitions/pdfvqa" target="_blank">Ranked 1st in PDFVQA Kaggle Competition</a>
    </p>
  </div>
</div>

<div class="grid-container">
  <div class="grid-item align-top">
    <p><span class="conference-details">EMNLP 2023</span></p>
  </div>
  <div class="grid-item align-top">
    <img src="/images/publications/2023_cas/teaser.png" alt="2023_cas">
  </div>
  <div class="grid-item align-top">
    <p>
      <span class="publication-title"> <a href="https://doi.org/10.18653/v1/2023.emnlp-main.372" target="_blank">GreedyCAS: Unsupervised Scientific Abstract Segmentation with Normalized Mutual Information</a></span><br>
      <span class="authors">Yingqiang Gao, Jessica Lam, Nianlong Gu, and Richard Hahnloser</span><br>
      <span class="conference-details">Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP) (2023).</span>
    </p>
    <p>
     <img src="/images/github-mark.svg" alt="GitHub Logo" class="github-logo">
      <a href="https://github.com/CharizardAcademy/GreedyCAS" target="_blank">Code</a>
    </p>
  </div>
</div>


<div class="grid-container-no-figure" >
  <div class="grid-item align-top">
    <p><span class="conference-details">LIRAI 2023</span></p>
  </div>
  <!-- <div class="grid-item align-top">
  </div> -->
  <div class="grid-item align-top">
    <p>
      <span class="publication-title"> <a href="https://doi.org/10.48550/arXiv.2305.08428" target="_blank">Legal extractive summarization of US court opinions</a></span><br>
      <span class="authors">Emmanuel Bauer, Dominik Stammbach, Nianlong Gu, and Elliott Ash</span><br>
      <span class="conference-details">Proceedings of the 1st Legal Information Retrieval meets Artificial Intelligence Workshop (LIRAI) (2023).</span>
    </p>
    <p>
     <img src="/images/github-mark.svg" alt="GitHub Logo" class="github-logo">
      <a href="https://github.com/bauerem/legal_memsum" target="_blank">Code</a>
    </p>
  </div>
</div>

<div class="grid-container-no-figure" >
  <div class="grid-item align-top">
    <p>Under Review</p>
  </div>
  <!-- <div class="grid-item align-top">
  </div> -->
  <div class="grid-item align-top">
    <p>
      <span class="publication-title"> <a href="https://doi.org/10.1101/2023.11.23.568086" target="_blank">Primate origins of human event cognition</a></span><br>
      <span class="authors">Vanessa AD Wilson, Sebastian Sauppe, Sarah Brocard, Erik Jacob Ringen, Moritz M Daum, Stephanie Wermelinger, Nianlong Gu, Caroline Andrews, Arrate Isasi-Isasmendi, Balthasar Bickel, Klaus Zuberbuehler</span><br>
      <span class="conference-details">arXiv preprint (2023).</span>
    </p>
  </div>
</div>

<div class="grid-container">
  <div class="grid-item align-top">
    <p><span class="conference-details">ACL 2022</span></p>
  </div>
  <div class="grid-item align-top">
    <img src="/images/publications/2022_memsum/teaser.png" alt="2022_memsum">
  </div>
  <div class="grid-item align-top">
    <p>
      <span class="publication-title"> <a href="https://doi.org/10.18653/v1/2022.acl-long.450" target="_blank">MemSum: Extractive Summarization of Long Documents Using Multi-Step Episodic Markov Decision Processes</a></span><br>
      <span class="authors">Nianlong Gu, Elliott Ash, and Richard Hahnloser</span><br>
      <span class="conference-details">Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics, ACL 2022. (2022).</span>
    </p>
    <p>
     <img src="/images/github-mark.svg" alt="GitHub Logo" class="github-logo">
      <a href="https://github.com/nianlonggu/MemSum" target="_blank">Code</a>
      &nbsp;&nbsp;
     <img src="/images/app-icon.svg" alt="App" class="github-logo">
      <a href="https://huggingface.co/spaces/nianlong/memsum-arxiv-summarizer" target="_blank">Online Demo</a>
    </p>
  </div>
</div>


<div class="grid-container">
  <div class="grid-item align-top">
    <p><span class="conference-details">ECIR 2022</span></p>
  </div>
  <div class="grid-item align-top">
    <img src="/images/publications/2022_hatten/teaser.png" alt="2022_hatten">
  </div>
  <div class="grid-item align-top">
    <p>
      <span class="publication-title"> <a href="https://link.springer.com/chapter/10.1007/978-3-030-99736-6_19" target="_blank">Local Citation Recommendation with Hierarchical-Attention Text Encoder and SciBERT-Based Reranking</a></span><br>
      <span class="authors">Nianlong Gu, Yingqiang Gao, and Richard Hahnloser</span><br>
      <span class="conference-details">Proceedings of ECIR 2022. (2022).</span>
    </p>
    <p>
     <img src="/images/github-mark.svg" alt="GitHub Logo" class="github-logo">
      <a href="https://github.com/nianlonggu/Local-Citation-Recommendation" target="_blank">Code</a>
    </p>
  </div>
</div>

<div class="grid-container">
  <div class="grid-item align-top">
    <p><span class="conference-details">Argument Mining Workshop 2022</span></p>
  </div>
  <div class="grid-item align-top">
    <img src="/images/publications/2022_discourse/teaser.png" alt="2022_discourse">
  </div>
  <div class="grid-item align-top">
    <p>
      <span class="publication-title"> <a href="https://aclanthology.org/2022.argmining-1.3" target="_blank">Do Discourse Indicators Reflect the Main Arguments in Scientific Papers?</a></span><br>
      <span class="authors">Yingqiang Gao, Nianlong Gu, Jessica Lam, and Richard H R Hahnloser</span><br>
      <span class="conference-details">Proceedings of the 9th Workshop on Argument Mining. (2022).</span>
    </p>
    <p>
     <img src="/images/github-mark.svg" alt="GitHub Logo" class="github-logo">
      <a href="https://github.com/CharizardAcademy/discourse-indicator" target="_blank">Code</a>
    </p>
  </div>
</div>

<div class="grid-container">
  <div class="grid-item align-top">
    <p><span class="conference-details">ACL Demo 2020</span></p>
  </div>
  <div class="grid-item align-top">
    <img src="/images/publications/2020_demo/teaser.png" alt="2020_demo">
  </div>
  <div class="grid-item align-top">
    <p>
      <span class="publication-title"> <a href="https://doi.org/10.18653/v1/2020.acl-demos.36" target="_blank">Embedding-based Scientific Literature Discovery in a Text Editor Application</a></span><br>
      <span class="authors">Onur Gökçe, Jonathan Prada, Nikola I. Nikolov, Nianlong Gu, and Richard H.R. Hahnloser</span><br>
      <span class="conference-details">Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics: System Demonstrations. (2020).</span>
    </p>
  </div>
</div>


<div class="grid-container">
  <div class="grid-item align-top">
    <p><span class="conference-details">WACV 2020</span></p>
  </div>
  <div class="grid-item align-top">
    <img src="/images/publications/2020_vae/teaser.png" alt="2020_vae">
  </div>
  <div class="grid-item align-top">
    <p>
      <span class="publication-title"> <a href="https://doi.org/10.1109/WACV45572.2020.9093319" target="_blank">Reverse Variational Autoencoder for Visual Attribute Manipulation and Anomaly Detection</a></span><br>
      <span class="authors">Lydia Gauerhof and Nianlong Gu</span><br>
      <span class="conference-details">Proceedings of the IEEE Winter Conference on Applications of Computer Vision (WACV). (2020).</span>
    </p>
    <p>
     <img src="/images/github-mark.svg" alt="GitHub Logo" class="github-logo">
      <a href="https://github.com/nianlonggu/reverse-variational-autoencoder" target="_blank">Code</a>
    </p>
  </div>
</div>

</body>
</html>

