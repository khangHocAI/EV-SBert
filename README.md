# EV-SBert

Metrics: cosine similarity + spearman rank correlation on STS vietnamese english

Base performance:
+ distiluse-base-multilingual-cased-v2 (50 languages -480mb) => 0.58
+ paraphrase-multilingual-MiniLM-L12-v2 (50 languages - 420mb) => 0.57
+ paraphrase-multilingual-mpnet-base-v2 (50 languages - 1.2gb) => 0.62

Problem:
-> really slow dataloader in library

Plan:
- fix dataloader (tfrecords)
- distil on english model by ted 2020 and phomt
- apply techniques from lexical analysis and NMT
