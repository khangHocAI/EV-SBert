from csv import reader
from sentence_transformers import SentenceTransformer, LoggingHandler, models, evaluation, losses



source_lang = "vi"
dest_lang = "en"

filepath = 'test_1000c.csv'
sts_data = {'sentences1': [], 'sentences2': [], 'scores': []}


# open file in read mode
with open(filepath, 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    # Iterate over each row in the csv using reader object
    for row in csv_reader:
        # row variable
        out = "" 
        for split in row:
            out += split 
        sent1, sent2, score = out.strip().split("|")
        score = float(score)
        sts_data['sentences1'].append(sent1)
        sts_data['sentences2'].append(sent2)
        sts_data['scores'].append(score)

inference_batch_size = 64 


test_evaluator = evaluation.EmbeddingSimilarityEvaluator(sts_data['sentences1'],sts_data['sentences2'], sts_data['scores'], batch_size=inference_batch_size, show_progress_bar=True)

model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

result = model.evaluate(test_evaluator)
print(result)

        