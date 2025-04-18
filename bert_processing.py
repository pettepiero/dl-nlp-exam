from transformers import BertTokenizer 
from bert_enn import BertInput

def postprocess_key(key):
    return (
        key.replace("model/featurizer/bert/", "").replace(":0", "").replace("self/", "")
    )

def tokenize_input(
    text: str, tokenizer: BertTokenizer, max_length: int = 128 
) -> BertInput:
    """Tokenizes input text and converts it into BertInput format."""
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="np",  # Convert to NumPy arrays
    )

    return BertInput(
        token_ids=encoding["input_ids"],
        segment_ids=encoding["token_type_ids"],
        input_mask=encoding["attention_mask"],
    )


def preprocess_dataset(dataset, tokenizer, max_length=128):
    def tokenize(example):
        encoding = tokenizer(
            example['sentence'],
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='np'
        )
        return {
            'input_ids': encoding['input_ids'][0],
            'token_type_ids': encoding['token_type_ids'][0],
            'attention_mask': encoding['attention_mask'][0],
            'label': example['label']
        }

    return dataset.map(tokenize)
