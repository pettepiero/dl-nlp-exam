import tensorflow as tf
from transformers import TFAutoModel
from transformers import TFAutoModel, AutoTokenizer
from datasets import load_dataset

# setting BATCH_SIZE to 64.
BATCH_SIZE = 64

def tokenize(batch):
    return tokenizer(batch["sentence"], padding=True, truncation=True)

def order(inp):
    '''
    This function will group all the inputs of BERT
    into a single dictionary and then output it with
    labels.
    '''
    data = list(inp.values())
    return {
        'input_ids': data[1],
        'attention_mask': data[2],
        'token_type_ids': data[3]
    }, data[0]

class BERTForClassification(tf.keras.Model):
    
    def __init__(self, bert_model, num_classes):
        super().__init__()
        self.bert = bert_model
        self.fc = tf.keras.layers.Dense(num_classes, activation='softmax')
        
    def call(self, inputs):
        x = self.bert(inputs)[1]
        return self.fc(x)

if __name__ == '__main__':
    model = TFAutoModel.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    inputs = tokenizer(
        ['Hello world', 'Hi how are you'], 
        padding=True, 
        truncation=True,
        return_tensors='tf'
    )
    output = model(inputs)
    print(f"Inputs: {inputs}")
    print(f"Output: {output}")

    sentences = load_dataset("stanfordnlp/sst2")

    sentences_encoded = sentences.map(tokenize, batched=True, batch_size=None)

    # setting 'input_ids', 'attention_mask', 'token_type_ids', and 'label'
    # to the tensorflow format. Now if you access this dataset you will get these
    # columns in `tf.Tensor` format

    sentences_encoded.set_format(
        'tf', 
        columns=['input_ids', 'attention_mask', 'token_type_ids', 'label']
    )

    # converting train split of `sentences_encoded` to tensorflow format
    train_dataset = tf.data.Dataset.from_tensor_slices(sentences_encoded['train'][:])
    # set batch_size and shuffle
    train_dataset = train_dataset.batch(BATCH_SIZE).shuffle(1000)
    # map the `order` function
    train_dataset = train_dataset.map(order, num_parallel_calls=tf.data.AUTOTUNE)

    # ... doing the same for test set ...
    test_dataset = tf.data.Dataset.from_tensor_slices(sentences_encoded['test'][:])
    test_dataset = test_dataset.batch(BATCH_SIZE)
    test_dataset = test_dataset.map(order, num_parallel_calls=tf.data.AUTOTUNE)

    inp, out = next(iter(train_dataset)) # a batch from train_dataset
    print("Input for the model:")
    print(inp, '\n\n', out)

    classifier = BERTForClassification(model, num_classes=2)

    classifier.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    history = classifier.fit(
        train_dataset,
        epochs=3
    )

    test_result = classifier.evaluate(test_dataset)
    model.export("./model_check/")

