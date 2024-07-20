from transformers import BertTokenizer

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize input text
text = "Hello, how are you?"
tokens = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

print("Input tokens:", tokens['input_ids'])
print("Attention mask:", tokens['attention_mask'])
