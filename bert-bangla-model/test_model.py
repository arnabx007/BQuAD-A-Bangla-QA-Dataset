import transformers
import numpy as np

tokenizer = transformers.BertTokenizer.from_pretrained('.')
sentence = 'উপজেলা পর্যায়ে বিভিন্ন দপ্তরের কাগজপত্র ও নথি অনুমোদনের জন্য উপজেলা নির্বাহী কর্মকর্তার\
     (ইউএনও) মাধ্যমে উপজেলা চেয়ারম্যানের কাছে উপস্থাপন করতে হবে'

print(tokenizer.tokenize(sentence))
encoded_input = tokenizer(sentence, return_tensors='pt')
print(encoded_input)
print([tokenizer.decode(x) for x in encoded_input['input_ids']])

model = transformers.BertModel.from_pretrained('.')
output = model(encoded_input['input_ids'])

print(output[0].shape)
print(output[1].shape)

if output:
     print('Model loaded successfully')