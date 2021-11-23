```python
from neon_utterance_KeyBERT_plugin import KeyBERTExtractor
from neon_utterance_zeroshot_topics_plugin import TarsTopicExtractor


kbert = KeyBERTExtractor()  # or RAKE or YAKE or all of them!

tars = TarsTopicExtractor()

utts = ["Dog is man's best friend. It is always loyal."]

_, context = kbert.transform(utts)

_, context = tars.transform(utts, context)
# {'zeroshot_topics': [('canine', 0.9103702902793884), ('domestic_animal', 0.8883659839630127)]} 

```