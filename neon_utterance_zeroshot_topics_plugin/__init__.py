# # NEON AI (TM) SOFTWARE, Software Development Kit & Application Development System
# # All trademark and other rights reserved by their respective owners
# # Copyright 2008-2021 Neongecko.com Inc.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from this
#    software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS  BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS;  OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE,  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import nltk
from flair.data import Sentence
from flair.models import TARSClassifier
from nltk.corpus import wordnet as wn

from neon_transformers import UtteranceTransformer


class TarsTopicExtractor(UtteranceTransformer):
    def __init__(self, name="TarsTopic", priority=99):
        super().__init__(name, priority)
        nltk.download("wordnet")
        self.thresh = self.config.get("thresh", 0.4)
        # Load pre-trained TARS model for English
        self.tars = TARSClassifier.load('tars-base')

    def get_keywords(self, context):
        # NOTE: you need to install one of the
        # neon transformers that provides keywords
        # priority if this plugin defaults to 99
        # to ensure keyword extractors run first

        # keybert
        kws = context.get("keybert_keywords") or []
        # yake
        kws += context.get("yake_keywords") or []
        # rake
        kws += context.get("rake_keywords") or []

        if kws:
            return [k[0] for k in kws]
        return []

    def get_parent_words(self, keywords):
        parents = []
        for kw in keywords:
            sym = wn.synsets(kw)[:2]
            parents += [j.name().split('.')
                        for i in sym for j in i.hypernyms()]
        parents = [i[0] for i in parents if i[1] != 'v']
        parents = [i for i in parents if i not in keywords]
        return list(set(parents))

    def get_labels(self, keywords):
        classes = []
        for k in keywords:
            classes += self.get_parent_words([k.replace(" ", "_")])
            classes += self.get_parent_words(k.split())
        return list(set(classes))

    def transform(self, utterances, context=None):
        context = context or {}
        topics = {}

        keywords = self.get_keywords(context)
        classes = self.get_labels(keywords)

        if classes:
            for utt in utterances:
                sentence = Sentence(utt)
                self.tars.predict_zero_shot(sentence, classes)
                for l in sentence.labels:
                    if l._value not in topics:
                        topics[l._value] = 0
                    if l._score > topics[l._value]:
                        topics[l._value] = l._score

        topics = [(k, v) for k, v in topics.items() if v >= self.thresh]
        topics = sorted(topics, key=lambda k: k[1], reverse=True)
        # return unchanged utterances + data
        return utterances, {"zeroshot_topics": topics}


