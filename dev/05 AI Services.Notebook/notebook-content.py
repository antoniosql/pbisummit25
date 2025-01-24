# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {}
# META }

# CELL ********************

import synapse.ml.core
from synapse.ml.cognitive.language import AnalyzeText
from pyspark.sql.functions import col

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Sentiment Analysis
df = spark.createDataFrame([
    ("Great atmosphere. Close to plenty of restaurants, hotels, and transit! Staff are friendly and helpful.",),
    ("What a sad story!",)
], ["text"])

model = (AnalyzeText()
        .setTextCol("text")
        .setKind("SentimentAnalysis")
        .setOutputCol("response"))

result = model.transform(df)\
        .withColumn("documents", col("response.documents"))\
        .withColumn("sentiment", col("documents.sentiment"))

display(result.select("text", "sentiment"))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

#Language Detector
df = spark.createDataFrame([
    (["Hello world"],),
    (["Bonjour tout le monde", "Hola mundo", "Tumhara naam kya hai?"],),
    (["你好"],),
    (["日本国（にほんこく、にっぽんこく、英"],)
], ["text"])

model = (AnalyzeText()
        .setTextCol("text")
        .setKind("LanguageDetection")
        .setOutputCol("response"))

result = model.transform(df)\
        .withColumn("documents", col("response.documents"))\
        .withColumn("detectedLanguage", col("documents.detectedLanguage.name"))

display(result.select("text", "detectedLanguage"))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

#Key Phrase Detector
df = spark.createDataFrame([
    ("en", "Microsoft was founded by Bill Gates and Paul Allen."),
    ("en", "Text Analytics is one of the Azure Cognitive Services."),
    ("en", "My cat might need to see a veterinarian.")
], ["language", "text"])

model = (AnalyzeText()
        .setTextCol("text")
        .setKind("KeyPhraseExtraction")
        .setOutputCol("response"))

result = model.transform(df)\
        .withColumn("documents", col("response.documents"))\
        .withColumn("keyPhrases", col("documents.keyPhrases"))

display(result.select("text", "keyPhrases"))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

#Name Entity Recognition

df = spark.createDataFrame([
    ("en", "Microsoft was founded by Bill Gates and Paul Allen."),
    ("en", "Pike place market is my favorite Seattle attraction.")
], ["language", "text"])

model = (AnalyzeText()
        .setTextCol("text")
        .setKind("EntityRecognition")
        .setOutputCol("response"))

result = model.transform(df)\
        .withColumn("documents", col("response.documents"))\
        .withColumn("entityNames", col("documents.entities.text"))

display(result.select("text", "entityNames"))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Translations
import synapse.ml.core
from synapse.ml.cognitive.translate import *
from pyspark.sql.functions import col, flatten

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

df = spark.createDataFrame([
  (["Hello, what is your name?", "Bye"],)
], ["text",])

translate = (Translate()
    .setTextCol("text")
    .setToLanguage(["zh-Hans", "fr"])
    .setOutputCol("translation")
    .setConcurrency(5))

result = translate.transform(df)\
        .withColumn("translation", flatten(col("translation.translations")))\
        .withColumn("translation", col("translation.text"))

display(result.select("text", "translation"))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

#OpenAI
import synapse.ml.core
from synapse.ml.services.openai import *

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from synapse.ml.services.openai import OpenAIChatCompletion
from pyspark.sql import Row
from pyspark.sql.types import *


def make_message(role, content):
    return Row(role=role, content=content, name=role)


chat_df = spark.createDataFrame(
    [
        (
            [
                make_message(
                    "system", "You are an AI chatbot with red as your favorite color"
                ),
                make_message("user", "Whats your favorite color"),
            ],
        ),
        (
            [
                make_message("system", "You are very excited"),
                make_message("user", "How are you today"),
            ],
        ),
    ]
).toDF("messages")


chat_completion = (
    OpenAIChatCompletion()
    .setDeploymentName("gpt-35-turbo-0125") # deploymentName could be one of {gpt-35-turbo-0125 or gpt-4-32k}
    .setMessagesCol("messages")
    .setErrorCol("error")
    .setOutputCol("chat_completions")
)

display(
    chat_completion.transform(chat_df).select(
        "messages", "chat_completions.choices.message.content"
    )
)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

df = spark.createDataFrame(
    [
        ("Once upon a time",),
        ("Best programming language award goes to",),
        ("SynapseML is ",)
    ]
).toDF("text")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

embedding = (
    OpenAIEmbedding()
    .setDeploymentName("text-embedding-ada-002") # set deployment_name as text-embedding-ada-002
    .setTextCol("text")
    .setOutputCol("out")
)
display(embedding.transform(df))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
