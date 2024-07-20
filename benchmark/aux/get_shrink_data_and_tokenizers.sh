#!/bin/bash

if [ "x$NLP_BENCH_ROOT" = "x" ]; then
    echo "NLP_BENCH_ROOT is not set. Please set it to the root directory of nlp-benchmark" >&2
    exit 1
fi

oscar_url=https://huggingface.co/bigscience/misc-test-data/resolve/main/stas/oscar-1GB.jsonl.xz
oscar_path=$NLP_BENCH_ROOT/benchmark/aux/oscar-1GB.jsonl.xz
output_path=$NLP_BENCH_ROOT/benchmark/aux/oscar-10MB.jsonl.xz


# Download test OSCAR subset if not already there.
if ! [ -f "$oscar_path" ]; then
    wget -O "$oscar_path" "$oscar_url"
fi
# The test OSCAR subset contains 79000 samples in total.
xz -dc "$oscar_path" | head -n 790 | xz > "$output_path"


TOKENIZER_VOCAB_URL=https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
TOKENIZER_MERGE_URL=https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt

VOCAB_FILE="$NLP_BENCH_ROOT/benchmark/aux/tokenizers/gpt2-vocab.json"
MERGE_FILE="$NLP_BENCH_ROOT/benchmark/aux/tokenizers/gpt2-merges.txt"


# Download tokenizer data.
if ! [ -f "$VOCAB_FILE" ]; then
    wget -O "$VOCAB_FILE" "$TOKENIZER_VOCAB_URL"
fi
if ! [ -f "$MERGE_FILE" ]; then
    wget -O "$MERGE_FILE" "$TOKENIZER_MERGE_URL"
fi

