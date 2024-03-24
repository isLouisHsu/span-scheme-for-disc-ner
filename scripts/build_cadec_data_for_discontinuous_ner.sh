# Download data from https://data.csiro.au/dap/landingpage?pid=csiro:10948&v=3&d=true
unzip data/raw/cadec/CADEC/CADEC.v2.zip -d data/raw/cadec/CADEC/CADEC.v2/
output_dir=data/raw/cadec/CADEC/CADEC.v2/cadec/processed
log_file=${output_dir}/build_cadec_data_for_discontinuous_ner.log
mkdir ${output_dir}

echo "Extract annotations ..." >> ${log_file}
python data/raw/cadec/extract_annotations.py \
    --input_ann data/raw/cadec/CADEC/CADEC.v2/cadec/original/ \
    --input_text data/raw/cadec/CADEC/CADEC.v2/cadec/text/ \
    --output_filepath ${output_dir}/ann \
    --type_of_interest ADR \
    --log_filepath ${output_dir}/${log_file}

echo "Tokenization ..." >> ${log_file}
python data/raw/cadec/tokenization.py \
    --input_dir data/raw/cadec/CADEC/CADEC.v2/cadec/text/ \
    --output_filepath ${output_dir}/tokens \
    --log_filepath ${log_file}

echo "Convert annotations from character level offsets to token level idx ..." >> ${log_file}
python data/raw/cadec/convert_ann_using_token_idx.py \
    --input_ann ${output_dir}/ann \
    --input_tokens ${output_dir}/tokens \
    --output_ann ${output_dir}/tokens.ann \
    --log_filepath ${log_file}

echo "Create text inline format ..." >> ${log_file}
python data/raw/cadec/convert_text_inline.py \
    --input_ann ${output_dir}/tokens.ann \
    --input_tokens ${output_dir}/tokens \
    --output_filepath ${output_dir}/text-inline \
    --log_filepath ${log_file}

echo "Split the data set into train, dev, test splits ..." >> ${log_file}
python data/raw/cadec/split_train_test.py \
    --input_filepath ${output_dir}/text-inline \
    --output_dir ${output_dir}/split

mkdir -p data/processed/cadec/
cp ${output_dir}/split/* data/processed/cadec/
