set -euo pipefail

root=$(dirname $0)

usage () {
    echo "usage: $0 ckp slang tlang < input > output" >&2
    exit 1
}

[ $# -eq 3 ] || usage

ckp=$1
slang=$2
tlang=$3
input=$4
fairseq_dir=$5

translate () {
    local ckp slang tlang
    ckp=$1
    slang=$2
    tlang=$3
    bash $root/preprocess/normalize_punctuation.sh $slang < $input | \
        python $fairseq_dir/scripts/spm_encode.py  --model $root/preprocess/flores200_sacrebleu_tokenizer_spm.model | \
        fairseq-interactive $root --input - -s $slang -t $tlang \
            --path $ckp --batch-size 256 --max-tokens 4096 --buffer-size 10000 \
            --beam 4 --lenpen 1.0 \
            --fp16 \
            --fixed-dictionary $root/dictionary.txt \
            --task translation_multi_simple_epoch \
            --decoder-langtok --encoder-langtok src \
            --langs $(cat $root/langs.txt) \
            --lang-pairs $slang-$tlang \
            --add-data-source-prefix-tags > $root/${tlang}_translated.hyp

}

translate $ckp $slang $tlang

grep "^H" $root/${tlang}_translated.hyp | sed 's/^H-//g' | sort -n | cut -f3 > $root/${tlang}_translated.true

python $fairseq_dir/scripts/spm_decode.py  --model $root/preprocess/flores200_sacrebleu_tokenizer_spm.model --input $root/${tlang}_translated.true > $root/${tlang}_translated.true.detok
