gcloud ml-engine jobs submit training $JOB_NAME --job-dir gs://rynet-425739-mlengine/$JOB_NAME/ --runtime-version 1.0 --module-name train.app \
--package-path train --region us-east1  --scale-tier=BASIC_GPU \
-- --pickle_dir gs://rynet-425739-mlengine/data/words_phonemes.p --learn_rate 0.08 --model_dir gs://rynet-425739-mlengine/$JOB_NAME/ \
--model_name rnn2_alphabets 

gcloud ml-engine jobs submit training $JOB_NAME --job-dir gs://rynet-425739-mlengine/$JOB_NAME/ --runtime-version 1.0 --module-name train.app \
--package-path train --region us-east1  --scale-tier=BASIC_GPU \
-- --pickle_dir gs://rynet-425739-mlengine/data/words_phonemes.p --learn_rate 0.085 --model_dir gs://rynet-425739-mlengine/$JOB_NAME \
--vocab_size 41 --model_name rnn2_phonemes

gcloud ml-engine jobs submit training $JOB_NAME --job-dir gs://rynet-425739-mlengine/$JOB_NAME/ --runtime-version 1.0 --module-name train.app \
--package-path train --region us-east1  --scale-tier=BASIC_GPU \
-- --pickle_dir gs://rynet-425739-mlengine/data/words_phonemes.p --learn_rate 0.008 --model_dir gs://rynet-425739-mlengine/nmt_alphabets_V2_9/rnn2_alphabets/ \
--model_name rnn2_alphabets 

gcloud ml-engine jobs submit training $JOB_NAME --job-dir gs://rynet-425739-mlengine/$JOB_NAME/ --runtime-version 1.0 --module-name train.app \
--package-path train --region us-east1  --scale-tier=BASIC_GPU \
-- --pickle_dir gs://rynet-425739-mlengine/data/words_phonemes.p --learn_rate 0.06 --model_dir gs://rynet-425739-mlengine/nmt_phonemes_V2_3/rnn2_phonemes/ \
--vocab_size 41 --model_name rnn2_phonemes  

gcloud ml-engine jobs submit training $JOB_NAME --job-dir gs://rynet-425739-mlengine/$JOB_NAME/ --runtime-version 1.0 --module-name train.converter \
--package-path train --region us-east1 \
-- --model_dir gs://rynet-425739-mlengine/nmt_alphabets_V2_9/rnn2_alphabets/rnn2_alphabets \
--model_export gs://rynet-425739-mlengine/nmt_alphabets_V2_9/rnn2_alphabets_en --vocab_size 28 --mode 1

gcloud ml-engine jobs submit training $JOB_NAME --job-dir gs://rynet-425739-mlengine/$JOB_NAME/ --runtime-version 1.0 --module-name train.converter \
--package-path train --region us-east1 \
-- --model_dir gs://rynet-425739-mlengine/nmt_phonemes_V2_3/rnn2_phonemes/rnn2_phonemes \
--model_export gs://rynet-425739-mlengine/nmt_phonemes_V2_3/rnn2_phonemes_de --vocab_size 41 --mode 0

gcloud ml-engine jobs submit training $JOB_NAME --job-dir gs://rynet-425739-mlengine/$JOB_NAME/ --runtime-version 1.0 --module-name train.translate \
--package-path train --region us-east1 --scale-tier=BASIC_GPU \
-- --pickle_dir gs://rynet-425739-mlengine/data/words_phonemes.p --learn_rate 0.6 --model_dir gs://rynet-425739-mlengine/$JOB_NAME/ \
--from_model gs://rynet-425739-mlengine/nmt_alphabets_V2_9/rnn2_alphabets_en \
--to_model gs://rynet-425739-mlengine/nmt_phonemes_V2_3/rnn2_phonemes_de

gcloud ml-engine jobs submit training $JOB_NAME --job-dir gs://rynet-425739-mlengine/$JOB_NAME/ --runtime-version 1.0 --module-name train.translate \
--package-path train --region us-east1  --scale-tier=BASIC_GPU \
-- --pickle_dir gs://rynet-425739-mlengine/data/words_phonemes.p --learn_rate 0.000004 --model_dir gs://rynet-425739-mlengine/atop_full_newxx_2/ \
--from_model gs://rynet-425739-mlengine/nmt_alphabets_V2_9/rnn2_alphabets_en \
--to_model gs://rynet-425739-mlengine/nmt_phonemes_V2_3/rnn2_phonemes_de

gcloud ml-engine jobs submit training $JOB_NAME --job-dir gs://rynet-425739-mlengine/$JOB_NAME/ --runtime-version 1.0 --module-name train.translate \
--package-path train --region us-east1 --scale-tier=BASIC_GPU \
-- --pickle_dir gs://rynet-425739-mlengine/data/words_phonemes.p --learn_rate 0.03 --model_dir gs://rynet-425739-mlengine/$JOB_NAME/ \
--from_model gs://rynet-425739-mlengine/$JOB_NAME/encoder_alpha --to_model gs://rynet-425739-mlengine/$JOB_NAME/decoder_alpha \
--model_name alphabets_ende

gcloud ml-engine jobs submit training $JOB_NAME --job-dir gs://rynet-425739-mlengine/$JOB_NAME/ --runtime-version 1.0 --module-name train.translate \
--package-path train --region us-east1 --scale-tier=BASIC_GPU \
-- --pickle_dir gs://rynet-425739-mlengine/data/words_phonemes.p --learn_rate 0.06 --model_dir gs://rynet-425739-mlengine/$JOB_NAME/ \
--from_model gs://rynet-425739-mlengine/$JOB_NAME/encoder_alpha --to_model gs://rynet-425739-mlengine/$JOB_NAME/decoder_alpha \
--model_name phonemes_ende

gcloud ml-engine jobs submit training $JOB_NAME --job-dir gs://rynet-425739-mlengine/$JOB_NAME/ --runtime-version 1.0 --module-name train.half_translate \
--package-path train --region us-east1  --scale-tier=BASIC_GPU \
-- --pickle_dir gs://rynet-425739-mlengine/data/words_phonemes.p --learn_rate 16.0 --model_dir gs://rynet-425739-mlengine/$JOB_NAME/ \
--from_model gs://rynet-425739-mlengine/nmt_alphabets_V2_9/rnn2_alphabets_en

gcloud ml-engine jobs submit training $JOB_NAME --job-dir gs://rynet-425739-mlengine/$JOB_NAME/ --runtime-version 1.0 --module-name train.e2e \
--package-path train --region us-east1  --scale-tier=BASIC_GPU \
-- --pickle_dir gs://rynet-425739-mlengine/data/words_phonemes.p --learn_rate 0.0008 --model_dir gs://rynet-425739-mlengine/atop_e2e_3/

gcloud ml-engine jobs submit training $JOB_NAME --job-dir gs://rynet-425739-mlengine/$JOB_NAME/ --runtime-version 1.0 --module-name train.e2e_ww \
--package-path train --region us-east1  --scale-tier=BASIC_GPU \
-- --pickle_dir gs://rynet-425739-mlengine/data/words_phonemes.p --learn_rate 0.5 --model_dir gs://rynet-425739-mlengine/$JOB_NAME/

gcloud ml-engine jobs submit training $JOB_NAME --job-dir gs://rynet-425739-mlengine/$JOB_NAME/ --runtime-version 1.0 --module-name train.e2e_ww \
--package-path train --region us-east1  --scale-tier=BASIC_GPU \
-- --pickle_dir gs://rynet-425739-mlengine/data/words_phonemes.p --learn_rate 0.5 --model_dir gs://rynet-425739-mlengine/atop_e2eww_12/