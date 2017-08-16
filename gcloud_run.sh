gcloud ml-engine jobs submit training $JOB_NAME --job-dir gs://rynet-425739-mlengine/$JOB_NAME/ --runtime-version 1.0 --module-name train.app \
--package-path train --region us-central1  --scale-tier=BASIC_GPU \
-- --pickle_dir gs://rynet-425739-mlengine/data/words_phonemes.p --learn_rate 0.25 --model_dir gs://rynet-425739-mlengine/$JOB_NAME/rnn2_alphabets/ \
--model_name rnn2_alphabets 

gcloud ml-engine jobs submit training $JOB_NAME --job-dir gs://rynet-425739-mlengine/$JOB_NAME/ --runtime-version 1.0 --module-name train.app \
--package-path train --region us-central1  --scale-tier=BASIC_GPU \
-- --pickle_dir gs://rynet-425739-mlengine/data/words_phonemes.p --learn_rate 0.2 --model_dir gs://rynet-425739-mlengine/$JOB_NAME/rnn2_phonemes/ \
--vocab_size 41 --model_name rnn2_phonemes 

gcloud ml-engine jobs submit training $JOB_NAME --job-dir gs://rynet-425739-mlengine/$JOB_NAME/ --runtime-version 1.0 --module-name train.app \
--package-path train --region us-central1  --scale-tier=BASIC_GPU \
-- --pickle_dir gs://rynet-425739-mlengine/data/words_phonemes.p --learn_rate 0.08 --model_dir gs://rynet-425739-mlengine/nmt_alphabets_V1_5/rnn2_alphabets/ \
--model_name rnn2_alphabets 

gcloud ml-engine jobs submit training $JOB_NAME --job-dir gs://rynet-425739-mlengine/$JOB_NAME/ --runtime-version 1.0 --module-name train.app \
--package-path train --region us-central1  --scale-tier=BASIC_GPU \
-- --pickle_dir gs://rynet-425739-mlengine/data/words_phonemes.p --learn_rate 0.06 --model_dir gs://rynet-425739-mlengine/nmt_phonemes_V1_5/rnn2_phonemes/ \
--vocab_size 41 --model_name rnn2_phonemes 

gcloud ml-engine jobs submit training $JOB_NAME --job-dir gs://rynet-425739-mlengine/$JOB_NAME/ --runtime-version 1.0 --module-name train.app \
--package-path train --region us-central1  --scale-tier=BASIC_GPU \
-- --pickle_dir gs://rynet-425739-mlengine/data/words_phonemes.p --model_dir gs://rynet-425739-mlengine/nmt_alphabets_V1_5/rnn2_alphabets/ \
--model_name rnn2_alphabets  --feed_forward

gcloud ml-engine jobs submit training $JOB_NAME --job-dir gs://rynet-425739-mlengine/$JOB_NAME/ --runtime-version 1.0 --module-name train.translate \
--package-path train --region us-central1  --scale-tier=BASIC_GPU \
-- --pickle_dir gs://rynet-425739-mlengine/data/words_phonemes.p --learn_rate 0.01 --model_dir gs://rynet-425739-mlengine/$JOB_NAME/ \
--from_model gs://rynet-425739-mlengine/nmt_alphabets_V1_5/rnn2_alphabets/ \
--to_model gs://rynet-425739-mlengine/nmt_phonemes_V1_5/rnn2_phonemes/

 gcloud ml-engine jobs submit training $JOB_NAME --job-dir gs://rynet-425739-mlengine/$JOB_NAME/ --runtime-version 1.0 --module-name train.translate \
--package-path train --region us-east1  --scale-tier=BASIC_GPU \
-- --pickle_dir gs://rynet-425739-mlengine/data/words_phonemes.p --learn_rate 0.015 --model_dir gs://rynet-425739-mlengine/$JOB_NAME/ \
--from_model gs://rynet-425739-mlengine/nmt_alphabets_V1_5/rnn2_alphabets_en \
--to_model gs://rynet-425739-mlengine/nmt_phonemes_V1_5/rnn2_phonemes_de

gcloud ml-engine jobs submit training $JOB_NAME --job-dir gs://rynet-425739-mlengine/$JOB_NAME/ --runtime-version 1.0 --module-name train.translate \
--package-path train --region us-east1 \
-- --pickle_dir gs://rynet-425739-mlengine/data/words_phonemes.p --learn_rate 0.01 --model_dir gs://rynet-425739-mlengine/$JOB_NAME/ \
--from_model gs://rynet-425739-mlengine/nmt_alphabets_V1_5/rnn2_alphabets_en \
--to_model gs://rynet-425739-mlengine/nmt_phonemes_V1_5/rnn2_phonemes_de

gcloud ml-engine jobs submit training $JOB_NAME --job-dir gs://rynet-425739-mlengine/$JOB_NAME/ --runtime-version 1.0 --module-name train.converter \
--package-path train --region us-east1 \
-- --model_dir gs://rynet-425739-mlengine/nmt_alphabets_V1_5/rnn2_alphabets/rnn2_alphabets \
--model_export gs://rynet-425739-mlengine/nmt_alphabets_V1_5/rnn2_alphabets_en --vocab_size 28 --mode 1

gcloud ml-engine jobs submit training $JOB_NAME --job-dir gs://rynet-425739-mlengine/$JOB_NAME/ --runtime-version 1.0 --module-name train.converter \
--package-path train --region us-east1 \
-- --model_dir gs://rynet-425739-mlengine/nmt_phonemes_V1_5/rnn2_phonemes/rnn2_phonemes \
--model_export gs://rynet-425739-mlengine/nmt_phonemes_V1_5/rnn2_phonemes_de --vocab_size 41 --mode 0

gcloud ml-engine jobs submit training $JOB_NAME --job-dir gs://rynet-425739-mlengine/$JOB_NAME/ --runtime-version 1.0 --module-name train.translate \
--package-path train --region us-east1  --scale-tier=BASIC_GPU \
-- --pickle_dir gs://rynet-425739-mlengine/data/words_phonemes.p --learn_rate 0.01 --model_dir gs://rynet-425739-mlengine/atop_full_p6/ \
--from_model gs://rynet-425739-mlengine/nmt_alphabets_V1_5/rnn2_alphabets_en \
--to_model gs://rynet-425739-mlengine/nmt_phonemes_V1_5/rnn2_phonemes_de

gcloud ml-engine jobs submit training $JOB_NAME --job-dir gs://rynet-425739-mlengine/$JOB_NAME/ --runtime-version 1.0 --module-name train.translate \
--package-path train --region us-east1  --scale-tier=BASIC_GPU \
-- --pickle_dir gs://rynet-425739-mlengine/data/words_phonemes.p --learn_rate 0.01 --model_dir gs://rynet-425739-mlengine/atop_full_p5/ \
--from_model gs://rynet-425739-mlengine/nmt_alphabets_V1_5/rnn2_alphabets_en \
--to_model gs://rynet-425739-mlengine/nmt_phonemes_V1_5/rnn2_phonemes_de