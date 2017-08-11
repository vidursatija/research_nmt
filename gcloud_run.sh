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