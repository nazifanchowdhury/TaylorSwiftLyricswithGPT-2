!pip install -q gpt-2-simple
import gpt_2_simple as gpt2
from google.colab import files

gpt2.download_gpt2(model_name="124M")
file_name = "TaylorLyrics.txt"
sess = gpt2.start_tf_sess()

gpt2.finetune(sess,
              dataset=file_name,
              model_name='124M',
              steps=1000
              )

gpt2.generate(sess)
