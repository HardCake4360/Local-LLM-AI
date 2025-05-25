import learning_step as ls
import torch
import torch.nn as nn
import os

#불러올 학습파일
checkpoint_iter = 35000

# 모델을 설정합니다
model_name = 'cb_model'
attn_model = 'dot'
#``attn_model = 'general'``
#``attn_model = 'concat'``
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64

corpus_name = "movie-corpus"
corpus = os.path.join("data", corpus_name)

#printLines(os.path.join(corpus, "utterances.jsonl"))
    
# 새 파일에 대한 경로를 정의합니다
datafile = os.path.join(corpus, "formatted_movie_lines.txt")

save_dir = os.path.join("data", "save")
voc, pairs = ls.loadPrepareData(corpus, corpus_name, datafile, save_dir)

# 불러올 checkpoint를 설정합니다. 처음부터 시작할 때는 None으로 둡니다.
loadFilename = None

    
loadFilename = os.path.join(save_dir, model_name, corpus_name,
                    '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
                    '{}_checkpoint.tar'.format(checkpoint_iter))
    
# ``loadFilename`` 이 존재하는 경우에는 모델을 불러옵니다
if loadFilename:
    # 모델을 학습할 때와 같은 기기에서 불러오는 경우
    checkpoint = torch.load(loadFilename)
    # GPU에서 학습한 모델을 CPU로 불러오는 경우
    #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']

# 단어 임베딩을 초기화합니다
embedding = nn.Embedding(voc.num_words, hidden_size)

# 인코더 및 디코더 모델을 초기화합니다
encoder = ls.EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = ls.LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)

# Dropout 레이어를 평가( ``eval`` ) 모드로 설정합니다
encoder.eval()
decoder.eval()

# 탐색 모듈을 초기화합니다
searcher = ls.GreedySearchDecoder(encoder, decoder)

# 채팅을 시작합니다
ls.evaluateInput(encoder, decoder, searcher, voc)