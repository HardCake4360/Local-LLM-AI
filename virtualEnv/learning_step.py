#virtual env adress
#C:\Users\user\Documents\GitHub\Local-LLM-AI\virtualEnv

import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math
import json
from torch.utils.tensorboard import SummaryWriter

############## 전역공간 ################

# 기본 단어 토큰 값
PAD_token = 0  # 짧은 문장을 채울(패딩, PADding) 때 사용할 제로 토큰
SOS_token = 1  # 문장의 시작(SOS, Start Of Sentence)을 나타내는 토큰
EOS_token = 2  # 문장의 끝(EOS, End Of Sentence)을 나태는 토큰
USR_TOKEN = "<usr>"
SYS_TOKEN = "<sys>"


MAX_LENGTH = 30  # 고려할 문장의 최대 길이

MIN_COUNT = 1    # 제외할 단어의 기준이 되는 등장 횟수

# 학습 및 최적화 설정
clip = 1.0
teacher_forcing_ratio = 1.0
learning_rate = 0.0001
decoder_learning_ratio = 5.0
checkpoint_iter = 35000
n_iteration = 100000
print_every = 1
save_every = 1000

##############################################################
class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # SOS, EOS, PAD를 센 것

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # 등장 횟수가 기준 이하인 단어를 정리합니다
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # 사전을 다시 초기화합니다
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3 # 기본 토큰을 센 것

        for word in keep_words:
            self.addWord(word)

class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # GRU를 초기화합니다. input_size와 hidden_size 매개변수는 둘 다 'hidden_size'로
        # 둡니다. 이는 우리 입력의 크기가 hideen_size 만큼의 피처를 갖는 단어 임베딩이기
        # 때문입니다.
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        # 단어 인덱스를 임베딩으로 변환합니다
        embedded = self.embedding(input_seq)
        # RNN 모듈을 위한 패딩된 배치 시퀀스를 패킹합니다
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # GRU로 포워드 패스를 수행합니다
        outputs, hidden = self.gru(packed, hidden)
        # 패딩을 언패킹합니다
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        # 양방향 GRU의 출력을 합산합니다
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        # 출력과 마지막 은닉 상태를 반환합니다
        return outputs, hidden

# Luong 어텐션 레이어
class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Attention 가중치(에너지)를 제안된 방법에 따라 계산합니다
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # max_length와 batch_size의 차원을 뒤집습니다
        attn_energies = attn_energies.t()

        # 정규화된 softmax 확률 점수를 반환합니다 (차원을 늘려서)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)
    
class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # 참조를 보존해 둡니다
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # 레이어를 정의합니다
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # 주의: 한 단위 시간에 대해 한 단계(단어)만을 수행합니다
        # 현재의 입력 단어에 대한 임베딩을 구합니다
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        # 무방향 GRU로 포워드 패스를 수행합니다
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # 현재의 GRU 출력을 바탕으로 어텐션 가중치를 계산합니다
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # 인코더 출력에 어텐션을 곱하여 새로운 "가중치 합" 문맥 벡터를 구합니다
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Luong의 논문에 나온 식 5를 이용하여 가중치 문맥 벡터와 GRU 출력을 결합합니다
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Luong의 논문에 나온 식 6을 이용하여 다음 단어를 예측합니다
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # 출력과 마지막 은닉 상태를 반환합니다
        return output, hidden
    
def get_teacher_forcing_ratio(iteration, total_iters):
    return max(0.5, 1.0 - (iteration / total_iters))


def printLines(file, n=10):
    with open(file, 'rb') as datafile:
        lines = datafile.readlines()
    for line in lines[:n]:
        print(line)
        
# 파일에 포함된 각 줄을 쪼개서 대사(line)와 대화(conversation)를 생성합니다.
def loadLinesAndConversations(fileName):
    lines = {}
    conversations = {}
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            lineJson = json.loads(line)
            # 필드를 추출하여 line 객체를 구성합니다
            lineObj = {}
            lineObj["lineID"] = lineJson["id"]
            lineObj["characterID"] = lineJson["speaker"]
            lineObj["text"] = lineJson["text"]
            lines[lineObj['lineID']] = lineObj

            # 필드를 추출하여 conversation 객체를 구성합니다
            if lineJson["conversation_id"] not in conversations:
                convObj = {}
                convObj["conversationID"] = lineJson["conversation_id"]
                convObj["movieID"] = lineJson["meta"]["movie_id"]
                convObj["lines"] = [lineObj]
            else:
                convObj = conversations[lineJson["conversation_id"]]
                convObj["lines"].insert(0, lineObj)
            conversations[convObj["conversationID"]] = convObj

    return lines, conversations

# conversation들에서 문장 쌍을 추출합니다
def extractSentencePairs(conversations):
    qa_pairs = []
    for conversation in conversations.values():
        # 대화를 이루는 각 대사에 대해 반복문을 수행합니다
        # 대화의 마지막 대사는 (그에 대한 응답이 없으므로) 무시합니다
        for i in range(len(conversation["lines"]) - 1):
            inputLine = conversation["lines"][i]["text"].strip()
            targetLine = conversation["lines"][i+1]["text"].strip()
            # 잘못된 샘플은 제거합니다(리스트가 하나라도 비어 있는 경우)
            if inputLine and targetLine:
                qa_pairs.append([inputLine, targetLine])
    return qa_pairs

def buildContextPairs(conversations, max_context_turns=2):
    qa_pairs = []
    for conv in conversations.values():
        context = []
        lines = conv["lines"]
        for i in range(1, len(lines)):
            prev = lines[i - 1]["text"].strip()
            curr = lines[i]["text"].strip()

            # 문맥 누적
            context.append(f"{USR_TOKEN.lower() if i % 2 == 1 else SYS_TOKEN} {prev}")
            if len(context) > max_context_turns:
                context = context[-max_context_turns:]

            full_context = ' '.join(context)
            response = f"{SYS_TOKEN.lower() if i % 2 == 1 else USR_TOKEN} {curr}"
            qa_pairs.append([
                    normalizeString(full_context), 
                    normalizeString(response)])
    return qa_pairs

# 유니코드 문자열을 아스키로 변환합니다
# https://stackoverflow.com/a/518232/2809427 참고
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# 소문자로 만들고, 공백을 넣고, 알파벳 외의 글자를 제거합니다
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z0-9:/.!?]+", r" ", s)  # 특수토큰 보존
    s = re.sub(r"\s+", r" ", s).strip()
    return s

# 질의/응답 쌍을 읽어서 voc 객체를 반환합니다
def readVocs(datafile, corpus_name):
    print("Reading lines...")
    # 파일을 읽고, 쪼개어 lines에 저장합니다
    lines = open(datafile, encoding='utf-8').\
        read().strip().split('\n')
    # 각 줄을 쪼개어 pairs에 저장하고 정규화합니다
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    voc = Voc(corpus_name)
    
    # 특수 토큰 강제 등록 및 카운트 보정
    # for token in [USR_TOKEN + ":", SYS_TOKEN + ":"]:
    #     voc.addWord(token)
    #     voc.word2count[token] = voc.word2count.get(token, 0) + 9999
    
    return voc, pairs

# 문장의 쌍 'p'에 포함된 두 문장이 모두 MAX_LENGTH라는 기준보다 짧은지를 반환합니다
def filterPair(p):
    # EOS 토큰을 위해 입력 시퀀스의 마지막 단어를 보존해야 합니다
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

# 조건식 ``filterPair`` 에 따라 pairs를 필터링합니다
def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

# 앞에서 정의한 함수를 이용하여 만든 voc 객체와 리스트 pairs를 반환합니다
def loadPrepareData(corpus, corpus_name, datafile, save_dir):
    print("Start preparing training data ...")
    
    # voc, pairs 불러오기
    voc, pairs = readVocs(datafile, corpus_name)
    
    voc.addWord(USR_TOKEN.lower())
    voc.word2count[USR_TOKEN.lower()] = 9999
    voc.addWord(SYS_TOKEN.lower())
    voc.word2count[SYS_TOKEN.lower()] = 9999
    
    print("Read {!s} sentence pairs".format(len(pairs)))

    # 문장 길이 필터링
    pairs = filterPairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))

    # 단어 추가
    print("Counting words...")
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])

    # # === 특수 토큰 직접 추가 및 보존 처리 ===
    # for token in [USR_TOKEN + ":", SYS_TOKEN + ":"]:
    #     voc.addWord(token)
    #     # 빈도수를 높게 설정해서 trim() 시 삭제되지 않도록
    #     voc.word2count[token] = voc.word2count.get(token, 0) + 9999
    # # =====================================

    print("Counted words:", voc.num_words)
    return voc, pairs

def trimRareWords(voc, pairs, MIN_COUNT):
    voc.trim(MIN_COUNT)

    # 🔁 재등록 보장 (지금까지 사용한 코드)
    for token in [USR_TOKEN.lower(), SYS_TOKEN.lower()]:
        if token not in voc.word2index:
            voc.addWord(token)
            voc.word2count[token] = 9999

    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = all(word in voc.word2index for word in input_sentence.split(' '))
        keep_output = all(word in voc.word2index for word in output_sentence.split(' '))
        if keep_input and keep_output:
            keep_pairs.append(pair)
        else:
            # 🔍 디버깅 출력 (임시)
            if not keep_input or not keep_output:
                unknown_words = [w for w in input_sentence.split(' ') + output_sentence.split(' ') if w not in voc.word2index]
                print(f" 제외된 pair (unknown: {unknown_words}): {pair}")

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs


def indexesFromSentence(voc, sentence):
    result = []
    for word in sentence.split(' '):
        if word in voc.word2index:
            result.append(voc.word2index[word])
        else:
            continue  # 또는 UNK 토큰 처리 가능
    return result + [EOS_token]


def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

# 입력 시퀀스 텐서에 패딩한 결과와 lengths를 반환합니다
def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

# 패딩한 목표 시퀀스 텐서, 패딩 마스크, 그리고 최대 목표 길이를 반환합니다
def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.BoolTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len

# 입력 배치를 이루는 쌍에 대한 모든 아이템을 반환합니다
def batch2TrainData(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len

def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()

def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding,
          encoder_optimizer, decoder_optimizer, batch_size, clip, current_iter, max_length=MAX_LENGTH):

    # 제로 그라디언트
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # device 옵션을 설정합니다
    input_variable = input_variable.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)
    # RNN 패킹의 길이는 항상 CPU에 위치해야 합니다
    lengths = lengths.to("cpu")

    # 변수를 초기화합니다
    loss = 0
    print_losses = []
    n_totals = 0

    # 인코더로 포워드 패스를 수행합니다
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # 초기 디코더 입력을 생성합니다(각 문장을 SOS 토큰으로 시작합니다)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # 디코더의 초기 은닉 상태를 인코더의 마지막 은닉 상태로 둡니다
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # 이번 반복에서 teacher forcing을 사용할지를 결정합니다
    use_teacher_forcing = random.random() < get_teacher_forcing_ratio(current_iter, n_iteration)

    # 배치 시퀀스를 한 번에 하나씩 디코더로 포워드 패스합니다
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # Teacher forcing 사용: 다음 입력을 현재의 목표로 둡니다
            decoder_input = target_variable[t].view(1, -1)
            # 손실을 계산하고 누적합니다
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # Teacher forcing 미사용: 다음 입력을 디코더의 출력으로 둡니다
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            # 손실을 계산하고 누적합니다
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    # 역전파를 수행합니다
    loss.backward()

    # 그라디언트 클리핑: 그라디언트를 제자리에서 수정합니다
    _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # 모델의 가중치를 수정합니다
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals

def trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size, print_every, save_every, clip, corpus_name, loadFilename):
    
    # 각 단계에 대한 배치를 읽어옵니다
    training_batches = [batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)])
                      for _ in range(n_iteration)]

    # 초기화
    print('Initializing ...')
    start_iteration = 1
    print_loss = 0
    if loadFilename:
        start_iteration = checkpoint['iteration'] + 1

    writer = SummaryWriter(log_dir=os.path.join("runs", model_name))

    # 학습 루프
    print("Training...")
    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = training_batches[iteration - 1]
        # 배치에서 각 필드를 읽어옵니다
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        # 배치에 대해 학습을 한 단계 진행합니다
        loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                     decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip,iteration)
        print_loss += loss

        # 경과를 출력합니다
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration, iteration / n_iteration * 100, print_loss_avg))
            writer.add_scalar("Loss/train", print_loss_avg, iteration)
            print_loss = 0


        # Checkpoint를 저장합니다
        if (iteration % save_every == 0):
            directory = os.path.join(save_dir, model_name, corpus_name, '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc_dict': voc.__dict__,
                'embedding': embedding.state_dict()
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))
    writer.close()

class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        # 인코더 모델로 입력을 포워드 패스합니다
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # 인코더의 마지막 은닉 레이어가 디코더의 첫 번째 은닉 레이어의 입력이 되도록 준비합니다
        decoder_hidden = encoder_hidden[:decoder.n_layers]
        # 디코더의 첫 번째 입력을 SOS_token으로 초기화합니다
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
        # 디코더가 단어를 덧붙여 나갈 텐서를 초기화합니다
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        # 반복적으로 각 단계마다 하나의 단어 토큰을 디코딩합니다
        for _ in range(max_length):
            # 디코더로의 포워드 패스를 수행합니다
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # 가장 가능성 높은 단어 토큰과 그 softmax 점수를 구합니다
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # 토큰과 점수를 기록합니다
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # 현재의 토큰을 디코더의 다음 입력으로 준비시킵니다(차원을 증가시켜서)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # 단어 토큰과 점수를 모아서 반환합니다
        return all_tokens, all_scores
    
def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):
    ### 입력 시퀀스를 배치 형태로 만듭니다
    # 단어 -> 인덱스
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # lengths 텐서를 만듭니다
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # 배치의 차원을 뒤집어서 모델이 사용하는 형태로 만듭니다
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # 적절한 디바이스를 사용합니다
    input_batch = input_batch.to(device)
    lengths = lengths.to("cpu")
    # searcher를 이용하여 문장을 디코딩합니다
    tokens, scores = searcher(input_batch, lengths, max_length)
    # 인덱스 -> 단어
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


def evaluateInput(encoder, decoder, searcher, voc):
    input_sentence = ''
    while(1):
        try:
            # 입력 문장을 받아옵니다
            input_sentence = input('> ')
            # 종료 조건인지 검사합니다
            if input_sentence == 'q' or input_sentence == 'quit': break
            # 문장을 정규화합니다
            input_sentence = normalizeString(input_sentence)
            # 문장을 평가합니다
            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
            # 응답 문장을 형식에 맞춰 출력합니다
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")
            
def removeUnknownPairs(voc, pairs):
    clean_pairs = []
    for pair in pairs:
        if all(word in voc.word2index for word in pair[0].split(' ')) and \
           all(word in voc.word2index for word in pair[1].split(' ')):
            clean_pairs.append(pair)
        else:
            print(f"unknown 제거됨: {pair}")
    return clean_pairs

if __name__ == '__main__':
    
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")

    corpus_name = "movie-corpus"
    corpus = os.path.join("data", corpus_name)

    #printLines(os.path.join(corpus, "utterances.jsonl"))
    
    # 새 파일에 대한 경로를 정의합니다
    datafile = os.path.join(corpus, "formatted_movie_lines.txt")

    delimiter = '\t'
    # 구분자에 대해 unescape 함수를 호출합니다
    delimiter = str(codecs.decode(delimiter, "unicode_escape"))

    # 대사 사전(lines dict)과 대화 사전(conversations dict)을 초기화합니다
    lines = {}
    conversations = {}
    # 대사와 대화를 불러옵니다
    print("\nProcessing corpus into lines and conversations...")
    lines, conversations = loadLinesAndConversations(os.path.join(corpus, "utterances.jsonl"))

    # 결과를 새로운 csv 파일로 저장합니다
    print("\nWriting newly formatted file...")
    with open(datafile, 'w', encoding='utf-8') as outputfile:
        writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
        for pair in buildContextPairs(conversations, max_context_turns=2):
            writer.writerow(pair)

    # 몇 줄을 예제 삼아 출력해 봅니다
    #print("\nSample lines from file:")
    #printLines(datafile)
    
    # voc와 pairs를 읽고 재구성합니다
    save_dir = os.path.join("data", "save")
    voc, pairs = loadPrepareData(corpus, corpus_name, datafile, save_dir)
    # 검증을 위해 pairs의 일부 내용을 출력해 봅니다
    print("\npairs:")
    for pair in pairs[:10]:
        print(pair)
        
    # voc와 pairs를 정돈합니다
    pairs = trimRareWords(voc, pairs, MIN_COUNT)
    
    # 검증용 예시
    small_batch_size = 5
    batches = batch2TrainData(voc, [random.choice(pairs) for _ in range(small_batch_size)])
    input_variable, lengths, target_variable, mask, max_target_len = batches

    # print("input_variable:", input_variable)
    # print("lengths:", lengths)
    # print("target_variable:", target_variable)
    # print("mask:", mask)
    # print("max_target_len:", max_target_len)
    
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

    # 불러올 checkpoint를 설정합니다. 처음부터 시작할 때는 None으로 둡니다.
    loadFilename = None

    
    # loadFilename = os.path.join(save_dir, model_name, corpus_name,
    #                 '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
    #                 '{}_checkpoint.tar'.format(checkpoint_iter))
    
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


    print('Building encoder and decoder ...')
    # 단어 임베딩을 초기화합니다
    embedding = nn.Embedding(voc.num_words, hidden_size)
    if loadFilename:
        embedding.load_state_dict(embedding_sd)
    # 인코더 및 디코더 모델을 초기화합니다
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
    if loadFilename:
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)
    # 적절한 디바이스를 사용합니다
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    print('Models built and ready to go!')
    
    #학습 단계
    if False:
        # Dropout 레이어를 학습 모드로 둡니다
        encoder.train()
        decoder.train()

        # Optimizer를 초기화합니다
        print('Building optimizers ...')
        encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
        if loadFilename:
            encoder_optimizer.load_state_dict(encoder_optimizer_sd)
            decoder_optimizer.load_state_dict(decoder_optimizer_sd)

        # CUDA가 있으면 CUDA를 설정합니다
        for state in encoder_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

        for state in decoder_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

        pairs = removeUnknownPairs(voc, pairs)

        # 학습 단계를 수행합니다
        print("Starting Training!")
        trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
           embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
           print_every, save_every, clip, corpus_name, loadFilename)
        print(f"학습쌍 수: {len(pairs)}")
    
    if True:
        # Dropout 레이어를 평가( ``eval`` ) 모드로 설정합니다
        encoder.eval()
        decoder.eval()

        # 탐색 모듈을 초기화합니다
        searcher = GreedySearchDecoder(encoder, decoder)

        # 채팅을 시작합니다 (다음 줄의 주석을 제거하면 시작해볼 수 있습니다)
        evaluateInput(encoder, decoder, searcher, voc)