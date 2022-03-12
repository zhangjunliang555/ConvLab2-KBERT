import numpy as np
import torch
import random
#from transformers import BertTokenizer

import math
from collections import Counter
from convlab2.nlu.jointBERT.uer.utils.constants import *
from convlab2.nlu.jointBERT.uer.utils.tokenizer import BertTokenizer

class Dataloader:
    def __init__(self, intent_vocab, tag_vocab, bert_vocab, knowledge_graph,args=None):
        """
        :param intent_vocab: list of all intents
        :param tag_vocab: list of all tags
        :param pretrained_weights: which bert, e.g. 'bert-base-uncased'
        """
        self.intent_vocab = intent_vocab
        self.tag_vocab = tag_vocab
        self.intent_dim = len(intent_vocab)
        self.tag_dim = len(tag_vocab)
        self.tag_id_PAD = self.tag_dim
        self.tag_id_ENT = self.tag_dim + 1
        self.tag_id_CLS = self.tag_dim+2
        self.tag_id_SEP = self.tag_dim + 3
        self.id2intent = dict([(i, x) for i, x in enumerate(intent_vocab)])
        self.intent2id = dict([(x, i) for i, x in enumerate(intent_vocab)])
        self.id2tag = dict([(i, x) for i, x in enumerate(tag_vocab)])
        self.tag2id = dict([(x, i) for i, x in enumerate(tag_vocab)])
        if args is not None:
            self.tokenizer = BertTokenizer(args)
        else:
            self.tokenizer=None
        self.bert_vocab=bert_vocab
        self.knowledge_graph = knowledge_graph
        self.data = {}
        self.intent_weight = [1] * len(self.intent2id)

    def load_data(self, data, data_key, cut_sen_len):
        """
        sample representation: [list of words, list of tags, list of intents, original dialog act]
        :param data_key: train/val/test
        :param data:
        :return:
        """
        #过滤无意图语料
        #print(data)
        data_list=[]
        sum=0
        i=0
        for d in data:
            i = i + 1
            if len(d[2])>0:
                data_list.append(d)
            else:
                sum+= 1
                #print(d)
                #print(i)
        print("bad intent sum=",sum)
        #self.data[data_key] = data
        self.data[data_key] = data_list
        max_sen_len, max_context_len = 0, 0
        sen_len = []
        context_len = []
        for d in self.data[data_key]:
            max_sen_len = max(max_sen_len, len(d[0]))
            sen_len.append(len(d[0]))
            # d = (tokens, tags, intents, da2triples(turn["dialog_act"], context(list of str))
            if cut_sen_len > 0:
                d[0] = d[0][:cut_sen_len]
                d[1] = d[1][:cut_sen_len]
                d[4] = [''.join(s.split()[:cut_sen_len]) for s in d[4]]

            #if len(self.knowledge_graph.spo_file_paths)>0:
            if 1 < 0:
                context = '[CLS]' + '[SEP]'.join(d[4])
                context_tokens, context_pos, context_vm, _ = self.knowledge_graph.add_knowledge_with_vm([context], max_length=cut_sen_len*3+30*3)
            else:
                #print(d[4])
                context = ["[CLS]"]
                for s in d[4]:
                    context += list(s)
                    context +=["[SEP]"]
                #print('len(context)=', len(context))
                #print(context)
                context_tokens, context_pos, context_vm, _ = self.knowledge_graph.raw_with_vm_batch([context],max_length=cut_sen_len * 3+4)
            context_tokens = context_tokens[0]
            context_pos = context_pos[0]
            context_vm = context_vm[0].astype("bool")
            max_context_len = max(max_context_len, len(context_tokens))
            context_len.append(len(context_tokens))
            context_seq = [self.bert_vocab.get(t) for t in context_tokens]
            d.append(context_pos)
            d.append(context_vm)
            d.append(context_seq)#-6

            if len(self.knowledge_graph.spo_file_paths)>0:
                text = '[CLS]' + ''.join(d[0])+'[SEP]'
                #增加了知识，最大长度增加一些
                tokens, pos, vm, tag = self.knowledge_graph.add_knowledge_with_vm([text],max_length=cut_sen_len+cut_sen_len//2,raw_sent_batch=[["[CLS]"]+d[0]+["[SEP]"]])
            else:
                text=["[CLS]"] + d[0]+["[SEP]"]
                #print('len(text)=', len(text))
                #print(text)
                tokens, pos, vm, tag = self.knowledge_graph.raw_with_vm_batch([text], max_length=cut_sen_len + 2)
            tokens = tokens[0]
            pos = pos[0]
            vm = vm[0].astype("bool")
            tag = tag[0]


            word_seq = [self.bert_vocab.get(t) for t in tokens]
            #new2ori = None
            tag_seq = []
            j = 0

            #print('len(d[1])=', len(d[1]))
            #print('len(pos)=', len(pos))
            #print('len(d[0])=', len(d[0]))
            #print(d[0])
            #print('len(tokens)=', len(tokens))
            #print(tokens)
            #print(word_seq)
            #print(d[1])
            #print('len(tag)=', len(tag))
            #print(tag)
            #token_tmp=""
            for i in range(len(word_seq)):
                if tag[i] == 0 and word_seq[i] != PAD_ID and word_seq[i] != CLS_ID and word_seq[i] != SEP_ID:
                    #print('i={},j={},word_seq={},d[1][j]={}'.format(i,j,word_seq[i],d[1][j]))
                    #token_tmp+=tokens[i]
                    #if token_tmp==d[0][j]:
                    if tokens[i] == d[0][j]:
                        tag_seq.append(self.tag2id[d[1][j]])
                        j += 1
                        #token_tmp=""
                elif tag[i] == 1 and word_seq[i] != PAD_ID and word_seq[i] != CLS_ID:  # 是添加的实体
                    tag_seq.append(self.tag_id_ENT)
                elif word_seq[i] == CLS_ID:
                    tag_seq.append(self.tag_id_CLS)
                elif word_seq[i] == SEP_ID:
                    tag_seq.append(self.tag_id_SEP)
                else:#CLS SEP PAD
                    tag_seq.append(self.tag_id_PAD)
            """
            if j!=len(d[0]) or  j!=len(d[1]):
                print(tokens)
                print(word_seq)
                print(tag_seq)
                print(d[0])
                print(d[1])
                assert False
            """

            d.append(pos)
            d.append(vm)
            #d.append(tag)
            #d.append(new2ori)
            d.append(word_seq)#-3
            #d.append(self.seq_tag2id(tag_seq))
            d.append(tag_seq)#-2
            d.append(self.seq_intent2id(d[2]))#-1
            # d = (tokens, tags, intents, da2triples(turn["dialog_act"]), context(token id), new2ori, new_word_seq, tag2id_seq, intent2id_seq)
            if data_key=='train':
                for intent_id in d[-1]:
                    self.intent_weight[intent_id] += 1
        if data_key == 'train':
            train_size = len(self.data['train'])
            #解决样本均衡
            for intent, intent_id in self.intent2id.items():
                neg_pos = (train_size - self.intent_weight[intent_id]) / self.intent_weight[intent_id]
                self.intent_weight[intent_id] = np.log10(neg_pos)
                #print(f",intent_id:{intent_id}，intent_weight:{self.intent_weight[intent_id]}")
            self.intent_weight = torch.tensor(self.intent_weight)
        print('max sen bert len', max_sen_len)
        print(sorted(Counter(sen_len).items()))
        print('max context bert len', max_context_len)
        print(sorted(Counter(context_len).items()))
    def load_utterance(self, utterance, context=list(), cut_sen_len=60):
        """
        :return:
        """
        d=[]
        d.append(self.tokenizer.tokenize(utterance))#d[0]
        d.append(['O'] * len(d[0]))#d[1]
        d.append([])
        d.append({})
        d.append(context[-3:])#d[4]
        if cut_sen_len > 0:
            d[0] = d[0][:cut_sen_len]
            d[1] = d[1][:cut_sen_len]
            d[4] = [''.join(s.split()[:cut_sen_len]) for s in d[4]]

        #if len(self.knowledge_graph.spo_file_paths)>0:
        if 1 < 0:
            context = '[CLS]' + '[SEP]'.join(d[4])
            context_tokens, context_pos, context_vm, _ = self.knowledge_graph.add_knowledge_with_vm([context], max_length=cut_sen_len*3+30*3)
        else:
            #print(d[4])
            context = ["[CLS]"]
            for s in d[4]:
                context += list(s)
                context +=["[SEP]"]
            #print('len(context)=', len(context))
            #print(context)
            context_tokens, context_pos, context_vm, _ = self.knowledge_graph.raw_with_vm_batch([context],max_length=cut_sen_len * 3+4)
        context_tokens = context_tokens[0]
        context_pos = context_pos[0]
        context_vm = context_vm[0].astype("bool")
        context_seq = [self.bert_vocab.get(t) for t in context_tokens]
        d.append(context_pos)#-8
        d.append(context_vm)
        d.append(context_seq)#-6

        if len(self.knowledge_graph.spo_file_paths)>0:
            text = '[CLS]' + ''.join(d[0])+'[SEP]'
            #增加了知识，最大长度增加一些
            tokens, pos, vm, tag = self.knowledge_graph.add_knowledge_with_vm([text],max_length=cut_sen_len+cut_sen_len//2,raw_sent_batch=[["[CLS]"]+d[0]+["[SEP]"]])
        else:
            text=["[CLS]"] + d[0]+["[SEP]"]
            #print('len(text)=', len(text))
            #print(text)
            tokens, pos, vm, tag = self.knowledge_graph.raw_with_vm_batch([text], max_length=cut_sen_len + 2)
        tokens = tokens[0]
        pos = pos[0]
        vm = vm[0].astype("bool")
        tag = tag[0]


        word_seq = [self.bert_vocab.get(t) for t in tokens]
        #new2ori = None
        tag_seq = []
        j = 0

        #print('len(d[1])=', len(d[1]))
        #print('len(pos)=', len(pos))
        #print('len(d[0])=', len(d[0]))
        #print(d[0])
        #print('len(tokens)=', len(tokens))
        #print(tokens)
        #print(word_seq)
        #print(d[1])
        #print('len(tag)=', len(tag))
        #print(tag)
        #token_tmp=""
        for i in range(len(word_seq)):
            if tag[i] == 0 and word_seq[i] != PAD_ID and word_seq[i] != CLS_ID and word_seq[i] != SEP_ID:
                #print('i={},j={},word_seq={},d[1][j]={}'.format(i,j,word_seq[i],d[1][j]))
                #token_tmp+=tokens[i]
                #if token_tmp==d[0][j]:
                if tokens[i] == d[0][j]:
                    tag_seq.append(self.tag2id[d[1][j]])
                    j += 1
                    #token_tmp=""
            elif tag[i] == 1 and word_seq[i] != PAD_ID and word_seq[i] != CLS_ID:  # 是添加的实体
                tag_seq.append(self.tag_id_ENT)
            elif word_seq[i] == CLS_ID:
                tag_seq.append(self.tag_id_CLS)
            elif word_seq[i] == SEP_ID:
                tag_seq.append(self.tag_id_SEP)
            else:#CLS SEP PAD
                tag_seq.append(self.tag_id_PAD)
        """
        if j!=len(d[0]) or  j!=len(d[1]):
            print(tokens)
            print(word_seq)
            print(tag_seq)
            print(d[0])
            print(d[1])
            assert False
        """

        d.append(pos)
        d.append(vm)
        #d.append(tag)
        #d.append(new2ori)
        d.append(word_seq)#-3
        #d.append(self.seq_tag2id(tag_seq))
        d.append(tag_seq)#-2
        d.append(self.seq_intent2id(d[2]))#-1
        #print("len:",len(d))
        #print(d)
        '''
        print('len(vm)=',len(vm))
        for i in range(0, len(vm)):
            # print(i)
            p=False
            for j in range(0, len(vm)):
                if vm[i][j]:
                    print(str(vm[i][j])+"("+str(i)+","+str(j) + ")   ", end='')
                    p = True
                    # print(" ",end='')
            if p:
                print('\n')
        '''
        return d


    '''
    def bert_tokenize(self, word_seq, tag_seq):
        split_tokens = []
        new_tag_seq = []
        new2ori = {}
        basic_tokens = self.tokenizer.basic_tokenizer.tokenize(' '.join(word_seq))
        accum = ''
        i, j = 0, 0
        for i, token in enumerate(basic_tokens):
            if (accum + token).lower() == word_seq[j].lower():
                accum = ''
            else:
                accum += token
            for sub_token in self.tokenizer.wordpiece_tokenizer.tokenize(basic_tokens[i]):
                new2ori[len(new_tag_seq)] = j
                split_tokens.append(sub_token)
                new_tag_seq.append(tag_seq[j])
            if accum == '':
                j += 1
        return split_tokens, new_tag_seq, new2ori
    '''
    def seq_tag2id(self, tags):
        return [self.tag2id[x] for x in tags if x in self.tag2id]

    def seq_id2tag(self, ids):
        return [self.id2tag[x] for x in ids]

    def seq_intent2id(self, intents):
        return [self.intent2id[x] for x in intents if x in self.intent2id]

    def seq_id2intent(self, ids):
        return [self.id2intent[x] for x in ids]
    def pad_batch(self, batch_data):
        batch_size = len(batch_data)
        max_seq_len = max([len(x[-3]) for x in batch_data])
        word_mask_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        word_seq_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        word_pos_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        word_vm_tensor = torch.zeros((batch_size, max_seq_len, max_seq_len), dtype=torch.long)
        tag_mask_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        tag_seq_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        intent_tensor = torch.zeros((batch_size, self.intent_dim), dtype=torch.float)
        context_max_seq_len = max([len(x[-6]) for x in batch_data])
        context_mask_tensor = torch.zeros((batch_size, context_max_seq_len), dtype=torch.long)
        context_seq_tensor = torch.zeros((batch_size, context_max_seq_len), dtype=torch.long)
        context_pos_tensor = torch.zeros((batch_size, context_max_seq_len), dtype=torch.long)
        context_vm_tensor = torch.zeros((batch_size, context_max_seq_len, context_max_seq_len), dtype=torch.long)
        for i in range(batch_size):
            indexed_tokens = batch_data[i][-3]
            tags = batch_data[i][-2]
            intents = batch_data[i][-1]
            sen_len = len(indexed_tokens)
            word_seq_tensor[i, :sen_len] = torch.LongTensor(indexed_tokens)
            tag_seq_tensor[i, :sen_len] = torch.LongTensor(tags)
            #word_mask_tensor[i, :sen_len] = torch.LongTensor([1] * sen_len)
            for j in range(sen_len):
                if indexed_tokens[j]!=PAD_ID:
                    word_mask_tensor[i, j] = torch.LongTensor([1])
            for j in range(len(tags)):
                if tags[j]<self.tag_dim:
                    tag_mask_tensor[i, j] = torch.LongTensor([1])
            word_pos_tensor[i, :sen_len] = torch.LongTensor([batch_data[i][-5]])
            word_vm_tensor[i, :sen_len, :sen_len] = torch.LongTensor([batch_data[i][-4]])
            for j in intents:
                intent_tensor[i, j] = 1.

            context_len = len(batch_data[i][-6])
            context_seq_tensor[i, :context_len] = torch.LongTensor(batch_data[i][-6])
            #context_mask_tensor[i, :context_len] = torch.LongTensor([1] * context_len)
            for j in range(context_len):
                if batch_data[i][-6][j]!=PAD_ID:
                    context_mask_tensor[i, j] = torch.LongTensor([1])
            context_pos_tensor[i, :context_len] = torch.LongTensor([batch_data[i][-8]])
            context_vm_tensor[i, :context_len, :context_len] = torch.LongTensor([batch_data[i][-7]])

        return word_seq_tensor, tag_seq_tensor, intent_tensor, word_mask_tensor, tag_mask_tensor, context_seq_tensor,\
               context_mask_tensor,word_pos_tensor, word_vm_tensor,context_pos_tensor, context_vm_tensor

    def get_train_batch(self, batch_size):
        batch_data = random.choices(self.data['train'], k=batch_size)
        return self.pad_batch(batch_data)

    def yield_batches(self, batch_size, data_key):
        batch_num = math.ceil(len(self.data[data_key]) / batch_size)
        for i in range(batch_num):
            batch_data = self.data[data_key][i * batch_size:(i + 1) * batch_size]
            yield self.pad_batch(batch_data), batch_data, len(batch_data)
