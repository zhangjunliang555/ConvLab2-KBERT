# coding: utf-8
"""
KnowledgeGraph
"""
import os
import brain.config as config
import pkuseg
import numpy as np


class KnowledgeGraph(object):
    """
    spo_files - list of Path of *.spo files, or default kg name. e.g., ['HowNet']
    """

    def __init__(self, spo_files, predicate=True):
        self.predicate = predicate
        self.spo_file_paths = [config.KGS.get(f, f) for f in spo_files]
        self.lookup_table = self._create_lookup_table()
        self.segment_vocab = list(self.lookup_table.keys()) + config.NEVER_SPLIT_TAG
        self.tokenizer = pkuseg.pkuseg(model_name="default", postag=False, user_dict=self.segment_vocab)
        self.special_tags = set(config.NEVER_SPLIT_TAG)

    def _create_lookup_table(self):
        lookup_table = {}
        for spo_path in self.spo_file_paths:
            print("[KnowledgeGraph] Loading spo from {}".format(spo_path))
            with open(spo_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        subj, pred, obje = line.strip().split("\t")    
                    except:
                        print("[KnowledgeGraph] Bad spo:", line)
                    if self.predicate:
                        value = pred + obje
                    else:
                        value = obje
                    if subj in lookup_table.keys():
                        lookup_table[subj].add(value)
                    else:
                        lookup_table[subj] = set([value])
        return lookup_table

    def add_knowledge_with_vm(self, sent_batch, max_entities=config.MAX_ENTITIES, max_length=128,raw_sent_batch=None):
        """
        input: sent_batch - list of sentences, e.g., ["abcd", "efgh"]
        return: know_sent_batch - list of sentences with entites embedding
                position_batch - list of position index of each character.
                visible_matrix_batch - list of visible matrixs
                seg_batch - list of segment tags
        """
        split_sent_batch=[]
        if raw_sent_batch:
            for n in range(len(sent_batch)):
                split_sent_batch.append(self.my_tokenizer_cut(sent_batch[n], raw_sent_batch[n]))
        else:
            split_sent_batch = [self.tokenizer.cut(sent) for sent in sent_batch]


        know_sent_batch = []#代表词和知识
        position_batch = []#代表词的相对位置和知识的相对位置,三层嵌入list,batch，sent,token
        visible_matrix_batch = []
        seg_batch = []#代表词类型，为0为原始输入，为1代表知识
        n=0
        for split_sent in split_sent_batch:

            # create tree
            sent_tree = []
            pos_idx_tree = []#pos_idx_tree包含词位置和词位置开始的知识位置
            abs_idx_tree = []
            pos_idx = -1
            abs_idx = -1
            abs_idx_src = []#所有输入词的决绝对位置

            k = 0
            tmp = ""
            for token in split_sent:

                entities = list(self.lookup_table.get(token, []))[:max_entities]
                sent_tree.append((token, entities))
                #print(f"[add_knowledge_with_vm] token:{token} ,entities:{entities}")
                token_pos_idx = []
                token_abs_idx = []
                if token in self.special_tags:
                    token_pos_idx = [pos_idx+1]
                    token_abs_idx = [abs_idx+1]
                    if raw_sent_batch:
                        if token==raw_sent_batch[n][k]:
                            k+=1
                else:
                    add_word = list(token)
                    i=0
                    if raw_sent_batch:
                        for sub_word in add_word:
                            tmp += sub_word
                            if tmp == raw_sent_batch[n][k]:
                                k += 1
                                i+=1
                                token_pos_idx +=[pos_idx +i]
                                token_abs_idx += [abs_idx + i]
                                tmp = ""
                    else:
                        token_pos_idx = [pos_idx+i for i in range(1, len(token)+1)]
                        token_abs_idx = [abs_idx+i for i in range(1, len(token)+1)]
                abs_idx = token_abs_idx[-1]

                entities_pos_idx = []
                entities_abs_idx = []
                for ent in entities:
                    ent_pos_idx = [token_pos_idx[-1] + i for i in range(1, len(ent)+1)]
                    entities_pos_idx.append(ent_pos_idx)
                    ent_abs_idx = [abs_idx + i for i in range(1, len(ent)+1)]
                    abs_idx = ent_abs_idx[-1]
                    entities_abs_idx.append(ent_abs_idx)

                pos_idx_tree.append((token_pos_idx, entities_pos_idx))
                pos_idx = token_pos_idx[-1]
                abs_idx_tree.append((token_abs_idx, entities_abs_idx))
                abs_idx_src += token_abs_idx

            # Get know_sent and pos
            #print("add_knowledge_with_vm,1,len(sent_tree):", len(sent_tree))
            #print(sent_tree)
            know_sent = []
            pos = []
            seg = []
            k=0
            tmp = ""
            for i in range(len(sent_tree)):
                word = sent_tree[i][0]
                if word in self.special_tags:
                    if raw_sent_batch:
                        #print("add_knowledge_with_vm,k:",k)
                        #print(raw_sent_batch[n][k])
                        if word==raw_sent_batch[n][k]:
                            k+=1
                    know_sent += [word]
                    seg += [0]
                else:
                    add_word = list(word)
                    #print("add_knowledge_with_vm,len(add_word):", len(add_word))
                    #print(add_word)
                    if raw_sent_batch:
                        for sub_word in add_word:
                            tmp +=sub_word
                            #print("add_knowledge_with_vm,k:", k)
                            #print(raw_sent_batch[n][k])
                            #print(tmp)
                            if tmp==raw_sent_batch[n][k]:
                                k+=1
                                know_sent += [tmp]
                                seg += [0]
                                tmp=""
                    else:
                        know_sent += add_word
                        seg += [0]*len(add_word)
                pos += pos_idx_tree[i][0]
                for j in range(len(sent_tree[i][1])):
                    add_word = list(sent_tree[i][1][j])
                    know_sent += add_word
                    seg += [1] * len(add_word)
                    pos += list(pos_idx_tree[i][1][j])

            #print("add_knowledge_with_vm,1,len(know_sent):", len(know_sent))
            #print(know_sent)
            token_num = len(know_sent)
            # Calculate visible matrix
            visible_matrix = np.zeros((token_num, token_num))
            for item in abs_idx_tree:
                src_ids = item[0]
                for id in src_ids:
                    #本输入词对所有输入词和本输入词下的知识可见
                    visible_abs_idx = abs_idx_src + [idx for ent in item[1] for idx in ent]
                    visible_matrix[id, visible_abs_idx] = 1
                for ent in item[1]:
                    for id in ent:
                        # 本输入词下的某一知识对其在的知识分支和本输入词可见
                        visible_abs_idx = ent + src_ids
                        visible_matrix[id, visible_abs_idx] = 1

            src_length = len(know_sent)
            if len(know_sent) < max_length:
                pad_num = max_length - src_length
                know_sent += [config.PAD_TOKEN] * pad_num
                seg += [0] * pad_num
                pos += [max_length - 1] * pad_num#每个词分配127个位置嵌入list
                visible_matrix = np.pad(visible_matrix, ((0, pad_num), (0, pad_num)), 'constant')  # pad 0
            else:
                # 如果知识导致总长度超过最大值，丢弃知识，原句输出
                if raw_sent_batch:
                    know_sent, pos, visible_matrix, seg = self.raw_with_vm(raw_sent_batch[n], max_length)
                else:
                    know_sent = know_sent[:max_length]
                    seg = seg[:max_length]
                    pos = pos[:max_length]#词为单位的位置嵌入list
                    visible_matrix = visible_matrix[:max_length, :max_length]
            #print("add_knowledge_with_vm,2,len(know_sent):",len(know_sent))
            #print(know_sent)
            '''
            print("know_sent len=", len(know_sent))
            print(know_sent)
            print("pos len=", len(pos))
            print(pos)
            print("visible_matrix len=", len(visible_matrix))
            print(visible_matrix)
            print("seg len=", len(seg))
            print(seg)
            '''
            know_sent_batch.append(know_sent)
            position_batch.append(pos)
            visible_matrix_batch.append(visible_matrix)
            seg_batch.append(seg)
            n+=1
        
        return know_sent_batch, position_batch, visible_matrix_batch, seg_batch

    def my_tokenizer_cut(self, sent,rawSent):
        '''
        分词不把原始语料单个词也拆掉,即原始语料词为最小分词单位。
        例如
        my_tokenizer_cut sent:
        [CLS]可以，电话是80##49##56##16。
        my_tokenizer_cut rawSent:
        ['[CLS]', '可', '以', '，', '电', '话', '是', '80', '##49', '##56', '##16', '。']
        my_tokenizer_cut cut_res:
        ['[CLS]', '可以', '，', '电话', '是', '80', '#', '#', '49', '#', '#', '56', '#', '#', '16', '。']
        my_tokenizer_cut res:
        ['[CLS]', '可以', '，', '电话', '是', '80', '##49', '##56', '##16', '。']
        :param sent:
        :return:
        '''
        #print("my_tokenizer_cut sent:")
        #print(sent)
        #print("my_tokenizer_cut rawSent:")
        #print(rawSent)
        cut_res=self.tokenizer.cut(sent)
        #print("my_tokenizer_cut cut_res:")
        #print(cut_res)

        res=[]
        tmp = ""
        tmp_cut = ""
        tmp_in=""
        i=0
        for cut in cut_res:
            #print("my_tokenizer_cut cut:", cut)
            #addlen = 0
            tmp_in = ""
            for cut_sub in list(cut):
                #if len(tmp_cut) > 0:
                    #addlen +=1
                if len(tmp) == 0:
                    if len(tmp_cut) > 0:
                        #print("my_tokenizer_cut, 1,res.append(tmp_cut):", tmp_cut)
                        res.append(tmp_cut)
                        tmp_cut = ""
                        #addlen += -1
                tmp += cut_sub
                '''
                print("my_tokenizer_cut cut_sub:",cut_sub)
                print("my_tokenizer_cut tmp:", tmp)
                print("my_tokenizer_cut tmp_cut:", tmp_cut)
                print("my_tokenizer_cut rawSent[i]:", rawSent[i])
                '''
                if rawSent[i] == tmp:
                    if len(tmp_cut)>0:
                        tmp_cut=tmp
                    else:
                        tmp_in +=tmp
                    tmp = ""
                    i+=1
            if len(tmp)==0:
                if len(tmp_cut)>0:
                    #print("my_tokenizer_cut, 2,res.append(tmp_cut):", tmp_cut)
                    res.append(tmp_cut)
                    tmp_cut = ""
                #else:
                    #if addlen>0:
                        #print("my_tokenizer_cut, 3,res.append(cut[addlen:]):", cut[addlen:])
                        #res.append(cut[addlen:])
                    #else:
                        #print("my_tokenizer_cut, 4,res.append(cut):", cut)
                        #res.append(cut)
            else:
                tmp_cut=tmp

            if len(tmp_in)>0:
                #print("my_tokenizer_cut, 5,res.append(tmp_in):", tmp_in)
                res.append(tmp_in)
                tmp_in = ""

        #print("my_tokenizer_cut res:")
        #print(res)
        return res

    def raw_with_vm(self, raw_sent, max_length=128):

        #print("raw_sent len=", len(raw_sent))
        #print(raw_sent)
        #if len(raw_sent)>max_length:
            #raw_sent=raw_sent[:max_length]
        # create tree
        know_sent = []
        pos = []
        seg = []

        sent_tree = []
        pos_idx_tree = []  # pos_idx_tree包含词位置和词位置开始的知识位置
        abs_idx_tree = []
        pos_idx = -1
        abs_idx = -1
        abs_idx_src = []  # 所有输入词的决绝对位置

        for token in raw_sent:
            entities =[]
            sent_tree.append((token, entities))
            token_pos_idx = [pos_idx + 1]
            token_abs_idx = [abs_idx + 1]
            abs_idx = token_abs_idx[-1]
            entities_pos_idx = []
            entities_abs_idx = []

            pos_idx_tree.append((token_pos_idx, entities_pos_idx))
            pos_idx = token_pos_idx[-1]
            abs_idx_tree.append((token_abs_idx, entities_abs_idx))
            abs_idx_src += token_abs_idx

        # Get know_sent and pos
        # print("add_knowledge_with_vm,1,len(sent_tree):", len(sent_tree))
        # print(sent_tree)
        for i in range(len(sent_tree)):
            word = sent_tree[i][0]
            know_sent += [word]
            seg += [0]
            pos += pos_idx_tree[i][0]

        # print("add_knowledge_with_vm,1,len(know_sent):", len(know_sent))
        # print(know_sent)
        token_num = len(know_sent)
        # Calculate visible matrix
        visible_matrix = np.zeros((token_num, token_num))
        for item in abs_idx_tree:
            src_ids = item[0]
            for id in src_ids:
                # 本输入词对所有输入词和本输入词下的知识可见
                visible_abs_idx = abs_idx_src
                visible_matrix[id, visible_abs_idx] = 1

        src_length = len(know_sent)
        if len(know_sent) < max_length:
            pad_num = max_length - src_length
            know_sent += [config.PAD_TOKEN] * pad_num
            seg += [0] * pad_num
            pos += [max_length - 1] * pad_num  # 每个词分配127个位置嵌入list
            visible_matrix = np.pad(visible_matrix, ((0, pad_num), (0, pad_num)), 'constant')  # pad 0
        else:
            know_sent = know_sent[:max_length]
            seg = seg[:max_length]
            pos = pos[:max_length]  # 词为单位的位置嵌入list
            visible_matrix = visible_matrix[:max_length, :max_length]
        '''
        print("know_sent len=",len(know_sent))
        print(know_sent)
        print("pos len=", len(pos))
        print(pos)
        print("visible_matrix len=", len(visible_matrix))
        print(visible_matrix)
        print("seg len=", len(seg))
        print(seg)
        '''
        return know_sent, pos, visible_matrix, seg

    def raw_with_vm_batch(self, raw_sent_batch, max_length=128):
        """
        input: sent_batch - list of sentences, e.g., ["abcd", "efgh"]
        return: know_sent_batch - list of sentences with entites embedding
                position_batch - list of position index of each character.
                visible_matrix_batch - list of visible matrixs
                seg_batch - list of segment tags
        """

        know_sent_batch = []  # 代表词和知识
        position_batch = []  # 代表词的相对位置和知识的相对位置,三层嵌入list,batch，sent,token
        visible_matrix_batch = []
        seg_batch = []  # 代表词类型，为0为原始输入，为1代表知识
        for split_sent in raw_sent_batch:
             know_sent, pos, visible_matrix, seg = self.raw_with_vm(split_sent, max_length)
             know_sent_batch.append(know_sent)
             position_batch.append(pos)
             visible_matrix_batch.append(visible_matrix)
             seg_batch.append(seg)

        return know_sent_batch, position_batch, visible_matrix_batch, seg_batch