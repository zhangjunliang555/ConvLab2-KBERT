import re
import torch


def is_slot_da(da):
    tag_da = {'Inform', 'Recommend'}
    not_tag_slot = '酒店设施'
    if da[0] in tag_da and not_tag_slot not in da[2]:
        return True
    return False


def calculateF1(predict_golden):
    TP, FP, FN = 0, 0, 0
    for item in predict_golden:
        predicts = item['predict']
        labels = item['golden']
        for ele in predicts:
            if ele in labels:
                TP += 1
            else:
                FP += 1
        for ele in labels:
            if ele not in predicts:
                FN += 1
    # print(TP, FP, FN)
    precision = 1.0 * TP / (TP + FP) if TP + FP else 0.
    recall = 1.0 * TP / (TP + FN) if TP + FN else 0.
    F1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.
    return precision, recall, F1


def tag2das(word_seq, tag_seq):
    #assert len(word_seq)==len(tag_seq)
    assert len(word_seq) >= len(tag_seq)
    #print("[tag2das] len(word_seq):", len(word_seq))
    #print(word_seq)
    #print("[tag2das] len(tag_seq):", len(tag_seq))
    #print(tag_seq)

    das = []
    i = 0
    while i < len(tag_seq):
        tag = tag_seq[i]
        if tag.startswith('B'):
            intent, domain, slot = tag[2:].split('+')
            value = word_seq[i]
            j = i + 1
            while j < len(tag_seq):
                if tag_seq[j].startswith('I') and tag_seq[j][2:] == tag[2:]:
                    # tag_seq[j][2:].split('+')[-1]==slot or tag_seq[j][2:] == tag[2:]
                    if word_seq[j].startswith('##'):
                        value += word_seq[j][2:]
                    else:
                        value += word_seq[j]
                    i += 1
                    j += 1
                else:
                    break
            das.append([intent, domain, slot, value])
        i += 1
    return das


def intent2das(intent_seq):
    triples = []
    for intent in intent_seq:
        intent, domain, slot, value = re.split('\+', intent)
        triples.append([intent, domain, slot, value])
    return triples


def recover_intent(dataloader, intent_logits, tag_logits, tag_mask_tensor, ori_word_seq, new2ori,intent_top=3,intent_score=0,intent_not_None=True):
    # tag_logits = [sequence_length, tag_dim]
    # intent_logits = [intent_dim]
    # tag_mask_tensor = [sequence_length]
    #'''
    #print("recover_intent len(intent_logits):", len(intent_logits))
    #print(intent_logits)
    #print("[recover_intent], len(tag_logits):", len(tag_logits))
    #print(tag_logits)
    #print("[recover_intent], len(tag_mask_tensor):", len(tag_mask_tensor))
    #print(tag_mask_tensor)
    #print("[recover_intent], max_seq_len:", tag_logits.size(0))
    #'''
    max_seq_len = tag_logits.size(0)
    das = []
    '''
    for j in range(dataloader.intent_dim):
        if intent_logits[j] > 0 :
            intent, domain, slot, value = re.split('\+', dataloader.id2intent[j])
            das.append([intent, domain, slot, value])
    '''
    #'''
    y_sort = intent_logits.argsort()
    start=-1
    stop=-(intent_top+1)
    step=-1
    #print(y_sort)
    #print(f"start={start},stop={stop},step={step},intent_top={intent_top},intent_score={intent_score}")
    for i in range(start,stop,step):
        j=int(y_sort[i])
        #print(f"i:{i},j:{j},intent_logits:{intent_logits[j]}")
        if intent_logits[j] > intent_score:
            intent, domain, slot, value = re.split('\+', dataloader.id2intent[j])
            das.append([intent, domain, slot, value])
    if len(das)<1 and intent_not_None:
        intent, domain, slot, value = re.split('\+', dataloader.id2intent[int(y_sort[-1])])
        das.append([intent, domain, slot, value])
    #'''

    tags = []
    for j in range(1 , max_seq_len):
        if tag_mask_tensor[j] == 1:
            #print(f"[recover_intent], j={j},len={len(tag_logits[j])}")
            #print(tag_logits[j])
            value, tag_id = torch.max(tag_logits[j], dim=-1)
            tags.append(dataloader.id2tag[tag_id.item()])

    '''
    print("recover_intent len(ori_word_seq):",len(ori_word_seq))
    print(ori_word_seq)
    print("recover_intent len(tags):", len(tags))
    print(tags)
    '''
    tag_intent = tag2das(ori_word_seq, tags)
    das += tag_intent
    return das
