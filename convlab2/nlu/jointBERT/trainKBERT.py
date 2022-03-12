import argparse
import os
import sys
import json
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np
import zipfile
import torch
import codecs
import math
from transformers import AdamW, get_linear_schedule_with_warmup

#print('trainKBERT.py path:', os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(root_dir)
#print('trainKBERT.py root_dir:', root_dir)
main_dir=os.path.dirname(os.path.abspath(__file__))

from convlab2.nlu.jointBERT.dataloaderKBERT import Dataloader
from convlab2.nlu.jointBERT.jointKBERT import JointBERT

from convlab2.nlu.jointBERT.brain.knowgraph import KnowledgeGraph
from convlab2.nlu.jointBERT.uer.utils.vocab import Vocab

from convlab2.nlu.jointBERT.crosswoz.postprocess import is_slot_da, calculateF1, recover_intent




def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


parser = argparse.ArgumentParser(description="Train a model.")
#parser.add_argument('--config_path_app',default=main_dir+"/crosswoz/configs/crosswoz_all_context_kg.json",help='path to config file')
parser.add_argument('--config_path_app',default=main_dir+"/crosswoz/configs/crosswoz_all_context.json",help='path to config file')
#parser.add_argument('--config_path_app',default=main_dir+"/crosswoz/configs/crosswoz_usr_context_kg.json",help='path to config file')
#parser.add_argument('--config_path_app',default=main_dir+"/crosswoz/configs/crosswoz_usr_context.json",help='path to config file')
# Path options.
parser.add_argument("--pretrained_model_path", default="/app/PM/bertPM/chinese_roberta_wwm_ext_L-12_H-768_A-12_pytorh/model.bin", type=str,help="Path of the pretrained model.")
#parser.add_argument("--pretrained_model_path", default="/app/PM/bertPM/chinese-bert-wwm-ext_12_pytorch/pytorch_model.bin", type=str,help="Path of the pretrained model.")
#parser.add_argument("--output_model_path", default="./crosswoz/output/all_context/jointKBERT_model.bin", type=str,
#                    help="Path of the output model.")
parser.add_argument("--vocab_path", default="/app/PM/bertPM/chinese_roberta_wwm_ext_L-12_H-768_A-12_pytorh/vocab.txt", type=str,help="Path of the vocabulary file.")
#parser.add_argument("--vocab_path", default="/app/PM/bertPM/chinese-bert-wwm-ext_12_pytorch/vocab.txt", type=str,help="Path of the vocabulary file.")
#parser.add_argument("--train_path", type=str, required=True,help="Path of the trainset.")
#parser.add_argument("--dev_path", type=str, required=True,help="Path of the devset.")
#parser.add_argument("--test_path", type=str, required=True,help="Path of the testset.")
parser.add_argument("--config_path", default="/app/PM/bertPM/chinese_roberta_wwm_ext_L-12_H-768_A-12_pytorh/config.json", type=str,help="Path of the config file.")
#parser.add_argument("--config_path", default="/app/PM/bertPM/chinese-bert-wwm-ext_12_pytorch/config.json", type=str,help="Path of the config file.")

# Model options.
#parser.add_argument("--batch_size", type=int, default=32,help="Batch size.")
parser.add_argument("--seq_length", type=int, default=256,
                    help="Sequence length.")
parser.add_argument("--encoder", choices=["bert", "lstm", "gru", \
                                          "cnn", "gatedcnn", "attn", \
                                          "rcnn", "crnn", "gpt", "bilstm"], \
                    default="bert", help="Encoder type.")
parser.add_argument("--target", default="bert", help="target type.")
parser.add_argument("--bidirectional", action="store_true", help="Specific to recurrent model.")
parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                    help="Pooling type.")

# Subword options.
parser.add_argument("--subword_type", choices=["none", "char"], default="none",
                    help="Subword feature type.")
'''
parser.add_argument("--sub_vocab_path", type=str, default="premodel/sub_vocab.txt",
                    help="Path of the subword vocabulary file.")
parser.add_argument("--subencoder", choices=["avg", "lstm", "gru", "cnn"], default="avg",
                    help="Subencoder type.")
parser.add_argument("--sub_layers_num", type=int, default=2, help="The number of subencoder layers.")
'''
'''
# Tokenizer options.
parser.add_argument("--tokenizer", choices=["bert", "char", "word", "space"], default="bert",
                    help="Specify the tokenizer."
                         "Original Google BERT uses bert tokenizer on Chinese corpus."
                         "Char tokenizer segments sentences into characters."
                         "Word tokenizer supports online word segmentation based on jieba segmentor."
                         "Space tokenizer segments sentences into words according to space."
                    )
'''
# Optimizer options.
#parser.add_argument("--learning_rate", type=float, default=2e-5,help="Learning rate.")
#parser.add_argument("--warmup", type=float, default=0.1,help="Warm up value.")

# Training options.
parser.add_argument("--dropout", type=float, default=0.5,help="Dropout.")
parser.add_argument("--epochs_num", type=int, default=10,help="Number of epochs.")
#parser.add_argument("--report_steps", type=int, default=100,help="Specific steps to print prompt.")
parser.add_argument("--seed", type=int, default=7,help="Random seed.")

# Evaluation options.
#parser.add_argument("--mean_reciprocal_rank", action="store_true", help="Evaluation metrics for DBQA dataset.")

# kg
parser.add_argument("--kg_name", default=main_dir+"/brain/kgs/CnDbpedia.spo",help="KG name or path")
#parser.add_argument("--kg_name", default="none",help="KG name or path")
#parser.add_argument("--workers_num", type=int, default=1, help="number of process for loading dataset")
parser.add_argument("--no_vm", action="store_true", help="Disable the visible_matrix")

#config
parser.add_argument("--context", type=bool, default=False, help="")
parser.add_argument("--finetune", type=bool, default=False, help="")
parser.add_argument("--context_grad", type=bool, default=False, help="")
parser.add_argument("--app_hidden_units", type=int, default=768, help="")


if __name__ == '__main__':
    args = parser.parse_args()
    config = json.load(open(args.config_path_app))
    data_dir = main_dir+"/"+config['data_dir']
    output_dir = main_dir+"/"+config['output_dir']
    log_dir = main_dir+"/"+config['log_dir']
    DEVICE = config['DEVICE']
    modelName="pytorch_model.bin"

    with codecs.open(args.config_path, "r", "utf-8") as f:
        param = json.load(f)
    args.emb_size = param.get("emb_size", 768)
    args.hidden_size = param.get("hidden_size", 768)
    args.kernel_size = param.get("kernel_size", 3)
    args.block_size = param.get("block_size", 2)
    args.feedforward_size = param.get("intermediate_size", 3072)
    args.heads_num = param.get("num_attention_heads", 12)
    args.layers_num = param.get("num_hidden_layers", 12)
    #args.dropout = param.get("dropout", 0.1)

    args.dropout = config['model']['dropout']
    args.context = config['model']['context']
    args.finetune = config['model']['finetune']
    args.context_grad = config['model']['context_grad']
    args.app_hidden_units = config['model']['hidden_units']
    args.seed = config['seed']

    set_seed(args.seed)
    '''
    if 'multiwoz' in data_dir:
        print('-'*20 + 'dataset:multiwoz' + '-'*20)
        from convlab2.nlu.jointBERT.multiwoz.postprocess import is_slot_da, calculateF1, recover_intent
    elif 'camrest' in data_dir:
        print('-' * 20 + 'dataset:camrest' + '-' * 20)
        from convlab2.nlu.jointBERT.camrest.postprocess import is_slot_da, calculateF1, recover_intent
    elif 'crosswoz' in data_dir:
        print('-' * 20 + 'dataset:crosswoz' + '-' * 20)
        from convlab2.nlu.jointBERT.crosswoz.postprocess import is_slot_da, calculateF1, recover_intent
    '''

    # Load vocabulary.
    vocab = Vocab()
    vocab.load(args.vocab_path)
    args.vocab = vocab

    # Build knowledge graph.
    args.kg_name = config['model']['kg_name']
    if args.kg_name == 'none':
        spo_files = []
        #args.no_vm = True
        kg = KnowledgeGraph(spo_files=spo_files, predicate=False)
        modelName = "pytorch_model_context.bin"
    else:
        #args.no_vm=False
        spo_files = [args.kg_name]
        kg = KnowledgeGraph(spo_files=spo_files, predicate=False)

    args.no_vm = False
    print(f'output_dir:{output_dir},modelName:{modelName}')
    print('args.kg_name:', args.kg_name)
    print('kg:', kg)
    print('args.no_vm:', args.no_vm)
    intent_vocab = json.load(open(os.path.join(data_dir, 'intent_vocab.json')))
    tag_vocab = json.load(open(os.path.join(data_dir, 'tag_vocab.json')))
    dataloader = Dataloader(intent_vocab=intent_vocab, tag_vocab=tag_vocab,bert_vocab=vocab,knowledge_graph=kg)
    print('intent num:', len(intent_vocab))
    print('tag num:', len(tag_vocab))

    #dataloader.load_data(json.load(open(os.path.join(data_dir, 'val_data.json'))), "val",cut_sen_len=config['cut_sen_len'])
    #assert False

    for data_key in ['train', 'val', 'test']:
        dataloader.load_data(json.load(open(os.path.join(data_dir, '{}_data.json'.format(data_key)))), data_key,cut_sen_len=config['cut_sen_len'])
        print('{} set size: {}'.format(data_key, len(dataloader.data[data_key])))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    writer = SummaryWriter(log_dir)

    model = JointBERT(args, DEVICE, dataloader.tag_dim, dataloader.intent_dim, dataloader.intent_weight)
    model.to(DEVICE)

    args.epochs_num = config['model']['epochs_num']
    check_step = config['model']['check_step']
    batch_size = config['model']['batch_size']
    max_step = math.ceil(len(dataloader.data['train']) / batch_size) * args.epochs_num
    print('max_step:{} batch_size: {} epochs_num:{}'.format(max_step, batch_size, args.epochs_num))

    if config['model']['finetune']:
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if
                        not any(nd in n for nd in no_decay) and p.requires_grad],
             'weight_decay': config['model']['weight_decay']},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=config['model']['learning_rate'],
                          eps=config['model']['adam_epsilon'])
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config['model']['warmup_steps'],
                                                    num_training_steps=max_step)
    else:
        for n, p in model.named_parameters():
            if 'bert' in n:
                p.requires_grad = False
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=config['model']['learning_rate'])

    for name, param in model.named_parameters():
        print(name, param.shape, param.device, param.requires_grad)

    model.zero_grad()
    train_slot_loss, train_intent_loss = 0, 0
    best_val_f1 = 0.

    writer.add_text('config', json.dumps(config))

    for step in range(1, max_step + 1):
        model.train()
        batched_data = dataloader.get_train_batch(batch_size)
        batched_data = tuple(t.to(DEVICE) for t in batched_data)
        word_seq_tensor, tag_seq_tensor, intent_tensor, word_mask_tensor, tag_mask_tensor, context_seq_tensor, \
        context_mask_tensor,word_pos_tensor, word_vm_tensor,context_pos_tensor, context_vm_tensor = batched_data
        if not config['model']['context']:
            context_seq_tensor, context_mask_tensor,context_pos_tensor, context_vm_tensor = None, None,None, None

        _, _, slot_loss, intent_loss = model.forward(word_seq_tensor, word_mask_tensor, tag_seq_tensor, tag_mask_tensor, \
                                                     intent_tensor, context_seq_tensor, context_mask_tensor,word_pos_tensor, \
                                                     word_vm_tensor,context_pos_tensor, context_vm_tensor)
        train_slot_loss += slot_loss.item()
        train_intent_loss += intent_loss.item()
        loss = slot_loss + intent_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if config['model']['finetune']:
            scheduler.step()  # Update learning rate schedule
        model.zero_grad()
        if step % check_step == 0:
            train_slot_loss = train_slot_loss / check_step
            train_intent_loss = train_intent_loss / check_step
            print('[%d|%d] step' % (step, max_step))
            print('\t slot loss:', train_slot_loss)
            print('\t intent loss:', train_intent_loss)

            predict_golden = {'intent': [], 'slot': [], 'overall': []}

            val_slot_loss, val_intent_loss = 0, 0
            model.eval()
            for pad_batch, ori_batch, real_batch_size in dataloader.yield_batches(batch_size, data_key='val'):
                pad_batch = tuple(t.to(DEVICE) for t in pad_batch)
                word_seq_tensor, tag_seq_tensor, intent_tensor, word_mask_tensor, tag_mask_tensor, context_seq_tensor, context_mask_tensor,\
                word_pos_tensor, word_vm_tensor,context_pos_tensor, context_vm_tensor = pad_batch
                if not config['model']['context']:
                    context_seq_tensor, context_mask_tensor,context_pos_tensor, context_vm_tensor = None, None,None, None

                with torch.no_grad():
                    slot_logits, intent_logits, slot_loss, intent_loss = model.forward(word_seq_tensor,
                                                                                       word_mask_tensor,
                                                                                       tag_seq_tensor,
                                                                                       tag_mask_tensor,
                                                                                       intent_tensor,
                                                                                       context_seq_tensor,
                                                                                       context_mask_tensor,word_pos_tensor,
                                                                                       word_vm_tensor,context_pos_tensor, context_vm_tensor)
                val_slot_loss += slot_loss.item() * real_batch_size
                val_intent_loss += intent_loss.item() * real_batch_size
                for j in range(real_batch_size):
                    predicts = recover_intent(dataloader, intent_logits[j], slot_logits[j], tag_mask_tensor[j],
                                              ori_batch[j][0], None)
                    labels = ori_batch[j][3]

                    predict_golden['overall'].append({
                        'predict': predicts,
                        'golden': labels
                    })
                    predict_golden['slot'].append({
                        'predict': [x for x in predicts if is_slot_da(x)],
                        'golden': [x for x in labels if is_slot_da(x)]
                    })
                    predict_golden['intent'].append({
                        'predict': [x for x in predicts if not is_slot_da(x)],
                        'golden': [x for x in labels if not is_slot_da(x)]
                    })

            for j in range(10):
                writer.add_text('val_sample_{}'.format(j),
                                json.dumps(predict_golden['overall'][j], indent=2, ensure_ascii=False),
                                global_step=step)

            total = len(dataloader.data['val'])
            val_slot_loss /= total
            val_intent_loss /= total
            print('%d samples val' % total)
            print('\t slot loss:', val_slot_loss)
            print('\t intent loss:', val_intent_loss)

            writer.add_scalar('intent_loss/train', train_intent_loss, global_step=step)
            writer.add_scalar('intent_loss/val', val_intent_loss, global_step=step)

            writer.add_scalar('slot_loss/train', train_slot_loss, global_step=step)
            writer.add_scalar('slot_loss/val', val_slot_loss, global_step=step)

            for x in ['intent', 'slot', 'overall']:
                precision, recall, F1 = calculateF1(predict_golden[x])
                print('-' * 20 + x + '-' * 20)
                print('\t Precision: %.2f' % (100 * precision))
                print('\t Recall: %.2f' % (100 * recall))
                print('\t F1: %.2f' % (100 * F1))

                writer.add_scalar('val_{}/precision'.format(x), precision, global_step=step)
                writer.add_scalar('val_{}/recall'.format(x), recall, global_step=step)
                writer.add_scalar('val_{}/F1'.format(x), F1, global_step=step)

            if F1 > best_val_f1+0.001:
                #print('best val F1 %.4f' % best_val_f1)
                print(f'best val F1: old_F1={best_val_f1},new_F1={F1}')
                best_val_f1 = F1
                torch.save(model.state_dict(), os.path.join(output_dir, modelName))
                print('save on', output_dir)

            train_slot_loss, train_intent_loss = 0, 0

    writer.add_text('val overall F1', '%.2f' % (100 * best_val_f1))
    writer.close()

    #model_path = os.path.join(output_dir, 'pytorch_model.bin')
    #zip_path = main_dir+"/"+config['zipped_model_path']
    #print('zip model to', zip_path)

    #with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        #zf.write(model_path)
