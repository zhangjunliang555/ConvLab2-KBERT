import argparse
import os
import json
import random
import numpy as np
import torch
import sys
import codecs

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(root_dir)
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


parser = argparse.ArgumentParser(description="Test a model.")
parser.add_argument('--config_path_app',default=main_dir+"/crosswoz/configs/crosswoz_all_context_kg.json",help='path to config file')
#parser.add_argument('--config_path_app',default=main_dir+"/crosswoz/configs/crosswoz_all_context.json",help='path to config file')
# Path options.
#parser.add_argument("--pretrained_model_path", default="/app/PM/bertPM/chinese_roberta_wwm_ext_L-12_H-768_A-12_pytorh/model.bin", type=str,help="Path of the pretrained model.")
parser.add_argument("--pretrained_model_path", default="/app/docker_app/ConvLab-2-master_KBERT/convlab2/nlu/jointBERT/crosswoz/output/all_context/pytorch_model.bin", type=str,help="Path of the pretrained model.")
#parser.add_argument("--pretrained_model_path", default="/app/docker_app/ConvLab-2-master_KBERT/convlab2/nlu/jointBERT/crosswoz/output/all_context/pytorch_model_context.bin", type=str,help="Path of the pretrained model.")
#parser.add_argument("--output_model_path", default="./crosswoz/output/all_context/jointKBERT_model.bin", type=str,
#                    help="Path of the output model.")
parser.add_argument("--vocab_path", default="/app/PM/bertPM/chinese_roberta_wwm_ext_L-12_H-768_A-12_pytorh/vocab.txt", type=str,
                    help="Path of the vocabulary file.")
#parser.add_argument("--train_path", type=str, required=True,help="Path of the trainset.")
#parser.add_argument("--dev_path", type=str, required=True,help="Path of the devset.")
#parser.add_argument("--test_path", type=str, required=True,help="Path of the testset.")
parser.add_argument("--config_path", default="/app/PM/bertPM/chinese_roberta_wwm_ext_L-12_H-768_A-12_pytorh/config.json", type=str,
                    help="Path of the config file.")

# Model options.
#parser.add_argument("--batch_size", type=int, default=32,help="Batch size.")
#parser.add_argument("--seq_length", type=int, default=256,help="Sequence length.")
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
#parser.add_argument("--epochs_num", type=int, default=20,help="Number of epochs.")
#parser.add_argument("--report_steps", type=int, default=100,help="Specific steps to print prompt.")
parser.add_argument("--seed", type=int, default=7,help="Random seed.")

# Evaluation options.
#parser.add_argument("--mean_reciprocal_rank", action="store_true", help="Evaluation metrics for DBQA dataset.")

# kg
parser.add_argument("--kg_name", default=main_dir+"/brain/kgs/CnDbpedia.spo",help="KG name or path")
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
    data_dir = main_dir + "/" + config['data_dir']
    output_dir = main_dir + "/" + config['output_dir']
    #log_dir = main_dir + "/" + config['log_dir']
    DEVICE = config['DEVICE']

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
    #args.finetune = config['model']['finetune']
    args.finetune =False
    #args.context_grad = config['model']['context_grad']
    args.context_grad =False
    args.app_hidden_units = config['model']['hidden_units']
    args.seed = config['seed']

    set_seed(config['seed'])

    # Load vocabulary.
    vocab = Vocab()
    vocab.load(args.vocab_path)
    args.vocab = vocab

    # Build knowledge graph.
    if args.kg_name == 'none':
        spo_files = []
        #args.no_vm = True
        kg = KnowledgeGraph(spo_files=spo_files, predicate=False)
    else:
        #args.no_vm=False
        spo_files = [args.kg_name]
        kg = KnowledgeGraph(spo_files=spo_files, predicate=False)

    args.no_vm = False
    print('args.kg_name:', args.kg_name)
    print('kg:', kg)
    print('args.no_vm:', args.no_vm)

    intent_vocab = json.load(open(os.path.join(data_dir, 'intent_vocab.json')))
    tag_vocab = json.load(open(os.path.join(data_dir, 'tag_vocab.json')))
    dataloader = Dataloader(intent_vocab=intent_vocab, tag_vocab=tag_vocab, bert_vocab=vocab, knowledge_graph=kg)
    print('intent num:', len(intent_vocab))
    print('tag num:', len(tag_vocab))
    for data_key in ['val', 'test']:
        dataloader.load_data(json.load(open(os.path.join(data_dir, '{}_data.json'.format(data_key)))), data_key,cut_sen_len=config['cut_sen_len'])
        print('{} set size: {}'.format(data_key, len(dataloader.data[data_key])))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    #if not os.path.exists(log_dir):
        #os.makedirs(log_dir)

    model = JointBERT(args, DEVICE, dataloader.tag_dim, dataloader.intent_dim)
    model.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model.bin'), DEVICE))
    model.to(DEVICE)
    model.eval()

    batch_size = config['model']['batch_size']

    data_key = 'test'
    predict_golden = {'intent': [], 'slot': [], 'overall': []}
    slot_loss, intent_loss = 0, 0
    for pad_batch, ori_batch, real_batch_size in dataloader.yield_batches(batch_size, data_key=data_key):
        pad_batch = tuple(t.to(DEVICE) for t in pad_batch)
        word_seq_tensor, tag_seq_tensor, intent_tensor, word_mask_tensor, tag_mask_tensor, context_seq_tensor, context_mask_tensor, \
        word_pos_tensor, word_vm_tensor, context_pos_tensor, context_vm_tensor = pad_batch
        if not config['model']['context']:
            context_seq_tensor, context_mask_tensor, context_pos_tensor, context_vm_tensor = None, None, None, None

        with torch.no_grad():
            slot_logits, intent_logits, slot_loss, intent_loss = model.forward(word_seq_tensor, word_mask_tensor, \
                                                                               tag_seq_tensor, tag_mask_tensor, \
                                                                               intent_tensor, \
                                                                               context_seq_tensor, context_mask_tensor, word_pos_tensor, \
                                                                               word_vm_tensor, context_pos_tensor, context_vm_tensor)
        slot_loss += slot_loss.item() * real_batch_size
        intent_loss += intent_loss.item() * real_batch_size
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
        print('[%d|%d] samples' % (len(predict_golden['overall']), len(dataloader.data[data_key])))

    total = len(dataloader.data[data_key])
    slot_loss /= total
    intent_loss /= total
    print('%d samples %s' % (total, data_key))
    print('\t slot loss:', slot_loss)
    print('\t intent loss:', intent_loss)

    for x in ['intent', 'slot', 'overall']:
        precision, recall, F1 = calculateF1(predict_golden[x])
        print('-' * 20 + x + '-' * 20)
        print('\t Precision: %.2f' % (100 * precision))
        print('\t Recall: %.2f' % (100 * recall))
        print('\t F1: %.2f' % (100 * F1))

    output_file = os.path.join(output_dir, 'output.json')
    json.dump(predict_golden['overall'], open(output_file, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
