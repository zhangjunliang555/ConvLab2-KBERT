import os
#import zipfile
import json
import torch
import argparse
import codecs
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(root_dir)
train_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(train_dir)

#from convlab2.util.file_util import cached_path
from convlab2.nlu.nlu import NLU
from convlab2.nlu.jointBERT.dataloaderKBERT import Dataloader
from convlab2.nlu.jointBERT.jointKBERT import JointBERT
from convlab2.nlu.jointBERT.crosswoz.postprocess import recover_intent
from convlab2.nlu.jointBERT.crosswoz.preprocess import preprocess
from convlab2.nlu.jointBERT.brain.knowgraph import KnowledgeGraph
from convlab2.nlu.jointBERT.uer.utils.vocab import Vocab

parser = argparse.ArgumentParser(description="Test a model.")
#parser.add_argument('--config_path_app',default=main_dir+"/crosswoz/configs/crosswoz_all_context.json",help='path to config file')
# Path options.
#parser.add_argument("--pretrained_model_path", default="/app/PM/bertPM/chinese_roberta_wwm_ext_L-12_H-768_A-12_pytorh/model.bin", type=str,help="Path of the pretrained model.")
#parser.add_argument("--pretrained_model_path", default="/app/docker_app/ConvLab-2-master_KBERT/convlab2/nlu/jointBERT/crosswoz/output/all_context/pytorch_model.bin", type=str,help="Path of the pretrained model.")
#parser.add_argument("--pretrained_model_path", default="/app/docker_app/ConvLab-2-master_KBERT/convlab2/nlu/jointBERT/crosswoz/output/all_context/pytorch_model_context.bin", type=str,help="Path of the pretrained model.")
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
#parser.add_argument("--kg_name", default="none",help="KG name or path")
parser.add_argument("--kg_name", default="/app/docker_app/ConvLab-2-master_KBERT/convlab2/nlu/jointBERT/brain/kgs/CnDbpedia.spo",help="KG name or path")
#parser.add_argument("--workers_num", type=int, default=1, help="number of process for loading dataset")
parser.add_argument("--no_vm", action="store_true", help="Disable the visible_matrix")

#config
parser.add_argument("--context", type=bool, default=False, help="")
parser.add_argument("--finetune", type=bool, default=False, help="")
parser.add_argument("--context_grad", type=bool, default=False, help="")
parser.add_argument("--app_hidden_units", type=int, default=768, help="")

class BERTNLU(NLU):
    def __init__(self, mode='all', config_file='crosswoz_all_context_kg.json',model_file='pytorch_model.bin'):
        assert mode == 'usr' or mode == 'sys' or mode == 'all'
        config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs/{}'.format(config_file))
        config = json.load(open(config_file))
        # DEVICE = config['DEVICE']
        DEVICE = 'cpu' if not torch.cuda.is_available() else config['DEVICE']
        print('DEVICE :', DEVICE)
        #DEVICE = 'cpu'
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(root_dir, config['data_dir'])
        output_dir = os.path.join(root_dir, config['output_dir'])

        args = parser.parse_args()
        with codecs.open(args.config_path, "r", "utf-8") as f:
            param = json.load(f)
        args.emb_size = param.get("emb_size", 768)
        args.hidden_size = param.get("hidden_size", 768)
        args.kernel_size = param.get("kernel_size", 3)
        args.block_size = param.get("block_size", 2)
        args.feedforward_size = param.get("intermediate_size", 3072)
        args.heads_num = param.get("num_attention_heads", 12)
        args.layers_num = param.get("num_hidden_layers", 12)
        # args.dropout = param.get("dropout", 0.1)

        args.dropout = config['model']['dropout']
        args.context = config['model']['context']
        # args.finetune = config['model']['finetune']
        args.finetune = False
        # args.context_grad = config['model']['context_grad']
        args.context_grad = False
        args.app_hidden_units = config['model']['hidden_units']
        args.seed = config['seed']

        # Load vocabulary.
        vocab = Vocab()
        vocab.load(args.vocab_path)
        args.vocab = vocab

        print('model_file :', model_file)
        # Build knowledge graph.
        if model_file!="pytorch_model.bin":
            args.kg_name = 'none'
        if args.kg_name == 'none':
            spo_files = []
            #args.no_vm = True
            kg = KnowledgeGraph(spo_files=spo_files, predicate=False)
        else:
            #args.no_vm = False
            spo_files = [args.kg_name]
            kg = KnowledgeGraph(spo_files=spo_files, predicate=False)

        print('args.kg_name:', args.kg_name)
        print('kg:', kg)
        args.no_vm = False
        #args.no_vm = True
        print('args.no_vm:', args.no_vm)

        if not os.path.exists(os.path.join(data_dir, 'intent_vocab.json')):
            preprocess(mode)

        intent_vocab = json.load(open(os.path.join(data_dir, 'intent_vocab.json')))
        tag_vocab = json.load(open(os.path.join(data_dir, 'tag_vocab.json')))
        dataloader = Dataloader(intent_vocab=intent_vocab, tag_vocab=tag_vocab, bert_vocab=vocab, knowledge_graph=kg,args=args)

        print('intent num:', len(intent_vocab))
        print('tag num:', len(tag_vocab))

        #best_model_path = os.path.join(output_dir, 'pytorch_model.bin')
        '''
        if not os.path.exists(best_model_path):
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            print('Load from model_file param')
            archive_file = cached_path(model_file)
            archive = zipfile.ZipFile(archive_file, 'r')
            archive.extractall(root_dir)
            archive.close()
        '''
        #print('Load from', best_model_path)
        args.pretrained_model_path=os.path.join(output_dir, model_file)
        model = JointBERT(args, DEVICE, dataloader.tag_dim, dataloader.intent_dim)
        model.load_state_dict(torch.load(os.path.join(output_dir, model_file), DEVICE))
        model.to(DEVICE)

        for name, param in model.named_parameters():
            print(name, param.shape, param.device, param.requires_grad)
        model.eval()

        self.model = model
        self.dataloader = dataloader
        self.cut_sen_len = config['cut_sen_len']
        self.context = args.context
        print("convlab2.nlu.jointBERT.crosswoz.nlu.BERTNLU#__init__ BERTNLU loaded")

    def predict(self, utterance, context=list(),intent_top=3,intent_score=0):
        batch_data = [self.dataloader.load_utterance(utterance, context, self.cut_sen_len)]
        pad_batch = self.dataloader.pad_batch(batch_data)
        pad_batch = tuple(t.to(self.model.device) for t in pad_batch)
        word_seq_tensor, tag_seq_tensor, intent_tensor, word_mask_tensor, tag_mask_tensor, context_seq_tensor, context_mask_tensor, \
        word_pos_tensor, word_vm_tensor, context_pos_tensor, context_vm_tensor = pad_batch
        if not self.context:
            context_seq_tensor, context_mask_tensor, context_pos_tensor, context_vm_tensor = None, None, None, None
        slot_logits, intent_logits,_,_ = self.model.forward(word_seq_tensor,word_mask_tensor,tag_seq_tensor,tag_mask_tensor,intent_tensor, \
                                                                               context_seq_tensor,context_mask_tensor, word_pos_tensor, \
                                                                               word_vm_tensor, context_pos_tensor,context_vm_tensor)
        intent = recover_intent(self.dataloader, intent_logits[0], slot_logits[0], tag_mask_tensor[0],batch_data[0][0], None,intent_top,intent_score)
        #print("BERTNLU: {},intent:{}".format(utterance,intent))
        print(f"BERTNLU:{utterance},intent:{intent}")
        return intent


if __name__ == '__main__':
    intent_top = 2
    intent_score = 0

    #nlu = BERTNLU()
    #nlu = BERTNLU(mode='usr',config_file='crosswoz_usr_context_kg.json', model_file='pytorch_model.bin')
    #nlu = BERTNLU(config_file='crosswoz_all_context.json', model_file='pytorch_model_context.bin')
    nlu = BERTNLU(mode='usr',config_file='crosswoz_usr_context.json',model_file='pytorch_model_context.bin')
    print(nlu.predict("天安门在哪", [],intent_top,intent_score))
    print(nlu.predict("天安门广场在哪", [],intent_top,intent_score))
    print(nlu.predict("天安门在哪?", [],intent_top,intent_score))
    print(nlu.predict("天安门广场在哪?", [], intent_top, intent_score))
    print(nlu.predict("天安门在哪里", [],intent_top,intent_score))
    print(nlu.predict("天安门广场在哪里", [], intent_top, intent_score))
    print(nlu.predict("天安门在哪里？", [],intent_top,intent_score))
    print(nlu.predict("天安门广场在哪里？", [], intent_top, intent_score))
    #print(nlu.predict("北京布提克精品酒店酒店是什么类型，有健身房吗？", []))
    #print(nlu.predict("北京布提克精品酒店酒店是什么类型，有健身房吗？", ['你好，给我推荐一个评分是5分，价格在100-200元的酒店。', '推荐您去北京布提克精品酒店。']))
