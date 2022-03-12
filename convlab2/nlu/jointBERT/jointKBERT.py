import torch
from torch import nn
from convlab2.nlu.jointBERT.uer.model_builder import build_model
import argparse

class JointBERT(nn.Module):
    def __init__(self, args, device, slot_dim, intent_dim, intent_weight=None):
        super(JointBERT, self).__init__()

        #args.target = "bert"
        model = build_model(args)
        model.load_state_dict(torch.load(args.pretrained_model_path, device), strict=False)
        print(f"JointKBERT] pretrained_model_path={args.pretrained_model_path},device={device}")
        self.bert = model
        self.bert_config_hidden_size = args.hidden_size

        self.embedding = model.embedding
        self.encoder = model.encoder
        #self.labels_num = args.labels_num
        #self.pooling = args.pooling
        #self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        #self.output_layer_2 = nn.Linear(args.hidden_size, args.labels_num)
        #self.softmax = nn.LogSoftmax(dim=-1)
        #self.criterion = nn.NLLLoss()
        self.use_vm = False if args.no_vm else True
        print("JointKBERT] use visible_matrix: {}".format(self.use_vm))

        self.pooler = BertPooler(self.bert_config_hidden_size)

        self.slot_num_labels = slot_dim
        self.intent_num_labels = intent_dim
        self.device = device
        self.intent_weight = intent_weight if intent_weight is not None else torch.tensor([1.]*intent_dim)

        self.dropout = nn.Dropout(args.dropout)
        self.context = args.context
        self.finetune = args.finetune
        self.context_grad = args.context_grad
        self.app_hidden_units = args.app_hidden_units

        if self.app_hidden_units > 0:
            if self.context:
                self.intent_classifier = nn.Linear(self.app_hidden_units, self.intent_num_labels)
                self.slot_classifier = nn.Linear(self.app_hidden_units, self.slot_num_labels)
                self.intent_hidden = nn.Linear(2 * self.bert_config_hidden_size, self.app_hidden_units)
                self.slot_hidden = nn.Linear(2 * self.bert_config_hidden_size, self.app_hidden_units)
            else:
                self.intent_classifier = nn.Linear(self.app_hidden_units, self.intent_num_labels)
                self.slot_classifier = nn.Linear(self.app_hidden_units, self.slot_num_labels)
                self.intent_hidden = nn.Linear(self.bert_config_hidden_size, self.app_hidden_units)
                self.slot_hidden = nn.Linear(self.bert_config_hidden_size, self.app_hidden_units)
            nn.init.xavier_uniform_(self.intent_hidden.weight)
            nn.init.xavier_uniform_(self.slot_hidden.weight)
        else:
            if self.context:
                self.intent_classifier = nn.Linear(2 * self.bert_config_hidden_size, self.intent_num_labels)
                self.slot_classifier = nn.Linear(2 * self.bert_config_hidden_size, self.slot_num_labels)
            else:
                self.intent_classifier = nn.Linear(self.bert_config_hidden_size, self.intent_num_labels)
                self.slot_classifier = nn.Linear(self.bert_config_hidden_size, self.slot_num_labels)
        nn.init.xavier_uniform_(self.intent_classifier.weight)
        nn.init.xavier_uniform_(self.slot_classifier.weight)

        self.intent_loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=self.intent_weight)
        self.slot_loss_fct = torch.nn.CrossEntropyLoss()

    def forward(self, word_seq_tensor, word_mask_tensor, tag_seq_tensor=None, tag_mask_tensor=None,
                intent_tensor=None, context_seq_tensor=None, context_mask_tensor=None,
                word_pos=None, word_vm=None,context_pos=None, context_vm=None):


        if not self.finetune:
            self.bert.eval()
            with torch.no_grad():
                outputs = self.kbert(word_seq_tensor,word_mask_tensor,word_pos, word_vm)
        else:
            outputs = self.kbert(word_seq_tensor,word_mask_tensor,word_pos, word_vm)

        sequence_output = outputs[0]
        pooled_output = outputs[1]

        if self.context and (context_seq_tensor is not None):
            if not self.finetune or not self.context_grad:
                with torch.no_grad():
                    context_output = self.kbert(context_seq_tensor, context_mask_tensor,context_pos, context_vm)[1]
            else:
                context_output = self.kbert(context_seq_tensor, context_mask_tensor,context_pos, context_vm)[1]
            sequence_output = torch.cat(
                [context_output.unsqueeze(1).repeat(1, sequence_output.size(1), 1),
                 sequence_output], dim=-1)
            pooled_output = torch.cat([context_output, pooled_output], dim=-1)

        if self.app_hidden_units > 0:
            if self.finetune:
                sequence_output = nn.functional.relu(self.slot_hidden(self.dropout(sequence_output)))
                pooled_output = nn.functional.relu(self.intent_hidden(self.dropout(pooled_output)))
            else:
                sequence_output = nn.functional.relu(self.slot_hidden(sequence_output))
                pooled_output = nn.functional.relu(self.intent_hidden(pooled_output))

        if self.finetune:
            sequence_output = self.dropout(sequence_output)
        slot_logits = self.slot_classifier(sequence_output)
        outputs = (slot_logits,)

        if self.finetune:
            pooled_output = self.dropout(pooled_output)
        intent_logits = self.intent_classifier(pooled_output)
        outputs = outputs + (intent_logits,)

        if tag_seq_tensor is not None:
            active_tag_loss = tag_mask_tensor.view(-1) == 1
            active_tag_logits = slot_logits.view(-1, self.slot_num_labels)[active_tag_loss]
            active_tag_labels = tag_seq_tensor.view(-1)[active_tag_loss]
            '''
            print("tag_seq_tensor.view(-1): len=", len(tag_seq_tensor.view(-1)))
            print(tag_seq_tensor.view(-1))
            print("tag_mask_tensor: len=", len(tag_mask_tensor))
            print(tag_mask_tensor)
            print("slot_logits.view(-1, self.slot_num_labels): len=", len(slot_logits.view(-1, self.slot_num_labels)))
            print(slot_logits.view(-1, self.slot_num_labels))
            print("active_tag_loss: len=",len(active_tag_loss))
            print(active_tag_loss)
            print("active_tag_labels:len=",len(active_tag_labels))
            print(active_tag_labels)
            print("active_tag_logits:len=",len(active_tag_logits))
            print(active_tag_logits)
            '''
            slot_loss = self.slot_loss_fct(active_tag_logits, active_tag_labels)
            outputs = outputs + (slot_loss,)

        if intent_tensor is not None:
            intent_loss = self.intent_loss_fct(intent_logits, intent_tensor)
            outputs = outputs + (intent_loss,)

        return outputs  # slot_logits, intent_logits, (slot_loss), (intent_loss),

    def kbert(self, src, mask, pos=None, vm=None):
        """
        Args:
            src: [batch_size x seq_length]
            label: [batch_size]
            mask: [batch_size x seq_length]
        """
        # Embedding.
        emb = self.embedding(src, mask, pos)
        # Encoder.
        if not self.use_vm:
            vm = None
        sequence_output = self.encoder(emb, mask, vm)
        pooled_output = self.pooler(sequence_output)

        return (sequence_output, pooled_output)

class BertPooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output