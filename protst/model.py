"""
Modifications copyright (C) 2025 LiShuang-mk<lishuang.mk@whu.edu.cn>
"""

import os
import warnings

import torch
from torch import nn
from torch.nn import functional as F

from torchdrug import core, layers, models, utils
from torchdrug.layers import functional
from torchdrug.core import Registry as R

from transformers import BertForMaskedLM, BertTokenizer, BertModel, pipeline
from torch_scatter import scatter_add

from torchdrug.layers.functional import variadic_to_padded
from protst import layer, data


@R.register("models.PubMedBERT")
class PubMedBERT(nn.Module, core.Configurable):
    """
    PubMedBERT encodes text description for proteins, starting
    from the pretrained weights provided in `https://microsoft.github.io/BLURB/models.html`

    Parameters:
        model (string, optional): model name. Available model names are ``PubMedBERT-abs``
            and ``PubMedBERT-full``. They differentiate from each other by training corpus,
            i.e., abstract only or abstract + full text.

    """

    def __init__(
        self,
        model_path,
        local_files_only=True,
        output_dim=512,
        num_mlp_layer=2,
        activation="relu",
        readout="mean",
        attribute=["prot_name", "function", "subloc", "similarity"],
    ):
        super(PubMedBERT, self).__init__()

        _model = eval("BertForMaskedLM").from_pretrained(
            model_path, local_files_only=local_files_only
        )
        _tokenizer = eval("BertTokenizer").from_pretrained(
            model_path, local_files_only=local_files_only
        )

        self.last_hidden_dim = 768
        self.output_dim = output_dim
        self.num_mlp_layer = num_mlp_layer

        self.model = _model
        self.tokenizer = _tokenizer
        self.pad_idx = self.tokenizer.pad_token_id
        self.sep_idx = self.tokenizer.sep_token_id
        self.cls_idx = self.tokenizer.cls_token_id
        self.mask_idx = self.tokenizer.mask_token_id

        self.activation = activation
        self.readout = readout
        self.attribute = attribute

        self.text_mlp = layers.MLP(
            self.last_hidden_dim,
            [self.last_hidden_dim] * (self.num_mlp_layer - 1) + [self.output_dim],
            activation=self.activation,
        )
        self.word_mlp = layers.MLP(
            self.last_hidden_dim,
            [self.last_hidden_dim] * (self.num_mlp_layer - 1) + [self.output_dim],
            activation=self.activation,
        )

    def _combine_attributes(self, graph, version=0):

        num_sample = len(graph)
        cls_ids = (
            torch.ones(num_sample, dtype=torch.long, device=self.device).unsqueeze(1)
            * self.cls_idx
        )
        sep_ids = (
            torch.ones(num_sample, dtype=torch.long, device=self.device).unsqueeze(1)
            * self.sep_idx
        )

        if version == 0:
            # [CLS] attr1 [PAD] ... [PAD] [SEP] attr2 [PAD] ... [PAD] [SEP] attrn [PAD] ... [PAD]
            input_ids = [cls_ids]
            for k in self.attribute:
                input_ids.append(graph.data_dict[k].long())
                input_ids.append(sep_ids)
            input_ids = torch.cat(input_ids[:-1], dim=-1)

        else:
            raise NotImplementedError

        return input_ids, input_ids != self.pad_idx

    def forward(
        self, graph, all_loss=None, metric=None, input_ids=None, attention_mask=None
    ):
        if input_ids is None or attention_mask is None:
            input_ids, attention_mask = self._combine_attributes(graph)
        model_inputs = {
            "input_ids": input_ids,
            "token_type_ids": torch.zeros_like(input_ids),
            "attention_mask": attention_mask,
        }
        model_outputs = self.model.bert(**model_inputs)
        if self.readout == "mean":
            is_special = (
                (input_ids == self.cls_idx)
                | (input_ids == self.sep_idx)
                | (input_ids == self.pad_idx)
            )
            text_mask = (~is_special).to(torch.float32).unsqueeze(-1)
            output = (model_outputs.last_hidden_state * text_mask).sum(1) / (
                text_mask.sum(1) + 1.0e-6
            )
        elif self.readout == "cls":
            output = model_outputs.last_hidden_state[:, 0, :]
        else:
            raise NotImplementedError

        output = self.text_mlp(output)
        word_output = self.word_mlp(model_outputs.last_hidden_state)

        return {"text_feature": output, "word_feature": word_output}


@R.register("models.PretrainESM")
class PretrainEvolutionaryScaleModeling(models.ESM):
    """
    Enable to pretrain ESM with MLM.
    """

    def __init__(
        self,
        path,
        model="ESM-1b",
        output_dim=512,
        num_mlp_layer=2,
        activation="relu",
        readout="mean",
        mask_modeling=False,
        use_proj=True,
    ):
        super(PretrainEvolutionaryScaleModeling, self).__init__(path, model, readout)
        self.mask_modeling = mask_modeling

        self.last_hidden_dim = self.output_dim
        self.output_dim = output_dim if use_proj else self.last_hidden_dim
        self.num_mlp_layer = num_mlp_layer
        self.activation = activation
        self.use_proj = use_proj

        self.graph_mlp = layers.MLP(
            self.last_hidden_dim,
            [self.last_hidden_dim] * (num_mlp_layer - 1) + [output_dim],
            activation=self.activation,
        )
        self.residue_mlp = layers.MLP(
            self.last_hidden_dim,
            [self.last_hidden_dim] * (num_mlp_layer - 1) + [output_dim],
            activation=self.activation,
        )

    def forward(self, graph, input, all_loss=None, metric=None):
        """
        Compute the residue representations and the graph representation(s).
        """
        input = graph.residue_type
        if self.mask_modeling:
            non_mask = ~(input == self.alphabet.mask_idx)
            input[non_mask] = self.mapping[input[non_mask]]
        else:
            input = self.mapping[input]
        size = graph.num_residues
        if (size > self.max_input_length).any():
            warnings.warn(
                "ESM can only encode proteins within %d residues. Truncate the input to fit into ESM."
                % self.max_input_length
            )
            starts = size.cumsum(0) - size
            size = size.clamp(max=self.max_input_length)
            ends = starts + size
            mask = functional.multi_slice_mask(starts, ends, graph.num_residue)
            input = input[mask]
            graph = graph.subresidue(mask)
        size_ext = size
        if self.alphabet.prepend_bos:
            bos = (
                torch.ones(graph.batch_size, dtype=torch.long, device=self.device)
                * self.alphabet.cls_idx
            )
            input, size_ext = functional._extend(
                bos, torch.ones_like(size_ext), input, size_ext
            )
        if self.alphabet.append_eos:
            eos = (
                torch.ones(graph.batch_size, dtype=torch.long, device=self.device)
                * self.alphabet.eos_idx
            )
            input, size_ext = functional._extend(
                input, size_ext, eos, torch.ones_like(size_ext)
            )
        input = functional.variadic_to_padded(
            input, size_ext, value=self.alphabet.padding_idx
        )[0]

        output = self.model(input, repr_layers=[33])
        residue_feature = output["representations"][33]

        residue_feature = functional.padded_to_variadic(residue_feature, size_ext)
        starts = size_ext.cumsum(0) - size_ext
        if self.alphabet.prepend_bos:
            starts = starts + 1
        ends = starts + size
        mask = functional.multi_slice_mask(starts, ends, len(residue_feature))
        residue_feature = residue_feature[mask]
        graph_feature = self.readout(graph, residue_feature)

        if self.use_proj:
            graph_feature = self.graph_mlp(graph_feature)
            residue_feature = self.residue_mlp(residue_feature)

        return {
            "graph_feature": graph_feature,
            "residue_feature": residue_feature,
        }


@R.register("models.CrossAttention")
class CrossAttention(nn.Module, core.Configurable):

    def __init__(
        self,
        hidden_dim=512,
        num_layers=1,
        num_heads=8,
        batch_norm=False,
        activation="relu",
    ):
        super(CrossAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.batch_norm = batch_norm

        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.layers.append(layer.CrossAttentionBlock(hidden_dim, num_heads))

        if batch_norm:
            self.protein_batch_norm_layers = nn.ModuleList()
            self.text_batch_norm_layers = nn.ModuleList()
            for _ in range(self.num_layers):
                self.protein_batch_norm_layers.append(nn.BatchNorm1d(hidden_dim))
                self.text_batch_norm_layers.append(nn.BatchNorm1d(hidden_dim))

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

    def forward(
        self, graph, protein_input, text_input, text_mask, all_loss=None, metric=None
    ):
        # Padding for protein inputs
        protein_input, protein_mask = functional.variadic_to_padded(
            protein_input, graph.num_residues, value=0
        )

        for i, layer in enumerate(self.layers):
            protein_input, text_input = layer(
                protein_input, text_input, protein_mask, text_mask
            )
            if self.batch_norm:
                protein_input = self.protein_batch_norm_layers[i](
                    protein_input.transpose(1, 2)
                ).transpose(1, 2)
                text_input = self.text_batch_norm_layers[i](
                    text_input.transpose(1, 2)
                ).transpose(1, 2)
            if self.activation:
                protein_input = self.activation(protein_input)
                text_input = self.activation(text_input)

        protein_output = functional.padded_to_variadic(
            protein_input, graph.num_residues
        )
        text_output = text_input

        return {"residue_feature": protein_output, "word_feature": text_output}


@R.register("models.OntoProtein")
class OntoProtein(nn.Module, core.Configurable):

    huggingface_card = "zjunlp/OntoProtein"
    output_dim = 1024

    def __init__(self, readout="pooler"):
        super(OntoProtein, self).__init__()
        model, tokenizer = self.load_weight()
        mapping = self.construct_mapping(tokenizer)
        self.model = model
        self.tokenizer = tokenizer
        self.register_buffer("mapping", mapping)

        if readout == "pooler":
            self.readout = None
        elif readout == "sum":
            self.readout = layers.SumReadout("residue")
        elif readout == "mean":
            self.readout = layers.MeanReadout("residue")
        else:
            raise ValueError("Unknown readout `%s`" % readout)

    def load_weight(self):
        tokenizer = BertTokenizer.from_pretrained(self.huggingface_card)
        model = BertModel.from_pretrained(self.huggingface_card)
        return model, tokenizer

    def construct_mapping(self, tokenizer):
        mapping = [0] * len(data.Protein.id2residue_symbol)
        for i, token in data.Protein.id2residue_symbol.items():
            mapping[i] = tokenizer._convert_token_to_id(token)
        mapping = torch.tensor(mapping)
        return mapping

    def forward(self, graph, input, all_loss=None, metric=None):
        input = graph.residue_type
        input = self.mapping[input]
        size = graph.num_residues
        size_ext = size
        bos = (
            torch.ones(graph.batch_size, dtype=torch.long, device=self.device)
            * self.tokenizer.cls_token_id
        )
        input, size_ext = functional._extend(
            bos, torch.ones_like(size_ext), input, size_ext
        )
        eos = (
            torch.ones(graph.batch_size, dtype=torch.long, device=self.device)
            * self.tokenizer.sep_token_id
        )
        input, size_ext = functional._extend(
            input, size_ext, eos, torch.ones_like(size_ext)
        )
        input = functional.variadic_to_padded(
            input, size_ext, value=self.tokenizer.pad_token_id
        )[0]

        output = self.model(input)
        residue_feature = output.last_hidden_state
        graph_feature = output.pooler_output

        residue_feature = functional.padded_to_variadic(residue_feature, size_ext)
        starts = size_ext.cumsum(0) - size_ext
        starts = starts + 1
        ends = starts + size
        mask = functional.multi_slice_mask(starts, ends, len(residue_feature))
        residue_feature = residue_feature[mask]
        if self.readout:
            graph_feature = self.readout(graph, residue_feature)

        return {"graph_feature": graph_feature, "residue_feature": residue_feature}


@R.register("models.PretrainProtBert")
class PretrainProtBert(nn.Module, core.Configurable):
    """
    Enable to pretrain ProtBert with MLM.
    """

    url = "https://zenodo.org/record/4633647/files/prot_bert_bfd.zip"
    md5 = "30fad832a088eb879e0ff88fa70c9655"
    last_hidden_dim = 1024

    def __init__(
        self,
        path,
        output_dim=512,
        num_mlp_layer=2,
        activation="relu",
        readout="pooler",
        mask_modeling=False,
        use_proj=True,
    ):
        super(PretrainProtBert, self).__init__()
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        model, tokenizer = self.load_weight(path)
        mapping = self.construct_mapping(tokenizer)
        self.model = model
        self.tokenizer = tokenizer
        self.register_buffer("mapping", mapping)

        if readout == "pooler":
            self.readout = None
        elif readout == "sum":
            self.readout = layers.SumReadout("residue")
        elif readout == "mean":
            self.readout = layers.MeanReadout("residue")
        else:
            raise ValueError("Unknown readout `%s`" % readout)

        self.mask_modeling = mask_modeling
        self.output_dim = output_dim if use_proj else self.last_hidden_dim
        self.num_mlp_layer = num_mlp_layer
        self.activation = activation
        self.use_proj = use_proj

        self.graph_mlp = layers.MLP(
            self.last_hidden_dim,
            [self.last_hidden_dim] * (num_mlp_layer - 1) + [output_dim],
            activation=self.activation,
        )
        self.residue_mlp = layers.MLP(
            self.last_hidden_dim,
            [self.last_hidden_dim] * (num_mlp_layer - 1) + [output_dim],
            activation=self.activation,
        )

    def load_weight(self, path):
        zip_file = utils.download(self.url, path, md5=self.md5)
        model_path = os.path.join(utils.extract(zip_file), "prot_bert_bfd")
        tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=False)
        model = BertModel.from_pretrained(model_path)
        return model, tokenizer

    def construct_mapping(self, tokenizer):
        mapping = [0] * len(data.Protein.id2residue_symbol)
        for i, token in data.Protein.id2residue_symbol.items():
            mapping[i] = tokenizer._convert_token_to_id(token)
        mapping = torch.tensor(mapping)
        return mapping

    def forward(self, graph, input, all_loss=None, metric=None):
        input = graph.residue_type
        if self.mask_modeling:
            non_mask = ~(input == self.tokenizer.mask_token_id)
            input[non_mask] = self.mapping[input[non_mask]]
        else:
            input = self.mapping[input]
        size = graph.num_residues
        size_ext = size
        bos = (
            torch.ones(graph.batch_size, dtype=torch.long, device=self.device)
            * self.tokenizer.cls_token_id
        )
        input, size_ext = functional._extend(
            bos, torch.ones_like(size_ext), input, size_ext
        )
        eos = (
            torch.ones(graph.batch_size, dtype=torch.long, device=self.device)
            * self.tokenizer.sep_token_id
        )
        input, size_ext = functional._extend(
            input, size_ext, eos, torch.ones_like(size_ext)
        )
        input = functional.variadic_to_padded(
            input, size_ext, value=self.tokenizer.pad_token_id
        )[0]

        output = self.model(input)
        residue_feature = output.last_hidden_state
        graph_feature = output.pooler_output

        residue_feature = functional.padded_to_variadic(residue_feature, size_ext)
        starts = size_ext.cumsum(0) - size_ext
        starts = starts + 1
        ends = starts + size
        mask = functional.multi_slice_mask(starts, ends, len(residue_feature))
        residue_feature = residue_feature[mask]
        if self.readout:
            graph_feature = self.readout(graph, residue_feature)

        if self.use_proj:
            graph_feature = self.graph_mlp(graph_feature)
            residue_feature = self.residue_mlp(residue_feature)

        return {"graph_feature": graph_feature, "residue_feature": residue_feature}


@R.register("models.STC_HRNN")
class HierarchicalRNN(nn.Module, core.Configurable):
    pad_token = "_PAD"
    mask_token = "_UNK"
    token_dict = {
        "_PAD": 0,
        "_GO": 1,
        "_EOS": 2,
        "_UNK": 3,
        "A": 4,
        "R": 5,
        "N": 6,
        "D": 7,
        "C": 8,
        "Q": 9,
        "E": 10,
        "G": 11,
        "H": 12,
        "I": 13,
        "L": 14,
        "K": 15,
        "M": 16,
        "F": 17,
        "P": 18,
        "S": 19,
        "T": 20,
        "W": 21,
        "Y": 22,
        "V": 23,
        "X": 24,
        "U": 25,
        "O": 26,
        "B": 27,
        "Z": 28,
    }
    id_dict = {v: k for k, v in token_dict.items()}
    
    data_mask_id = len(data.Protein.id2residue_symbol)
    """
    torchdrug定义的Protein类已经将残基字典定义好了,
    并且0已经被氨基酸G给占用了,
    因此我们只能使用一个新加入的id号作为mask的id号;
    这个id在训练和推理过程中会被mapping映射到3(_UNK)上。
    """

    def __init__(
        self, hier_i, hier_j, output_dim=512, readout="mean", mask_modeling=False
    ):
        super(HierarchicalRNN, self).__init__()

        self.aminoAcid_embedding = nn.Embedding(len(self.token_dict), output_dim)
        mapping = self.construct_mapping(self.token_dict)
        self.register_buffer("mapping", mapping)

        self.hier_i = hier_i
        self.hier_j = hier_j
        self.all_length = hier_i * hier_j
        self.output_dim = output_dim

        self.gru0 = nn.GRU(output_dim, output_dim, batch_first=True)
        self.gru1 = nn.GRU(output_dim, output_dim, batch_first=True)

        if readout == "sum":
            self.readout = layers.SumReadout("residue")
        elif readout == "mean":
            self.readout = layers.MeanReadout("residue")
        else:
            raise ValueError("Unknown readout `%s`" % readout)

        self.mask_modeling = mask_modeling

    def construct_mapping(self, alphabet: dict):
        """
        这个函数生成一个映射, 将torchdrug定义的残基字典映射到alphabet定义的残基字典
        """
        mapping = [self.token_dict[self.pad_token]] * max(len(data.Protein.id2residue_symbol), len(alphabet))
        for i, token in data.Protein.id2residue_symbol.items():
            mapping[i] = alphabet[token]
        mapping[self.data_mask_id] = alphabet[self.mask_token]
        mapping = torch.tensor(mapping)
        return mapping

    def forward(self, graph, input, all_loss=None, metric=None):
        self.gru0.flatten_parameters()
        self.gru1.flatten_parameters()
        
        batch_size = graph.batch_size
        input = graph.residue_type
        input = self.mapping[input]

        # 截断
        size = graph.num_residues
        if (size > self.all_length).any():
            starts = size.cumsum(0) - size
            size = size.clamp(max=self.all_length)
            ends = starts + size
            mask = functional.multi_slice_mask(starts, ends, graph.num_residue)
            input = input[mask]
            graph = graph.subresidue(mask)
        # 填充
        size_ext = size
        device = input.device
        pad_sizes = torch.full_like(size_ext, self.all_length, device=device) - size_ext
        pad_tensor = torch.full(
            [pad_sizes.sum()], self.token_dict[self.pad_token], device=device
        )
        input, size = functional._extend(input, size_ext, pad_tensor, pad_sizes)
        input = input.reshape(batch_size, -1)

        residue_feature = self.aminoAcid_embedding(input)
        residue_feature = residue_feature.reshape(
            batch_size * self.hier_i, self.hier_j, self.output_dim
        )
        residue_feature, _ = self.gru0(residue_feature)
        residue_feature = residue_feature.reshape(
            batch_size * self.hier_j, self.hier_i, self.output_dim
        )
        residue_feature, _ = self.gru1(residue_feature)

        residue_feature = residue_feature.reshape(-1, self.output_dim)
        graph.num_residues = torch.full_like(graph.num_residues, residue_feature.shape[0] / batch_size)
        graph_feature = self.readout(graph, residue_feature)

        return {
            "graph_feature": graph_feature,
            "residue_feature": residue_feature,
        }


@R.register("models.STC_Transformer")
class Transformer(nn.Module, core.Configurable):
    pad_token = "_PAD"
    mask_token = "_UNK"
    token_dict = {
        "_PAD": 0,
        "_GO": 1,
        "_EOS": 2,
        "_UNK": 3,
        "A": 4,
        "R": 5,
        "N": 6,
        "D": 7,
        "C": 8,
        "Q": 9,
        "E": 10,
        "G": 11,
        "H": 12,
        "I": 13,
        "L": 14,
        "K": 15,
        "M": 16,
        "F": 17,
        "P": 18,
        "S": 19,
        "T": 20,
        "W": 21,
        "Y": 22,
        "V": 23,
        "X": 24,
        "U": 25,
        "O": 26,
        "B": 27,
        "Z": 28,
    }
    id_dict = {v: k for k, v in token_dict.items()}
    
    data_mask_id = len(data.Protein.residue_symbol2id)

    def __init__(self, output_dim=512, readout="mean", mask_modeling=False):
        super(Transformer, self).__init__()

        self.aminoAcid_embedding = nn.Embedding(len(self.token_dict), output_dim)
        mapping = self.construct_mapping(self.token_dict)
        self.register_buffer("mapping", mapping)

        encoder_layer = nn.TransformerEncoderLayer(output_dim, 8, output_dim, 0.5, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, 6)

        if readout == "sum":
            self.readout = layers.SumReadout("residue")
        elif readout == "mean":
            self.readout = layers.MeanReadout("residue")
        else:
            raise ValueError("Unknown readout `%s`" % readout)

        self.mask_modeling = mask_modeling

    def construct_mapping(self, alphabet):
        mapping = [-1] * max(len(data.Protein.id2residue_symbol) + 1, len(alphabet))
        for i, token in data.Protein.id2residue_symbol.items():
            mapping[i] = alphabet[token]
        mapping[self.data_mask_id] = alphabet[self.mask_token]
        mapping = torch.tensor(mapping)
        return mapping
    
    def forward(self, graph, input, all_loss=None, metric=None):
        batch_size = graph.batch_size
        input = graph.residue_type

        input = self.mapping[input]

        # 截断
        size = graph.num_residues
        if (size > self.all_length).any():
            starts = size.cumsum(0) - size
            size = size.clamp(max=self.all_length)
            ends = starts + size
            mask = functional.multi_slice_mask(starts, ends, graph.num_residue)
            input = input[mask]
            graph = graph.subresidue(mask)
        # 填充
        size_ext = size
        device = input.device
        pad_sizes = torch.full_like(size_ext, self.all_length, device=device) - size_ext
        pad_tensor = torch.full(
            [pad_sizes.sum()], self.token_dict[self.pad_token], device=device
        )
        input, size = functional._extend(input, size_ext, pad_tensor, pad_sizes)
        input = input.reshape(batch_size, -1)

        residue_feature = self.aminoAcid_embedding(input)
        residue_feature = self.transformer(residue_feature)

        residue_feature = residue_feature.reshape(-1, self.output_dim)
        graph.num_residues = torch.full_like(graph.num_residues, residue_feature.shape[0] / batch_size)
        graph_feature = self.readout(graph, residue_feature)

        return {
            "graph_feature": graph_feature,
            "residue_feature": residue_feature,
        }
