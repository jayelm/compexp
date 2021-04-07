"""
Wrappers around ONMT data loading pipeline
"""

import codecs
import data
import onmt
import onmt.inputters as inputters
from onmt.translate.translator import Translator
from onmt.utils.misc import split_corpus, use_gpu
from onmt.utils.parse import ArgumentParser
from onmt.model_builder import build_base_model
import torch
import settings


def ltm(opt, model_path=None):
    if model_path is None:
        model_path = opt.models[0]
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)

    model_opt = ArgumentParser.ckpt_model_opts(checkpoint["opt"])
    ArgumentParser.update_model_opts(model_opt)
    ArgumentParser.validate_model_opts(model_opt)
    vocab = checkpoint["vocab"]
    if inputters.old_style_vocab(vocab):
        fields = inputters.load_old_vocab(
            vocab, opt.data_type, dynamic_dict=model_opt.copy_attn
        )
    else:
        fields = vocab

    # This will randomly initialize
    if settings.RANDOM_WEIGHTS:
        checkpoint = None
    model = build_base_model(model_opt, fields, use_gpu(opt), checkpoint, opt.gpu)
    if opt.fp32:
        model.float()
    model.eval()
    model.generator.eval()
    return fields, model, model_opt


def build_translator(opt, report_score=True, logger=None, out_file=None):
    if out_file is None:
        out_file = codecs.open(opt.output, "w+", "utf-8")

    load_test_model = (
        onmt.decoders.ensemble.load_test_model if len(opt.models) > 1 else ltm
    )
    fields, model, model_opt = load_test_model(opt)

    scorer = onmt.translate.GNMTGlobalScorer.from_opt(opt)

    translator = Translator.from_opt(
        model,
        fields,
        opt,
        model_opt,
        global_scorer=scorer,
        out_file=out_file,
        report_align=opt.report_align,
        report_score=report_score,
        logger=logger,
    )
    return translator


def load_from_opt(opt):
    translator = build_translator(opt, report_score=True)
    model = translator.model
    if settings.CUDA:
        model = model.cuda()
    fields = translator.fields
    src_shards = split_corpus(opt.src, opt.shard_size)
    # only first shard for now
    src_shard = next(src_shards)

    dataset = data.analysis.AnalysisDataset(src_shard, fields)
    return model, dataset, fields, translator
